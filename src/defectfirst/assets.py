from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from filelock import FileLock

from defectfirst.io import read_json, sha256, within, write_json


def download_http(
    url: str, destination: Path, expected_hash: str | None = None, expected_bytes: int | None = None
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".download.lock", timeout=0):
        if destination.is_file():
            checksum = sha256(destination)
            if expected_hash and checksum != expected_hash:
                raise ValueError(f"Existing file checksum mismatch: {destination}")
            if expected_bytes and destination.stat().st_size != expected_bytes:
                raise ValueError("Existing file size mismatch")
            if not expected_hash:
                receipt_path = destination.with_suffix(destination.suffix + ".download.json")
                if not receipt_path.exists():
                    raise ValueError(
                        "Unverified preexisting download; inspect and register it explicitly"
                    )
                receipt = read_json(receipt_path)
                if receipt["sha256"] != checksum or receipt["url"] != url:
                    raise ValueError(
                        "Cached file or requested source changed since download verification"
                    )
            return {
                "path": str(destination),
                "sha256": checksum,
                "bytes": destination.stat().st_size,
                "official_hash_verified": expected_hash is not None,
                "cached": True,
            }
        partial = destination.with_suffix(destination.suffix + ".part")
        metadata_path = partial.with_suffix(partial.suffix + ".json")
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "DefectFirst/0.1"}
        if offset and metadata.get("url") == url and metadata.get("etag"):
            headers.update({"Range": f"bytes={offset}-", "If-Range": metadata["etag"]})
        else:
            offset = 0
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code == 416:
                raise ValueError(
                    "Server cannot resume this partial; inspect its metadata before retry"
                ) from error
            raise
        with response:
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an incorrect resume range")
            else:
                offset = 0  # A server ignoring Range must overwrite, never append.
            write_json(metadata_path, {"url": url, "etag": response.headers.get("ETag")})
            content_length = response.headers.get("Content-Length")
            with partial.open("ab" if offset else "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if content_length and partial.stat().st_size != offset + int(content_length):
                raise ValueError("Incomplete HTTP response; partial preserved")
        if expected_bytes and partial.stat().st_size != expected_bytes:
            raise ValueError("Downloaded byte count mismatch")
        checksum = sha256(partial)
        if expected_hash and checksum != expected_hash:
            raise ValueError("Downloaded checksum mismatch; partial preserved")
        os.replace(partial, destination)
        result = {
            "url": url,
            "path": str(destination),
            "sha256": checksum,
            "bytes": destination.stat().st_size,
            "official_hash_verified": expected_hash is not None,
            "cached": False,
        }
        write_json(destination.with_suffix(destination.suffix + ".download.json"), result)
        return result


def extract_archive(archive: Path, output: Path) -> None:
    """Extract regular files/directories only; reject traversal, links and overwrite."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Extract into a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                within(output, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not allowed")
            handle.extractall(output)
    else:
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                within(output, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive contains a link/device/non-regular member")
            handle.extractall(output)  # All member paths/types were validated above.


def download_asset(asset: dict, root: Path) -> dict:
    kind = asset["kind"]
    destination = within(root, asset["path"])
    if kind == "manual":
        return {
            "id": asset["id"],
            "status": "WAITING_ACCESS",
            "entry_url": asset["url"],
            "next_action": "Obtain the official dataset and register its archive path and checksum",
        }
    if kind == "http":
        result = download_http(asset["url"], destination, asset.get("sha256"), asset.get("bytes"))
    elif kind == "git":
        revision = asset["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Git source requires a full commit hash")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".download.lock", timeout=0):
            if not destination.exists():
                subprocess.run(["git", "init", str(destination)], check=True, capture_output=True)
                subprocess.run(
                    ["git", "-C", str(destination), "remote", "add", "origin", asset["url"]],
                    check=True,
                )
            remote = subprocess.check_output(
                ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
            ).strip()
            if remote != asset["url"]:
                raise ValueError("Existing source checkout has a different origin")
            subprocess.run(
                ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
            dirty = subprocess.check_output(
                ["git", "-C", str(destination), "status", "--porcelain"], text=True
            )
            if dirty:
                raise ValueError("Refusing to change a modified source checkout")
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", revision], check=True
            )
            result = {"path": asset["path"], "revision": revision}
    elif kind == "huggingface":
        from huggingface_hub import snapshot_download

        if not re.fullmatch(r"[0-9a-f]{40}", asset["revision"]):
            raise ValueError("Model revision must be an immutable Hub commit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".download.lock", timeout=0):
            snapshot_download(
                repo_id=asset["repo_id"],
                revision=asset["revision"],
                local_dir=destination,
                allow_patterns=asset.get("allow_patterns"),
                max_workers=2,
            )
            index = read_json(destination / "model_index.json")
            for component, component_type in index.items():
                if (
                    not component.startswith("_")
                    and isinstance(component_type, list)
                    and component_type[0] is not None
                ):
                    if not (destination / component).is_dir():
                        raise ValueError(f"Missing pipeline component: {component}")
            result = {
                "path": asset["path"],
                "revision": asset["revision"],
                "files": {
                    p.relative_to(destination).as_posix(): {
                        "bytes": p.stat().st_size,
                        "sha256": sha256(p),
                    }
                    for p in destination.rglob("*")
                    if p.is_file() and ".cache" not in p.parts
                },
            }
    else:
        raise ValueError(f"Unknown asset kind: {kind}")
    return {"id": asset["id"], "status": "READY", **result}


def download_all(config: dict, root: Path, output: Path, dry_run: bool = False) -> dict:
    assets = config["assets"]
    if len({a["id"] for a in assets}) != len(assets):
        raise ValueError("Duplicate asset IDs")
    if len({a["path"] for a in assets}) != len(assets):
        raise ValueError("Multiple asset jobs cannot share a destination")
    if dry_run:
        return {"status": "DRY_RUN", "assets": assets}
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.get("max_workers", 2)
    ) as executor:
        futures = {executor.submit(download_asset, asset, root): asset for asset in assets}
        for future in concurrent.futures.as_completed(futures):
            asset = futures[future]
            try:
                result = future.result()
            except Exception as error:
                # Do not include environment variables or credentials in logs.
                result = {
                    "id": asset["id"],
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            results.append(result)
            write_json(output / f"{asset['id']}.lock.json", result)
            write_json(
                output / "download_status.json",
                {"assets": results, "complete": len(results) == len(assets)},
            )
    result = {
        "assets": results,
        "status": "READY" if all(r["status"] == "READY" for r in results) else "INCOMPLETE",
    }
    write_json(output / "assets.lock.json", result)
    return result
