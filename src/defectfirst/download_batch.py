"""Preserve each download attempt before publishing a mutable latest receipt index."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from defectfirst.io import sha256, write_json


def run_batch(root: Path, config: Path, output: Path, latest: Path) -> int:
    root, config, output, latest = (p.resolve() for p in (root, config, output, latest))
    if output == latest or output in latest.parents or latest in output.parents:
        raise ValueError("Batch evidence and latest index must be separate directories")
    latest.parent.mkdir(parents=True, exist_ok=True)
    # Covers execution AND publication; a second batch cannot overwrite the working index.
    with FileLock(str(latest) + ".batch.lock", timeout=0):
        output.mkdir(parents=True, exist_ok=False)
        frozen = output / ("assets" + config.suffix)
        shutil.copyfile(config, frozen)
        receipts = output / "receipts"
        argv = [
            sys.executable,
            str(root / "scripts/download_assets.py"),
            "--root",
            str(root),
            "--config",
            str(frozen),
            "--output-dir",
            str(receipts),
        ]
        record = {
            "argv": argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "exit_code": None,
            "config_sha256": sha256(frozen),
            "log_path": "download.log",
            "receipt_files": {},
            "latest_published": False,
        }
        write_json(output / "command.json", record)
        try:
            with (output / "download.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(argv, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            record["exit_code"] = result.returncode
        except KeyboardInterrupt:
            record.update(exit_code=130, error="Interrupted; preserve partial receipts")
        except Exception as error:
            record.update(exit_code=1, error=f"{type(error).__name__}: {error}")
        finally:
            record["ended_at"] = datetime.now(timezone.utc).isoformat()
            record["receipt_files"] = {
                path.name: sha256(path) for path in sorted(receipts.glob("*.json"))
            }
            write_json(output / "command.json", record)
        # Original receipts already live in an exclusive per-attempt directory. Never
        # regenerate or alter them when later download attempts resume the asset cache.
        try:
            latest.mkdir(parents=True, exist_ok=True)
            for name in record["receipt_files"]:
                shutil.copyfile(receipts / name, latest / name)
            write_json(
                latest / "batch_index.json",
                {
                    "batch": str(output),
                    "exit_code": record["exit_code"],
                    "receipt_files": record["receipt_files"],
                },
            )
            record["latest_published"] = True
        except Exception as error:
            record["publication_error"] = f"{type(error).__name__}: {error}"
        write_json(output / "command.json", record)
        return int(record["exit_code"]) if record["latest_published"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New batch evidence directory"
    )
    parser.add_argument("--latest-dir", type=Path, default=Path("reports/S01_download"))
    args = parser.parse_args()
    return run_batch(args.root, args.config, args.output_dir, args.latest_dir)


if __name__ == "__main__":
    raise SystemExit(main())
