from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from defectfirst.io import canonical_bytes, read_json, sha256, within, write_json


def check_environment(output: Path, require_gpu: bool = True) -> dict:
    packages = {}
    for package in (
        "torch",
        "numpy",
        "Pillow",
        "scipy",
        "scikit-learn",
        "diffusers",
        "transformers",
        "huggingface-hub",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu, failures = [], []
    cuda_available, device_count = None, 0
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
    except Exception as error:
        failures.append(
            {"scope": "cuda_discovery", "error": str(error), "traceback": traceback.format_exc()}
        )
    for index in range(device_count):
        check = {"index": index, "status": "FAIL", "backward_passed": False}
        gpu.append(check)  # Preserve the device record even if initialization fails.
        try:
            properties = torch.cuda.get_device_properties(index)
            check.update(name=properties.name, total_bytes=properties.total_memory)
            device = torch.device(f"cuda:{index}")
            x = torch.ones(8, device=device, requires_grad=True)
            x.square().sum().backward()
            torch.cuda.synchronize(device)
            check["backward_passed"] = bool(torch.equal(x.grad, torch.full_like(x, 2)))
            if check["backward_passed"]:
                check["status"] = "PASS"
            else:
                check["error"] = "Gradient differs from the expected all-2 vector"
        except Exception as error:
            check.update(error=str(error), traceback=traceback.format_exc())
        if check["status"] == "FAIL":
            failures.append({"scope": f"cuda:{index}", "error": check["error"]})
    smi = None
    smi_record = {"status": "NOT_AVAILABLE"}
    if shutil.which("nvidia-smi"):
        argv = ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free", "--format=csv"]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, check=False)
            smi = process.stdout
            smi_record = {
                "status": "PASS" if process.returncode == 0 else "FAIL",
                "argv": argv,
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
            if process.returncode != 0:
                failures.append({"scope": "nvidia-smi", "error": process.stderr})
        except Exception as error:
            smi_record = {
                "status": "FAIL",
                "argv": argv,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append({"scope": "nvidia-smi", "error": str(error)})
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "FAIL" if failures else "PASS" if gpu or not require_gpu else "WAITING_RESOURCE",
        "cpu_check_only": not require_gpu,
        "require_gpu": require_gpu,
        "cuda_available": cuda_available,
        "visible_device_count": device_count,
        "failures": failures,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpus": gpu,
        "nvidia_smi": smi,
        "nvidia_smi_check": smi_record,
        "disk_free_bytes": shutil.disk_usage(output).free,
        "research_observation": "NOT_EVALUATED",
    }
    write_json(output / "env.json", result)
    return result


def validate_stage_report(path: Path, root: Path) -> dict:
    report = read_json(path)
    required = {
        "stage",
        "batch_id",
        "engineering_status",
        "research_observation",
        "code_commit",
        "config_sha256",
        "manifest_sha256",
        "commands",
        "criteria",
        "evidence_paths",
        "failed_items",
        "pending_human_review",
        "running_jobs",
        "next_action",
    }
    if not required <= report.keys():
        raise ValueError(f"Stage report missing fields: {required - report.keys()}")
    for relative in report["evidence_paths"]:
        if not within(root, relative).is_file():
            raise ValueError(f"Missing evidence: {relative}")
    if report["engineering_status"] == "PASS":
        if not report["commands"] or not report["criteria"] or not report["evidence_paths"]:
            raise ValueError("PASS requires executed commands, criteria and actual evidence files")
        if report["failed_items"] or report["pending_human_review"]:
            raise ValueError("PASS contradicts unresolved failures/reviews")
        for command in report["commands"]:
            if (
                command.get("exit_code") != 0
                or not command.get("started_at")
                or not command.get("ended_at")
            ):
                raise ValueError("PASS requires successful timestamped commands")
            if not within(root, command["log_path"]).is_file():
                raise ValueError("Missing raw command log")
        for criterion in report["criteria"]:
            if (
                criterion.get("passed") is not True
                or not within(root, criterion["evidence_path"]).is_file()
            ):
                raise ValueError("Criterion has no passing evidence")
    return report


def create_handoff(
    root: Path, paths: list[str], output: Path, max_bytes: int = 20 * 1024**2
) -> dict:
    if output.exists():
        raise FileExistsError("Use a new timestamped handoff package")
    files = []
    for relative in paths:
        path = within(root, relative)
        if not path.is_file():
            raise ValueError(f"Handoff path is not a file: {relative}")
        if any(
            part in {".git", ".venv", ".cache", "models_cache", "data"}
            for part in path.relative_to(root).parts
        ):
            raise ValueError(
                "Select a small exported example instead of raw data/model/cache files"
            )
        if path.name.startswith(".env") or any(
            word in path.name.lower() for word in ("token", "cookie", "credential")
        ):
            raise ValueError("Credential files must not enter handoff")
        files.append(path)
    if sum(p.stat().st_size for p in files) > max_bytes:
        raise ValueError("Handoff exceeds size budget; create separate evidence packages")
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("checksums.json", canonical_bytes(checksums))
    return {"path": str(output), "sha256": sha256(output), "files": len(files)}


def inference_benchmark(model, tensor: torch.Tensor, warmup: int = 20, repeats: int = 100) -> dict:
    if tensor.shape[0] != 1 or warmup < 0 or repeats < 1:
        raise ValueError("Benchmark uses batch 1 and positive repetitions")
    model.eval()
    device = tensor.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= warmup:
                values.append(time.perf_counter() - start)
    return {
        "scope": "network_forward_only",
        "batch_size": 1,
        "warmup": warmup,
        "repeats": repeats,
        "seconds": values,
        "p50_ms": float(np.median(values) * 1000),
        "p95_ms": float(np.quantile(values, 0.95) * 1000),
        "device": str(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
