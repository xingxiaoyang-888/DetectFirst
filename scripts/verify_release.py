"""Run actual pre-server checks and retain timestamped command evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from defectfirst.io import sha256, source_fingerprint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-dino",
        action="store_true",
        help="Use locally pinned official source with random test weights",
    )
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Validate installed FLUX API without downloading weights",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Choose a new evidence directory; old raw outputs must be preserved")
    output.mkdir(parents=True)
    source_sha = source_fingerprint()
    commands = [
        ("pip_check", [sys.executable, "-m", "pip", "check"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
        ("mypy", [sys.executable, "-m", "mypy"]),
        ("compile", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"]),
        (
            "pytest",
            [sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={output / 'tests.xml'}"],
        ),
        ("smoke", [sys.executable, "scripts/smoke_test.py", "--output-dir", str(output / "smoke")]),
    ]
    if args.include_generation:
        program = (
            "import inspect,json,torch,diffusers,transformers; from diffusers import FluxFillPipeline; "
            "p=inspect.signature(FluxFillPipeline.__call__); "
            "assert all(k in p.parameters for k in ['image','mask_image','prompt','height','width','num_inference_steps','guidance_scale','generator']); "
            "print(json.dumps({'torch':torch.__version__,'diffusers':diffusers.__version__,'transformers':transformers.__version__,'api':'PASS','weights_loaded':False}))"
        )
        commands.append(("flux_api", [sys.executable, "-c", program]))
    environment = dict(os.environ)
    environment["DEFECTFIRST_TEST_DINO"] = "1" if args.include_dino else "0"
    records = []
    for name, argv in commands:
        log_path = output / f"{name}.log"
        started = datetime.now(timezone.utc).isoformat()
        print(f"Running {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                argv,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        records.append(
            {
                "name": name,
                "argv": argv,
                "started_at": started,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": result.returncode,
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256(log_path),
            }
        )
        print(f"{name}: exit {result.returncode}", flush=True)
    unchanged = source_sha == source_fingerprint()
    report = {
        "status": "PASS" if unchanged and all(r["exit_code"] == 0 for r in records) else "FAIL",
        "source_sha256": source_sha,
        "source_unchanged_during_checks": unchanged,
        "commands": records,
        "official_dino_random_architecture_checked": args.include_dino,
        "generation_api_checked": args.include_generation,
        "unvalidated": [
            "CUDA execution and throughput",
            "FLUX weights and generated-image quality",
            "Official pretrained DINO on real data",
            "External upstream adapters and their reproduction",
            "Research performance conclusions",
        ],
    }
    write_json(output / "verification.json", report)
    print(
        json.dumps({"status": report["status"], "report": str(output / "verification.json")}),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
