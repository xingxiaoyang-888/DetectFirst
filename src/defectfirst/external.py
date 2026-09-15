from __future__ import annotations

import subprocess
from pathlib import Path

from defectfirst.io import read_jsonl, sha256, within, write_json

SOURCES = {
    "E_SSN": "https://github.com/blaz-r/SuperSimpleNet",
    "E_AVFM": "https://github.com/MaticFuc/AnomalyVFM",
    "E_SEAS": "https://github.com/HUST-SLOW/SeaS",
    "E_O2MAG": "https://github.com/echrao/O2MAG",
}


def run_external(config: dict, root: Path, output: Path) -> dict:
    """Execute a reviewed upstream adapter, preserving its native implementation.

    An adapter must declare every target-anomaly use before execution. This
    runner is not itself a reproduction of the four upstream algorithms.
    """
    if config["method"] not in SOURCES:
        raise ValueError("Unknown external method")
    if config.get("adapter_status") != "REVIEWED":
        raise ValueError("External upstream adapter must be implemented and reviewed before launch")
    source = within(root, config["source_dir"])
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["source_commit"] or len(commit) != 40:
        raise ValueError("Upstream commit does not match the experiment lock")
    rows = read_jsonl(within(root, config["manifest"]))
    allowed = set(config["selected_support_ids"])
    by_id = {r["sample_id"]: r for r in rows}
    if any(by_id[s]["role"] != "anomaly_support_pool" for s in allowed):
        raise ValueError("Selected supports are not from the fixed anomaly pool")
    used = set()
    for field in (
        "gradient_anomaly_ids",
        "generator_reference_ids",
        "mask_training_ids",
        "embedding_training_ids",
    ):
        if field not in config:
            raise ValueError(f"Must disclose {field}, including an explicit empty list")
        used.update(config[field])
    if not used <= allowed:
        raise PermissionError("External method uses extra, calibration, or test anomalies")
    argv = config["command"]
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        raise ValueError(
            "External command must be an explicit argv list; shell execution is disabled"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "fairness.json",
        {
            **config,
            "actual_source_commit": commit,
            "unique_target_anomaly_ids": sorted(used),
            "manifest_sha256": sha256(within(root, config["manifest"])),
        },
    )
    with (output / "upstream.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            argv, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    result = {
        "method": config["method"],
        "exit_code": process.returncode,
        "status": "PROCESS_COMPLETED" if process.returncode == 0 else "FAIL",
        "research_observation": "NOT_EVALUATED",
        "metrics_validation": "REQUIRED",
    }
    write_json(output / "process.json", result)
    if process.returncode:
        raise RuntimeError(f"Upstream process exited {process.returncode}; see upstream.log")
    return result
