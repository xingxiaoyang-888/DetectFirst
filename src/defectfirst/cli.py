from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defectfirst.io import read_config, within, write_json, write_jsonl

COMMANDS = [
    "summarize-generation",
    "prepare-rois",
    "plan-generation",
    "compile-experiments",
    "check-env",
    "download-assets",
    "prepare-data",
    "generate-primitives",
    "propose-annotation",
    "build-crossed",
    "audit-controls",
    "export-review",
    "train",
    "calibrate",
    "evaluate",
    "recompute-metrics",
    "diagnose",
    "measure-resources",
    "aggregate",
    "validate-handoff",
    "run-external-baseline",
    "plan-experiments",
    "smoke-test",
    "audit-model",
    "extract-archive",
]


def dispatch(command: str, config: dict, root: Path, output: Path, args) -> dict:
    if command == "summarize-generation":
        from defectfirst.generation_report import summarize_generation

        return summarize_generation(config, root, output)
    if command in {"prepare-rois", "plan-generation", "compile-experiments"}:
        from defectfirst.planning import compile_experiments, plan_generation, prepare_rois

        return {
            "prepare-rois": prepare_rois,
            "plan-generation": plan_generation,
            "compile-experiments": compile_experiments,
        }[command](config, root, output)
    if command == "check-env":
        from defectfirst.quality import check_environment

        return check_environment(output, config.get("require_gpu", True))
    if command == "download-assets":
        from defectfirst.assets import download_all

        return download_all(config, root, output)
    if command == "prepare-data":
        from defectfirst.commands import prepare

        return prepare(config, root, output)
    if command == "generate-primitives":
        from defectfirst.controls.generation import generate

        return generate(config, root, output)
    if command == "propose-annotation":
        from defectfirst.commands import propose_annotation

        return propose_annotation(config, root, output)
    if command == "build-crossed":
        from defectfirst.commands import build_crossed

        return build_crossed(config, root, output)
    if command == "audit-controls":
        from defectfirst.commands import audit_controls

        return audit_controls(config, root, output)
    if command == "export-review":
        from defectfirst.review_ui import export_review

        return export_review(config, root, output)
    if command == "train":
        from defectfirst.config import TrainConfig
        from defectfirst.training.engine import train

        config["root"] = str(root)
        return train(TrainConfig.from_dict(config), output, args.resume, args.stop_after)
    if command == "calibrate":
        from defectfirst.evaluation.pipeline import calibrate_run

        return calibrate_run(config, root, output)
    if command == "evaluate":
        from defectfirst.evaluation.pipeline import evaluate_run

        return evaluate_run(config, root, output)
    if command == "recompute-metrics":
        from defectfirst.evaluation.pipeline import recompute

        result = recompute(config, root)
        write_json(output / "replay.json", result)
        return result
    if command == "diagnose":
        from defectfirst.server_checks import diagnose

        return diagnose(config, root, output)
    if command == "measure-resources":
        from defectfirst.server_checks import benchmark

        return benchmark(config, root, output)
    if command == "aggregate":
        from defectfirst.commands import aggregate

        return aggregate(config, root, output)
    if command == "run-external-baseline":
        from defectfirst.external import run_external

        return run_external(config, root, output)
    if command == "validate-handoff":
        from defectfirst.quality import create_handoff, validate_stage_report

        if not config.get("reports"):
            raise ValueError("Handoff validation requires at least one actual stage report")
        reports = [validate_stage_report(within(root, path), root) for path in config["reports"]]
        result = {"status": "PASS", "validated_stages": [r["stage"] for r in reports]}
        if config.get("files"):
            result["package"] = create_handoff(root, config["files"], output / "handoff.zip")
        write_json(output / "validation.json", result)
        return result
    if command == "plan-experiments":
        from defectfirst.protocol import experiment_matrix

        jobs = experiment_matrix(config.get("include_mechanisms", True))
        write_jsonl(output / "experiments.jsonl", jobs)
        return {"status": "PLANNED", "jobs": len(jobs), "executed": 0}
    if command == "smoke-test":
        from defectfirst.server_checks import smoke

        return smoke(output)
    if command == "audit-model":
        from defectfirst.server_checks import audit_model

        return audit_model(config, root, output)
    if command == "extract-archive":
        from defectfirst.assets import extract_archive

        extract_archive(within(root, config["archive"]), output)
        return {
            "status": "EXTRACTED",
            "validation": "Run prepare-data to verify image counts and labels",
        }
    raise ValueError(f"Unknown command {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DefectFirst: reproducible research tools")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--config", type=Path, help="YAML or JSON configuration; paths inside are project-relative"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project / artifact root")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/command"))
    parser.add_argument("--resume", action="store_true", help="Resume the same frozen training run")
    parser.add_argument(
        "--stop-after", type=int, help="Stop training at this absolute step, retaining resume state"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect configuration; do not execute or claim readiness",
    )
    args = parser.parse_args(argv)
    config = read_config(args.config) if args.config else {}
    if args.command not in {"check-env", "smoke-test", "plan-experiments"} and args.config is None:
        parser.error("this command requires --config")
    if args.resume and args.command != "train":
        parser.error(
            "--resume applies to training; downloads/generation resume their own immutable jobs"
        )
    root = args.root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else within(root, str(args.output_dir))
    )
    if args.dry_run:
        if args.command == "train":
            from defectfirst.config import TrainConfig

            TrainConfig.from_dict(config)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "command": args.command,
                    "config": config,
                    "root": str(root),
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        result = dispatch(args.command, config, root, output, args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return (
        0
        if result.get("status")
        not in {
            "FAIL",
            "INCOMPLETE",
            "WAITING_RESOURCE",
            "WAITING_ACCESS",
            "WAITING_HUMAN",
            "INSUFFICIENT_TIMINGS",
        }
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
