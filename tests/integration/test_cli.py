import json
import subprocess
import sys
from pathlib import Path

from defectfirst.cli import COMMANDS, main
from defectfirst.fixtures import create_fixture
from defectfirst.io import read_jsonl, write_jsonl
from defectfirst.review_ui import export_review


def test_all_script_entrypoints_have_help():
    root = Path(__file__).resolve().parents[2]
    for command in COMMANDS:
        path = root / "scripts" / (command.replace("-", "_") + ".py")
        result = subprocess.run(
            [sys.executable, str(path), "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "--config" in result.stdout


def test_dry_run_does_not_create_execution_outputs(tmp_path, capsys):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps({"model": {"backbone": "tiny_test"}, "test_only": True}), encoding="utf-8"
    )
    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--root",
                str(tmp_path),
                "--output-dir",
                "output",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "DRY_RUN" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_offline_review_exports_native_images_and_no_preselected_answers(tmp_path):
    create_fixture(tmp_path)
    rows = read_jsonl(tmp_path / "groups.jsonl")
    write_jsonl(tmp_path / "candidates.jsonl", rows)
    result = export_review({"candidates": "candidates.jsonl"}, tmp_path, tmp_path / "review")
    page = (tmp_path / "review/index.html").read_text(encoding="utf-8")
    assert result["status"] == "WAITING_HUMAN"
    assert '<option value="">Unanswered</option>' in page
    assert "Native defect" in page and "Native normal" in page
    assert (tmp_path / "review/group_0_M.png").is_file()
