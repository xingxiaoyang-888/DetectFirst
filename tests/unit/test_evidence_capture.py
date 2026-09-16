import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from defectfirst import download_batch
from defectfirst.io import read_json, write_json


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Server Bash regression")
def test_record_preserves_nonzero_exit_under_errexit_and_refuses_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/record.sh"
    program = 'set -e; source "$1"; BATCH=batch; STAGE=S00; record probe bash -c "exit 7"'
    result = subprocess.run(["bash", "-c", program, "test", str(script)], cwd=tmp_path)
    base = tmp_path / "logs/batch/S00/probe"
    assert result.returncode == 7
    assert (base / "exitcode").read_text().strip() == "7"
    assert (base / "ended").is_file()
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    repeat = subprocess.run(["bash", "-c", program, "test", str(script)], cwd=tmp_path)
    assert repeat.returncode == 73
    assert {p.name: p.read_bytes() for p in base.iterdir()} == before


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Server Bash regression")
def test_parallel_stage_labels_have_separate_evidence(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/record.sh"
    program = 'set -e; source "$1"; BATCH=batch; STAGE="$2"; record probe printf "%s" "$2"'
    processes = [
        subprocess.Popen(["bash", "-c", program, "test", str(script), stage], cwd=tmp_path)
        for stage in ("S00", "S01")
    ]
    assert [process.wait() for process in processes] == [0, 0]
    for stage in ("S00", "S01"):
        assert (tmp_path / f"logs/batch/{stage}/probe/output.log").read_text() == stage


def test_failed_download_receipts_survive_retry_and_latest_update(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"assets": []})
    calls = []

    def fake_run(argv, **kwargs):
        receipts = Path(argv[argv.index("--output-dir") + 1])
        calls.append(argv)
        write_json(receipts / "asset.lock.json", {"attempt": len(calls), "status": "FAIL"})
        write_json(receipts / "assets.lock.json", {"status": "INCOMPLETE"})
        kwargs["stdout"].write("failure evidence\n")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(download_batch.subprocess, "run", fake_run)
    latest = tmp_path / "latest"
    first, second = tmp_path / "first", tmp_path / "second"
    assert download_batch.run_batch(tmp_path, config, first, latest) == 2
    before = {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    assert download_batch.run_batch(tmp_path, config, second, latest) == 2
    assert {
        str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()
    } == before
    assert read_json(latest / "asset.lock.json")["attempt"] == 2
    assert read_json(first / "command.json")["exit_code"] == 2
    with pytest.raises(FileExistsError):
        download_batch.run_batch(tmp_path, config, first, latest)
    assert len(calls) == 2


def test_download_launch_exception_keeps_command_record(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"assets": []})

    def fail(*args, **kwargs):
        raise OSError("cannot start downloader")

    monkeypatch.setattr(download_batch.subprocess, "run", fail)
    output = tmp_path / "batch"
    assert download_batch.run_batch(tmp_path, config, output, tmp_path / "latest") == 1
    record = read_json(output / "command.json")
    assert record["ended_at"] and record["exit_code"] == 1
    assert "cannot start downloader" in record["error"]
