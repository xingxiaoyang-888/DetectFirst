"""Atomic resource leases for the authorized small closed loop, separate from night."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from defectfirst.io import write_json

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUBMITTED", "SUSPENDED"}


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "reports"):
        raise ValueError("Pilot ledger must remain in project reports")
    with path.with_suffix(".json.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        try:
            yield state
        finally:
            state["updated_at_utc"] = now()
            write_json(path, state)


def refresh(state):
    if state["jobs"]:
        output = subprocess.check_output(
            [
                "sacct",
                "-j",
                ",".join(state["jobs"]),
                "--format=JobID,State,ElapsedRaw,ExitCode,NodeList",
                "-n",
                "-P",
            ],
            text=True,
            timeout=20,
        )
        for line in output.splitlines():
            fields = line.split("|")
            if fields[0] in state["jobs"]:
                state["jobs"][fields[0]].update(
                    state=fields[1].split()[0].rstrip("+"),
                    elapsed_seconds=int(fields[2]),
                    exit_code=fields[3],
                    node=fields[4],
                )
    state["round_GPU_seconds"] = sum(row["elapsed_seconds"] for row in state["jobs"].values())
    uncertain = [e for e in state["submissions"] if e["status"] in {"SUBMITTING", "UNCERTAIN"}]
    state["reserved_GPU_seconds"] = sum(
        max(0, row["limit_seconds"] - row["elapsed_seconds"])
        for row in state["jobs"].values()
        if row["state"] in ACTIVE
    ) + sum(e["limit_seconds"] for e in uncertain)
    state["unresolved_submissions"] = len(uncertain)


def submit(path, script, name, seconds, node=None):
    script = Path(script).resolve()
    if (
        not script.is_relative_to(ROOT / "reports")
        or not script.is_file()
        or not name.startswith("dfpilot-")
        or seconds < 1
    ):
        raise ValueError("Use an explicit project-owned pilot script/name/limit")
    with locked(path) as state:
        refresh(state)
        if state["unresolved_submissions"] or any(e["name"] == name for e in state["submissions"]):
            raise RuntimeError("Reconcile uncertain submissions; never blindly resubmit a name")
        if (
            state["round_GPU_seconds"] + state["reserved_GPU_seconds"] + seconds
            > state["GPU_seconds_limit"]
        ):
            raise RuntimeError("Pilot spent plus all worst-case reservations exceeds2cardhours")
        if (
            sum(row["state"] in ACTIVE for row in state["jobs"].values())
            >= state["active_card_limit"]
        ):
            raise RuntimeError("Current pilot parallel card limit reached")
        limit = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        argv = [
            "sbatch",
            "--parsable",
            "--account=p_p15016",
            "--qos=cpu-500_core-l40-8_card-a800-8_card",
            "--partition=L40",
            "--gres=gpu:l40:1",
            "--ntasks=1",
            "--cpus-per-task=6",
            "--mem=96G",
            f"--time={limit}",
            f"--job-name={name}",
            f"--chdir={ROOT}",
            "--exclude=gpu4020",
            f"--export=ALL,PILOT_RESOURCE_LEDGER={Path(path).resolve()}",
        ]
        if node:
            argv.append(f"--nodelist={node}")
        argv.append(str(script))
        event = dict(
            name=name, limit_seconds=seconds, status="SUBMITTING", argv=argv, submitted_at_utc=now()
        )
        state["submissions"].append(event)
        write_json(path, state)
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            event["status"] = "UNCERTAIN"
            raise
        event.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            event["status"] = "REJECTED"
            raise RuntimeError("Pilot sbatch rejected; durable receipt retained")
        job = result.stdout.strip().split(";")[0]
        if not job.isdecimal():
            event["status"] = "UNCERTAIN"
            raise RuntimeError("Unexpected job response; reservation retained")
        event.update(status="SUBMITTED", job_id=job)
        state["jobs"][job] = dict(
            name=name,
            state="SUBMITTED",
            limit_seconds=seconds,
            elapsed_seconds=0,
            script=str(script),
        )
        return dict(job_id=job, reserved_GPU_seconds=seconds)


def claim(parent, attempt):
    path = os.environ["PILOT_RESOURCE_LEDGER"]
    with locked(path) as state:
        refresh(state)
        job = os.environ["SLURM_JOB_ID"]
        if job not in state["jobs"] or len(state["generation_calls"]) >= 8:
            raise RuntimeError("Unrecorded GPU lease or8repair calls reached")
        if any(
            c["parent_id"] == parent and c["attempt"] == attempt for c in state["generation_calls"]
        ):
            raise RuntimeError("An attempt is immutable; no blind reroll")
        identifier = len(state["generation_calls"]) + 1
        state["generation_calls"].append(
            dict(
                id=identifier,
                parent_id=parent,
                attempt=attempt,
                job_id=job,
                status="STARTED",
                started_at_utc=now(),
            )
        )
        return identifier


def finish(identifier, status, details):
    with locked(os.environ["PILOT_RESOURCE_LEDGER"]) as state:
        call = next(row for row in state["generation_calls"] if row["id"] == identifier)
        call.update(status=status, details=details, ended_at_utc=now())


def execute(path, command):
    with locked(path) as state:
        refresh(state)
        row = state["jobs"][os.environ["SLURM_JOB_ID"]]
        allowed = min(
            row["limit_seconds"] - row["elapsed_seconds"] - 2,
            state["GPU_seconds_limit"] - state["round_GPU_seconds"],
        )
        row["execution_guard"] = dict(
            started_at_utc=now(), allowed_seconds=allowed, whole_child_process_group=True
        )
    if allowed <= 0:
        return 124
    child = subprocess.Popen(command, start_new_session=True)

    def stop(signum, frame=None):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if frame is not None:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return child.wait(timeout=allowed)
    except subprocess.TimeoutExpired:
        stop(signal.SIGTERM)
        return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    add = sub.add_parser("submit")
    add.add_argument("--script", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--seconds", type=int, required=True)
    add.add_argument("--node")
    run = sub.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "submit":
        print(json.dumps(submit(args.ledger, args.script, args.name, args.seconds, args.node)))
    elif args.mode == "status":
        with locked(args.ledger) as state:
            refresh(state)
            print(json.dumps(state, indent=2))
    else:
        return execute(
            args.ledger, args.command[1:] if args.command[:1] == ["--"] else args.command
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
