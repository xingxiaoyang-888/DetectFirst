"""Server-only atomic night accounting, absolute cutoff and project-owned Slurm jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUBMITTED", "SUSPENDED"}


class BudgetStop(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc)


def deadline(state):
    return datetime.fromisoformat(state["deadline_utc"].replace("Z", "+00:00"))


def cutoff(state):
    return deadline(state) - timedelta(seconds=state.get("cutoff_safety_seconds", 0))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "reports"):
        raise ValueError("Ledger must be in this project reports directory")
    with path.with_suffix(path.suffix + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        try:
            yield state
        finally:
            state["updated_at"] = utc().isoformat()
            atomic(path, state)


def refresh(state):
    if state["jobs"]:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(state["jobs"]),
                "--format=JobID,State,ElapsedRaw,ExitCode,NodeList",
                "-n",
                "-P",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
        )
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if fields[0] in state["jobs"]:
                job = state["jobs"][fields[0]]
                job.update(
                    state=fields[1].split()[0].rstrip("+"),
                    elapsed_seconds=int(fields[2]),
                    exit_code=fields[3],
                    node=fields[4],
                )
    spent = sum(job["elapsed_seconds"] * job["cards"] for job in state["jobs"].values())
    reserved = sum(
        max(0, job["limit_seconds"] - job["elapsed_seconds"]) * job["cards"]
        for job in state["jobs"].values()
        if job["state"] in ACTIVE
    )
    uncertain = [
        event
        for event in state["submission_receipts"]
        if event["status"] in {"SUBMITTING", "UNCERTAIN"}
    ]
    reserved += sum(event["reserved_GPU_seconds"] for event in uncertain)
    state.update(
        night_GPU_seconds=spent,
        active_reserved_GPU_seconds=reserved,
        project_GPU_seconds=state["window_baseline_GPU_seconds"] + spent,
        development_calls=state["window_baseline_development_calls"] + len(state["calls"]),
        unresolved_submissions=len(uncertain),
    )
    return state


def available(state, seconds, cards=1):
    if state.get("unresolved_submissions", 0):
        raise BudgetStop("RECONCILE_UNCERTAIN_SUBMISSION_BEFORE_ANOTHER_JOB")
    if utc() >= cutoff(state):
        raise BudgetStop("ABSOLUTE_DEADLINE_REACHED")
    if (
        state["night_GPU_seconds"] + state["active_reserved_GPU_seconds"] + seconds * cards
        > state["night_GPU_seconds_limit"]
    ):
        raise BudgetStop("GPU_BUDGET_WITH_ACTIVE_RESERVATIONS_EXCEEDED")
    active_cards = sum(job["cards"] for job in state["jobs"].values() if job["state"] in ACTIVE)
    if active_cards + cards > state["active_card_limit"]:
        raise BudgetStop("CURRENT_PARALLEL_CARD_LIMIT_EXCEEDED")


def owned_job(job_id, expected_name):
    result = subprocess.run(
        ["scontrol", "show", "job", str(job_id), "-o"], text=True, capture_output=True, timeout=20
    )
    if result.returncode:
        return None
    values = dict(part.split("=", 1) for part in result.stdout.split() if "=" in part)
    if values.get("WorkDir") != str(ROOT) or values.get("JobName") != expected_name:
        raise ValueError(f"Job {job_id} is not this ledger-owned project job")
    return values


def cancel(path, job_id, pending_only=False):
    with locked(path) as state:
        try:
            refresh(state)
        except Exception as error:
            state["accounting_refresh_error"] = str(error)
        job = state["jobs"][str(job_id)]
        values = owned_job(job_id, job["name"])
        if values and values.get("JobState") in ACTIVE:
            if pending_only and values["JobState"] != "PENDING":
                raise ValueError("Only cancel an actually pending probe here")
            subprocess.run(["scancel", str(job_id)], check=True, timeout=20)
            job["cancel_requested_at"] = utc().isoformat()
        return {
            "job_id": str(job_id),
            "checked_owned": bool(values),
            "cancel_requested": "cancel_requested_at" in job,
        }


def submit(path, script, name, seconds, node=None):
    script = Path(script).resolve()
    if not script.is_relative_to(ROOT / "reports") or not name.startswith("dfnight-"):
        raise ValueError("Night submissions require this project report script and prefix")
    if seconds <= 0 or not script.is_file():
        raise ValueError("Submission needs a positive limit and existing script")
    with locked(path) as state:
        refresh(state)
        if any(event["name"] == name for event in state["submission_receipts"]):
            raise ValueError("Use a unique name per submission receipt; never blindly retry")
        seconds = min(seconds, math.floor((cutoff(state) - utc()).total_seconds()) - 3)
        if seconds < 1:
            raise BudgetStop("NO_EXECUTION_TIME_BEFORE_DEADLINE")
        available(state, seconds)
        time_limit = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
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
            f"--time={time_limit}",
            f"--job-name={name}",
            f"--chdir={ROOT}",
            f"--export=ALL,OVERNIGHT_LEDGER={Path(path).resolve()}",
        ]
        if node:
            argv.append(f"--nodelist={node}")
        argv.append(str(script))
        started = utc().isoformat()
        event = {
            "submitted_at": started,
            "argv": argv,
            "name": name,
            "status": "SUBMITTING",
            "script": str(script),
            "reserved_GPU_seconds": seconds,
        }
        state["submission_receipts"].append(event)
        atomic(path, state)
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            event.update(
                status="UNCERTAIN",
                stderr=str(error),
                stdout=(error.stdout or b"").decode()
                if isinstance(error.stdout, bytes)
                else error.stdout,
            )
            raise RuntimeError(
                "Submission outcome uncertain; reservation retained, reconcile exact name"
            ) from error
        event.update(exit_code=result.returncode, stderr=result.stderr, stdout=result.stdout)
        if result.returncode:
            event["status"] = "REJECTED_WITHOUT_ALLOCATION"
            raise RuntimeError("Slurm rejected night submission; receipt is in ledger")
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            event["status"] = "UNCERTAIN"
            raise RuntimeError("Unexpected Slurm job id; inspect submission receipt")
        event.update(status="SUBMITTED", job_id=job_id)
        state["jobs"][job_id] = {
            "name": name,
            "cards": 1,
            "limit_seconds": seconds,
            "elapsed_seconds": 0,
            "state": "SUBMITTED",
            "script": str(script),
            "submitted_at": started,
        }
        return {
            "job_id": job_id,
            "maximum_reserved_GPU_seconds": seconds,
            "deadline_utc": state["deadline_utc"],
        }


def claim_call(lane, group, seed):
    path = os.environ.get("OVERNIGHT_LEDGER")
    if not path:
        return None
    with locked(path) as state:
        refresh(state)
        if utc() >= cutoff(state):
            raise BudgetStop("ABSOLUTE_DEADLINE_REACHED_BETWEEN_IMAGES")
        if state["night_GPU_seconds"] >= state["night_GPU_seconds_limit"]:
            raise BudgetStop("GPU_BUDGET_REACHED_BETWEEN_IMAGES")
        if state["development_calls"] >= state["development_calls_limit"]:
            raise BudgetStop("DEVELOPMENT_CALL_LIMIT_REACHED")
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id not in state["jobs"]:
            raise ValueError("Generation job is absent from atomic project ledger")
        identifier = len(state["calls"]) + 1
        state["calls"].append(
            {
                "id": identifier,
                "lane": lane,
                "group": group,
                "seed": str(seed),
                "job_id": job_id,
                "started_at": utc().isoformat(),
                "status": "STARTED",
            }
        )
        state["development_calls"] += 1
        return identifier


def finish_call(identifier, status, details=None):
    if identifier is None:
        return
    with locked(os.environ["OVERNIGHT_LEDGER"]) as state:
        call = next(value for value in state["calls"] if value["id"] == identifier)
        call.update(status=status, ended_at=utc().isoformat(), details=details or {})


def execute(path, command):
    if not command:
        raise ValueError("A GPU child command is required")
    job_id = os.environ.get("SLURM_JOB_ID")
    with locked(path) as state:
        refresh(state)
        if job_id not in state["jobs"]:
            raise ValueError("GPU process must belong to a recorded lease")
        job = state["jobs"][job_id]
        left = math.floor((cutoff(state) - utc()).total_seconds()) - 2
        budget_left = (state["night_GPU_seconds_limit"] - state["night_GPU_seconds"]) // job[
            "cards"
        ]
        allowed = min(left, budget_left, max(0, job["limit_seconds"] - job["elapsed_seconds"] - 2))
        job["execution_guard"] = {
            "started_at": utc().isoformat(),
            "allowed_seconds": allowed,
            "deadline_utc": state["deadline_utc"],
            "entire_GPU_child_group_covered": True,
        }
    if allowed <= 0:
        return 124
    child = subprocess.Popen(command, start_new_session=True)
    previous = {}

    def terminate_group(signum, _frame=None):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        if _frame is not None:
            raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, terminate_group)
    try:
        return child.wait(timeout=allowed)
    except subprocess.TimeoutExpired:
        terminate_group(signal.SIGTERM)
        with locked(path) as state:
            state["jobs"][job_id]["execution_stop_reason"] = "DEADLINE_OR_RESERVED_RUNTIME_LIMIT"
        return 124
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def watch(path):
    state = json.loads(Path(path).read_text())
    left = (cutoff(state) - utc()).total_seconds()
    while left > 0:
        time.sleep(min(60, left))
        left = (cutoff(state) - utc()).total_seconds()
    state = json.loads(Path(path).read_text())
    errors = []
    for event in state["submission_receipts"]:
        if event["status"] not in {"SUBMITTING", "UNCERTAIN"}:
            continue
        try:
            result = subprocess.run(
                ["squeue", "--me", "--name", event["name"], "-h", "-o", "%i"],
                text=True,
                capture_output=True,
                timeout=20,
                check=True,
            )
        except Exception as error:
            errors.append({"submission_name": event["name"], "error": str(error)})
            continue
        for candidate in result.stdout.splitlines():
            try:
                if candidate.isdecimal() and owned_job(candidate, event["name"]):
                    subprocess.run(["scancel", candidate], check=True, timeout=20)
            except Exception as error:
                errors.append({"job_id": candidate, "error": str(error)})
    for job_id in list(state["jobs"]):
        try:
            cancel(path, job_id)
        except Exception as error:
            errors.append({"job_id": job_id, "error": str(error)})
    with locked(path) as state:
        try:
            refresh(state)
        except Exception as error:
            errors.append({"accounting_refresh_error": str(error)})
        state["deadline_watchdog_completed_at"] = utc().isoformat()
        state["status"] = "ABSOLUTE_WINDOW_ENDED"
        state["deadline_watchdog_errors"] = errors
        state["all_owned_jobs_terminal"] = not any(
            job["state"] in ACTIVE for job in state["jobs"].values()
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    sub.add_parser("watch")
    run = sub.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    add = sub.add_parser("submit")
    add.add_argument("--script", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--seconds", type=int, required=True)
    add.add_argument("--node")
    stop = sub.add_parser("cancel")
    stop.add_argument("--job", required=True)
    stop.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "status":
        with locked(args.ledger) as state:
            refresh(state)
            print(json.dumps(state, indent=2))
        return 0
    if args.mode == "watch":
        return watch(args.ledger)
    if args.mode == "run":
        return execute(
            args.ledger,
            args.command[1:] if args.command and args.command[0] == "--" else args.command,
        )
    if args.mode == "cancel":
        print(json.dumps(cancel(args.ledger, args.job, args.pending_only)))
    else:
        print(json.dumps(submit(args.ledger, args.script, args.name, args.seconds, args.node)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
