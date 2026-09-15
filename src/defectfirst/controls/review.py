from __future__ import annotations

from defectfirst.io import digest, stable_seed

CHECKS = {"mask_correct", "normal_valid", "defect_preserved", "boundary_valid", "editing_effective"}


def requires_second(group: dict, fraction: float = 0.2) -> bool:
    if not 0 <= fraction <= 1:
        raise ValueError("Invalid audit fraction")
    # Deterministic within each unit, independent of model predictions and accept decisions.
    return (
        group["role"] == "normal_diag"
        or stable_seed("review-v1", group["unit"], group["group_id"]) / (2**63 - 1) < fraction
    )


def accept_group(group: dict, reviews: list[dict], second_fraction: float = 0.2) -> dict:
    relevant = [r for r in reviews if r.get("group_id") == group["group_id"]]
    matching = [r for r in relevant if r.get("content_sha256") == group["content_sha256"]]
    if not matching:
        raise ValueError("WAITING_HUMAN: no review of this exact artifact revision")
    reviewers = [r.get("reviewer") for r in matching]
    if any(not isinstance(r, str) or not r.strip() for r in reviewers) or len(
        set(reviewers)
    ) != len(reviewers):
        raise ValueError(
            "Reviews must identify distinct reviewers; merge duplicate submissions explicitly"
        )
    for review in matching:
        if review.get("seconds", -1) < 0 or review.get("decision") not in {
            "accept",
            "reject",
            "uncertain",
            "corrected",
        }:
            raise ValueError("Review must record time and a valid decision")
        if set(review.get("conditions", {})) != {str(e) for e in group["condition_ids"]}:
            raise ValueError("Every condition must be reviewed")
        for checks in review["conditions"].values():
            if set(checks) != CHECKS or any(type(v) is not bool for v in checks.values()):
                raise ValueError("Each condition needs explicit boolean checks")
    if any(r["decision"] in {"reject", "uncertain"} for r in matching):
        raise ValueError("WAITING_HUMAN: rejection/disagreement requires a new resolved revision")
    needed = (
        2
        if requires_second(group, second_fraction)
        or any(r["decision"] == "corrected" for r in matching)
        else 1
    )
    if len(matching) < needed:
        raise ValueError("WAITING_HUMAN: second independent review required")
    accepted = []
    for e in group["condition_ids"]:
        names = CHECKS - {"editing_effective"} if e == 0 else CHECKS
        if all(all(r["conditions"][str(e)][name] for name in names) for r in matching):
            accepted.append(e)
    if 0 not in accepted or len(accepted) < 2:
        raise ValueError("No audited original-plus-sham paired group; retain the rejection record")
    return {
        **group,
        "review_status": "ACCEPTED",
        "accepted_conditions": accepted,
        "review_evidence": matching,
        "review_policy": {
            "id": "primary-all-secondary-hash-v1",
            "second_fraction": second_fraction,
            "diagnosis_double_review": True,
            "evidence_sha256": digest(matching),
        },
    }


def verify_review(group: dict) -> None:
    if group.get("review_status") != "ACCEPTED":
        raise PermissionError("Unreviewed group cannot enter training/evaluation")
    verified = accept_group(
        group, group["review_evidence"], group["review_policy"]["second_fraction"]
    )
    if (
        verified["accepted_conditions"] != group["accepted_conditions"]
        or verified["review_policy"] != group["review_policy"]
    ):
        raise ValueError("Acceptance record does not match review evidence")
