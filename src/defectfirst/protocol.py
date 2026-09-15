from __future__ import annotations

from defectfirst.data.splits import DEVELOPMENT, DIAGNOSTIC

VISA = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]
MVTEC = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
FORMAL = sorted(
    [f"visa/{p}" for p in VISA]
    + [f"mvtec/{p}" for p in MVTEC if f"mvtec/{p}" not in DEVELOPMENT]
    + ["ksdd2/surface"]
)
MAIN = ["B0", "B3", "B8", "B9", "B10", "B11", "B12", "P"]
EXTERNAL = ["E_SSN", "E_AVFM", "E_SEAS", "E_O2MAG"]
DIAGNOSTIC_METHODS = ["B1", "B2", "B4", "B5", "B7"]


def experiment_matrix(include_mechanisms: bool = True) -> list[dict]:
    jobs = []

    def add(unit, method, seed, variant="main", k=5, extra=None):
        jobs.append(
            {
                "run_id": f"{unit.replace('/', '_')}__{method}__k{k}__s{seed}__{variant}",
                "unit": unit,
                "method": method,
                "seed": seed,
                "k": k,
                "variant": variant,
                "overrides": extra or {},
                "status": "NOT_STARTED",
                "external": method in EXTERNAL,
            }
        )

    for unit in FORMAL:
        for method in MAIN + EXTERNAL:
            for seed in (11, 22, 33):
                add(unit, method, seed)
    for unit in sorted(DIAGNOSTIC):
        for seed in (11, 22, 33):
            for method in DIAGNOSTIC_METHODS:
                add(unit, method, seed, "diagnostic")
            if not include_mechanisms:
                continue
            for method in ("B3", "B12", "P"):
                for k in (0, 1, 10):
                    add(unit, method, seed, "K", k)
                add(unit, method, seed, "head_refit")
            for name, overrides in (
                ("no_inv", {"loss.inv_weight": 0.0}),
                ("no_sep", {"loss.sep_weight": 0.0}),
                ("unconstrained_head", {"model.head_mode": "linear_ablation"}),
            ):
                add(unit, "P", seed, name, extra=overrides)
            add(unit, "B10", seed, "plus_inv", extra={"loss.extra_invariance": True})
            add(unit, "B9", seed, "direct_h", extra={"loss.projection": False})
            add(unit, "B11", seed, "normal_only", extra={"loss.mmd_normal_only": True})
            for pairing in ("pooled_matched", "pooled_shuffled"):
                add(unit, "P", seed, pairing, extra={"pairing_mode": pairing})
            for method in ("B3", "B10", "B12", "P"):
                for rho in (0.1, 0.5, 0.9):
                    add(unit, method, seed, f"rho_{rho}", extra={"loss.rho": rho})
    return jobs
