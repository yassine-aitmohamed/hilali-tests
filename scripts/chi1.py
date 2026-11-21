#!/usr/bin/env python3
"""Test the derivation delta on models with homotopy Euler characteristic -1.

For every elliptic degree type with ``p = q + 1`` odd generators, ``q >= 2``
even generators and ``N_min <= N <= N_max``, in which some even generator can
have a nonzero differential, random models over ``F_p`` are drawn with
:func:`hilali.generate.generic_model`. For each elliptic model with a
non-closed even generator, the derivation ``delta: A -> H_1`` of the odd
spectral sequence is computed degree by degree. Hilali's conjecture for such a
model is the inequality ``dim ker delta >= q + 1``.

For the first models of each type, the prediction ``dim H^even = dim ker
delta`` and ``dim H^odd = dim coker delta`` is compared with the cohomology
computed directly. One JSON line is written per degree type.

Example:
    python3 scripts/chi1.py --n-min 2 --n-max 14 --samples 10 --out data/chi1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hilali.degree_types import elliptic_degree_types  # noqa: E402
from hilali.generate import generic_model  # noqa: E402
from hilali.pure import (  # noqa: E402
    delta_analysis,
    has_chain_level_d1,
    has_nonclosed_even,
    linkage_test,
)


def is_sum_of(target: int, parts: Sequence[int]) -> bool:
    """``target`` is a sum of one or more elements of ``parts``."""
    if target <= 0:
        return False
    reachable = [True] + [False] * target
    for v in range(1, target + 1):
        reachable[v] = any(g <= v and reachable[v - g] for g in parts)
    return reachable[target]


def admits_nonclosed_even(evens: Sequence[int], odds: Sequence[int]) -> bool:
    """Some ``d(x)`` can contain a term ``y * (product of lower even generators)``."""
    for x in evens:
        lower = [e for e in evens if e < x]
        if any(y < x and is_sum_of(x + 1 - y, lower) for y in odds):
            return True
    return False


def candidate_types(n_min: int, n_max: int, max_generators: int) -> List[tuple]:
    out = []
    for evens, odds, n in elliptic_degree_types(n_max, n_min):
        if len(odds) != len(evens) + 1 or len(evens) < 2:
            continue
        if len(evens) + len(odds) > max_generators or evens.count(2) > 3:
            continue
        if admits_nonclosed_even(evens, odds):
            out.append((evens, odds, n))
    return out


def study_type(evens, odds, n, samples: int, checks: int, seconds: float, rng: random.Random) -> dict:
    degrees = sorted(evens + odds)
    m = len(degrees)
    q = len(evens)
    stats = dict(
        evens=list(evens), odds=list(odds), N=n, q=q, elliptic=0, delta_nonzero=0,
        min_dim_kernel=None, chain_level_d1=0, unlinked=0, checked=0, matched=0,
        higher_koszul=0, examples=[],
    )
    start = time.time()
    for k in range(samples):
        pattern = "g" * m if k < 2 else "".join(rng.choice("ggzs") for _ in range(m))
        model = generic_model(degrees, rng, pattern)
        if not has_nonclosed_even(model) or not model.is_elliptic():
            continue
        result = delta_analysis(model, compare=stats["checked"] < checks)
        _update(stats, model, result, pattern, rng)
        if time.time() - start > seconds:
            break
    stats["seconds"] = round(time.time() - start, 1)
    return stats


def _update(stats: dict, model, result: dict, pattern: str, rng: random.Random) -> None:
    stats["elliptic"] += 1
    if "match" in result:
        stats["checked"] += 1
        stats["matched"] += int(result["match"])
    stats["higher_koszul"] += int(result["dim_higher"] != 0)
    stats["chain_level_d1"] += int(has_chain_level_d1(model))
    stats["unlinked"] += int(not any(linkage_test(model, rng)))
    stats["delta_nonzero"] += int(result["rank"] > 0)
    if stats["min_dim_kernel"] is None or result["dim_kernel"] < stats["min_dim_kernel"]:
        stats["min_dim_kernel"] = result["dim_kernel"]
    if result["rank"] > 0 or result.get("match") is False or len(stats["examples"]) < 1:
        stats["examples"].append(dict(pattern=pattern, rank=result["rank"],
                                      dim_kernel=result["dim_kernel"], model=model.to_dict()))
        stats["examples"] = stats["examples"][:3]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=14)
    parser.add_argument("--samples", type=int, default=20, help="samples per degree type")
    parser.add_argument("--checks", type=int, default=3, help="direct cohomology checks per type")
    parser.add_argument("--max-generators", type=int, default=9)
    parser.add_argument("--seconds", type=float, default=40.0, help="time budget per type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    types = candidate_types(args.n_min, args.n_max, args.max_generators)
    print(f"{len(types)} degree types", flush=True)
    totals = dict(models=0, delta_nonzero=0, checked=0, matched=0, slack=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        for evens, odds, n in types:
            stats = study_type(evens, odds, n, args.samples, args.checks, args.seconds, rng)
            out.write(json.dumps(stats) + "\n")
            out.flush()
            print(f"{evens} {odds} N={n}: {stats['elliptic']} models, delta != 0 in "
                  f"{stats['delta_nonzero']}, min dim ker delta = {stats['min_dim_kernel']} "
                  f"(q+1 = {len(evens) + 1}), checks {stats['matched']}/{stats['checked']}", flush=True)
            _accumulate(totals, stats)
    print(f"done: {totals['models']} models, delta != 0 in {totals['delta_nonzero']}, "
          f"direct checks {totals['matched']}/{totals['checked']}, "
          f"min(dim ker delta - (q+1)) = {totals['slack']}")
    return 0


def _accumulate(totals: dict, stats: dict) -> None:
    totals["models"] += stats["elliptic"]
    totals["delta_nonzero"] += stats["delta_nonzero"]
    totals["checked"] += stats["checked"]
    totals["matched"] += stats["matched"]
    if stats["min_dim_kernel"] is not None:
        slack = stats["min_dim_kernel"] - (stats["q"] + 1)
        if totals["slack"] is None or slack < totals["slack"]:
            totals["slack"] = slack


if __name__ == "__main__":
    sys.exit(main())
