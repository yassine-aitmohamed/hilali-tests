#!/usr/bin/env python3
"""Generic differentials on all elliptic degree types in a range of formal dimensions.

For each degree type satisfying the Friedlander-Halperin condition with
``N_min <= N <= N_max``, models over ``F_p`` are drawn with
:func:`hilali.generate.generic_model` for a fixed list of patterns: three
generic ones, the ones with a single closed generator, the pure one, and
random patterns. The smallest total Betti number found, and the one of the
generic patterns, are recorded together with the two excesses.

For types with as many odd as even generators (``chi_pi = 0``), the
cohomology of every elliptic model is a complete intersection and
``dim H = prod(|y_j| + 1) / prod |x_i|``, so no sampling is needed.

Example:
    python3 scripts/generic.py --n-min 2 --n-max 12 --out data/generic_2_12.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hilali.degree_types import elliptic_degree_types, max_basis_size  # noqa: E402
from hilali.generate import generic_model  # noqa: E402


def patterns(degrees: Sequence[int], rng: random.Random, extra: int) -> List[str]:
    m = len(degrees)
    out = ["g" * m] * 3
    out += ["g" * k + "z" + "g" * (m - k - 1) for k in range(m)]
    out.append("".join("p" if g % 2 else "z" for g in degrees))
    out.append("".join("g" if g % 2 else "z" for g in degrees))
    out += ["".join(rng.choice("ggzs") for _ in range(m)) for _ in range(extra)]
    return out


def complete_intersection_dimension(evens: Sequence[int], odds: Sequence[int]) -> int:
    value = Fraction(1)
    for y in odds:
        value *= y + 1
    for x in evens:
        value /= x
    return int(value)


def study(evens, odds, n, extra: int, seed: int) -> dict:
    degrees = sorted(evens + odds)
    p = len(odds)
    record = dict(evens=list(evens), odds=list(odds), N=n, dim_V=len(degrees), p=p)
    if len(evens) == len(odds):
        h = complete_intersection_dimension(evens, odds)
        record.update(chi_pi_zero=True, h_min=h, h_generic=h)
    else:
        rng = random.Random(seed)
        start = time.time()
        samples = []
        for pattern in patterns(degrees, rng, extra):
            model = generic_model(degrees, rng, pattern)
            if model.is_elliptic():
                betti = model.betti_numbers()
                samples.append((pattern, sum(betti), betti))
            else:
                samples.append((pattern, None, None))
        elliptic = [s for s in samples if s[1] is not None]
        generic = [s for s in samples[:3] if s[1] is not None]
        best = min(elliptic, key=lambda s: s[1]) if elliptic else None
        record.update(
            h_generic=min(s[1] for s in generic) if generic else None,
            h_min=best[1] if best else None,
            best_pattern=best[0] if best else None,
            best_betti=best[2] if best else None,
            non_elliptic=len(samples) - len(elliptic),
            seconds=round(time.time() - start, 2),
        )
    if record["h_min"] is not None:
        record["excess"] = record["h_min"] - record["dim_V"]
        record["strong_excess"] = record["h_min"] - 2 * p
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--max-generators", type=int, default=99)
    parser.add_argument("--max-basis", type=int, default=4000, help="skip larger types")
    parser.add_argument("--extra", type=int, default=8, help="random patterns per type")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    types = [
        t for t in elliptic_degree_types(args.n_max, args.n_min)
        if len(t[0]) + len(t[1]) <= args.max_generators
    ]
    print(f"{len(types)} degree types", flush=True)
    best = {"excess": None, "strong_excess": None}
    skipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        for evens, odds, n in types:
            size = max_basis_size(evens + odds)
            if len(evens) != len(odds) and size > args.max_basis:
                out.write(json.dumps(dict(evens=list(evens), odds=list(odds), N=n, skipped=size)) + "\n")
                skipped += 1
                continue
            record = study(evens, odds, n, args.extra, args.seed)
            out.write(json.dumps(record) + "\n")
            out.flush()
            for key in best:
                if key in record and (best[key] is None or record[key] < best[key]):
                    best[key] = record[key]
    print(f"done: {len(types) - skipped} types sampled, {skipped} skipped, "
          f"min(dim H - dim V)={best['excess']}, min(dim H - 2p)={best['strong_excess']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
