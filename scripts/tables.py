#!/usr/bin/env python3
"""Summary table of the output of ``search.py``.

Reads one or more JSON lines files, removes duplicate models, and prints the
smallest values of ``dim H - dim V`` and of ``dim H - 2p`` (``p = dim V^odd``)
for all models, for the models with ``21 <= N <= 40``, for the non-pure
models, and for the models outside the classes recorded by the structural
flags (see ``hilali/classify.py``). Each entry reports whether the extremal
value was recomputed with exact arithmetic.

With ``--exact``, every extremal value computed modulo primes is recomputed
over Q (modular Betti numbers are upper bounds, so the minimum can only
decrease), until all the minima are exact.

Usage: python3 scripts/tables.py [--exact] data/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hilali.records import load_model  # noqa: E402
from hilali.sullivan import is_poincare_duality_sequence  # noqa: E402


def model_key(record: dict) -> tuple:
    model = record["model"]
    return (
        tuple(record["degrees"]),
        tuple(record["betti"]),
        tuple(tuple((tuple(a), c) for a, c in v) for v in model["differential"]),
    )


def load(paths: Iterable[Path]) -> List[dict]:
    seen, out = set(), []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("elliptic") or "model" not in record:
                    continue
                key = model_key(record)
                if key not in seen:
                    seen.add(key)
                    out.append(record)
    return out


ROWS: Dict[str, Callable[[dict], bool]] = {
    "All models": lambda r: True,
    "Models with 21 <= N <= 40": lambda r: 21 <= r["N"] <= 40,
    "Non-pure models": lambda r: not r["flags"]["pure"],
    "Models outside the known classes": lambda r: not r["flags"]["known_class"],
}


def minimum(records: Sequence[dict], key: str) -> Optional[str]:
    if not records:
        return None
    value = min(r[key] for r in records)
    exact = all(r["exact"] for r in records if r[key] == value)
    return f"{value}{'' if exact else ' (modular)'}"


def make_exact(record: dict) -> None:
    """Replace the modular cohomology fields of a record by exact ones."""
    algebra = load_model(record)
    betti = algebra.betti_numbers(method="exact")
    dim_h = sum(betti)
    record.update(
        betti=betti,
        dim_H=dim_h,
        excess=dim_h - algebra.m,
        strong_excess=dim_h - 2 * algebra.num_odd,
        poincare_duality=is_poincare_duality_sequence(betti),
        exact=True,
    )


def exact_minima(records: Sequence[dict]) -> int:
    """Recompute exactly the records attaining a minimum, until all minima are exact."""
    count = 0
    for keep in ROWS.values():
        subset = [r for r in records if keep(r)]
        for key in ("excess", "strong_excess"):
            while subset:
                value = min(r[key] for r in subset)
                pending = [r for r in subset if r[key] == value and not r["exact"]]
                if not pending:
                    break
                for record in pending:
                    make_exact(record)
                    count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--exact", action="store_true", help="recompute the extremal values over Q")
    args = parser.parse_args(argv)
    records = load(args.files)
    print(f"{len(records)} distinct elliptic models")
    if args.exact:
        print(f"{exact_minima(records)} extremal models recomputed over Q")
    print(f"Poincare duality in all models: {all(r['poincare_duality'] for r in records)}")
    print(f"dim H <= dim H(pure) in all models: "
          f"{all(r['dim_H'] <= r.get('dim_H_pure', r['dim_H']) for r in records)}")
    print()
    print(f"{'Models':36s} {'count':>7s} {'min(dim H - dim V)':>20s} {'min(dim H - 2p)':>17s}")
    for name, keep in ROWS.items():
        subset = [r for r in records if keep(r)]
        print(f"{name:36s} {len(subset):7d} {str(minimum(subset, 'excess')):>20s} "
              f"{str(minimum(subset, 'strong_excess')):>17s}")
    histogram = Counter(r["excess"] for r in records)
    print()
    print("dim H - dim V histogram (values <= 6):",
          sorted((k, v) for k, v in histogram.items() if k <= 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
