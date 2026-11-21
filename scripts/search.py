#!/usr/bin/env python3
"""Random search for elliptic minimal Sullivan algebras with small cohomology.

Each sample draws a degree sequence, builds a random integral minimal model
generator by generator, tests ellipticity, and records ``dim H - dim V``
(Hilali's conjecture) and ``dim H - 2 dim V^odd`` (the strong form). Extremal
values are recomputed with exact rational arithmetic. Elliptic models are
written as JSON lines, see ``data/README.md``.

Degree classes for ``--mode random``:

* ``odd``: only odd generators;
* ``small``: one to three even generators, formal dimension at most 30;
* ``mixed``: one to four even generators, formal dimension 21 to 40;
* ``core``: an even generator ``x`` with no odd generator of degree
  ``2|x| - 1``, so that ``x`` cannot be killed by a quadratic relation.

With ``--mode exhaustive`` every degree sequence with at most
``--max-generators`` generators (even degrees 2 to 8, odd degrees 3 to 19,
formal dimension at most 40, satisfying the Friedlander-Halperin condition)
is sampled ``--repeats`` times.

Examples:
    python3 scripts/search.py --mode random --cls small --samples 300 --out data/small.jsonl
    python3 scripts/search.py --mode exhaustive --max-generators 6 --repeats 1 --out data/exh6.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hilali.degree_types import formal_dimension, max_basis_size, satisfies_sac  # noqa: E402
from hilali.generate import random_minimal_model  # noqa: E402
from hilali.records import model_record  # noqa: E402

MAX_BASIS = 2500
COEFFICIENT_SETS = [(1, -1), (1, -1, 2, -2, 3), (1, 2, 3, 5, -7), (1, -1, 2)]


# ---------------------------------------------------------------- degrees
def _split(degrees: Sequence[int]) -> Tuple[List[int], List[int]]:
    return [g for g in degrees if g % 2 == 0], [g for g in degrees if g % 2]


def _acceptable(degrees: Sequence[int], cls: str) -> bool:
    evens, odds = _split(degrees)
    n = formal_dimension(evens, odds)
    if n < 2:
        return False
    if cls == "mixed" and not 21 <= n <= 40:
        return False
    if cls == "small" and n > 30:
        return False
    if cls == "core" and n > 40:
        return False
    return max_basis_size(degrees, offset=2) <= MAX_BASIS


def _draw(cls: str, rng: random.Random) -> Optional[List[int]]:
    if cls == "odd":
        return [rng.choice([3, 3, 3, 5, 5, 7, 7, 9, 11, 13]) for _ in range(rng.randint(3, 9))]
    if cls == "small":
        ne = rng.randint(1, 3)
        no = rng.randint(ne, ne + 2)
        return [rng.choice([2, 2, 2, 4, 4, 6]) for _ in range(ne)] + [
            rng.choice([3, 3, 5, 5, 7, 9, 11]) for _ in range(no)
        ]
    if cls == "core":
        a = rng.choice([2, 2, 4])
        m = rng.choice([3, 3, 4])
        ne = rng.randint(1, 3)
        no = rng.randint(ne, ne + 3)
        evens = [a] + [rng.choice([2, 4, 6]) for _ in range(ne - 1)]
        odds = [m * a - 1] + [
            rng.choice([3, 3, 5, 5, 7, 7, 9, 11, 13, 15]) for _ in range(no - 1)
        ]
        if any(y == 2 * a - 1 for y in odds):
            return None
        return evens + odds
    if cls == "mixed":
        ne = rng.randint(1, 4)
        no = rng.randint(ne, min(ne + 4, 10 - ne))
        return [rng.choice([2, 2, 2, 4, 4, 6, 8]) for _ in range(ne)] + [
            rng.choice([3, 3, 5, 5, 7, 7, 9, 11, 13, 15, 17, 19, 23]) for _ in range(no)
        ]
    raise ValueError(f"unknown class {cls!r}")


def sample_degrees(cls: str, rng: random.Random) -> List[int]:
    """Draw a degree sequence of the given class, within the size limits."""
    while True:
        degrees = _draw(cls, rng)
        if degrees and _acceptable(degrees, cls):
            return sorted(degrees)


def exhaustive_sequences(max_generators: int) -> List[Tuple[int, ...]]:
    """All degree sequences of the exhaustive mode."""
    out = []
    for ne in range(0, 4):
        for no in range(max(ne, 1), max_generators - ne + 1):
            for evens in itertools.combinations_with_replacement([2, 4, 6, 8], ne):
                for odds in itertools.combinations_with_replacement(range(3, 20, 2), no):
                    n = formal_dimension(evens, odds)
                    if not 2 <= n <= 40:
                        continue
                    if evens and not satisfies_sac(evens, odds):
                        continue
                    degrees = tuple(sorted(evens + odds))
                    if max_basis_size(degrees, offset=2) <= MAX_BASIS:
                        out.append(degrees)
    return out


# ---------------------------------------------------------------- work
Task = Tuple[str, Optional[Tuple[int, ...]], int, Optional[int]]


def generator_options(rng: random.Random, variant: Optional[int]) -> dict:
    """Parameters of :func:`random_minimal_model`.

    In the exhaustive mode, the repetitions of a sequence cycle through four
    fixed variants. In the random mode, each parameter is drawn independently.
    """
    if variant is not None:
        return dict(
            density=[1.0, 0.5, 0.25, 0.6][variant % 4],
            coefficients=COEFFICIENT_SETS[variant % 4],
            zero_probability=0.0 if variant % 4 < 2 else 0.15,
        )
    return dict(
        density=rng.choice([0.25, 0.5, 0.75, 1.0, 1.0]),
        coefficients=rng.choice(COEFFICIENT_SETS[:3]),
        zero_probability=rng.choice([0.0, 0.0, 0.1]),
    )


def run_task(task: Task) -> dict:
    """Build and analyse one model. ``task = (class, degrees or None, seed, variant or None)``."""
    cls, degrees, seed, variant = task
    rng = random.Random(seed)
    if degrees is None:
        degrees = tuple(sample_degrees(cls, rng))
    model = random_minimal_model(degrees, rng, **generator_options(rng, variant))
    if model is None:
        return {"class": cls, "seed": seed, "degrees": list(degrees), "failed": True}
    record = model_record(model)
    record["class"] = cls
    record["seed"] = seed
    return record


def tasks(args: argparse.Namespace) -> Iterator[Task]:
    if args.mode == "exhaustive":
        sequences = exhaustive_sequences(args.max_generators)
        print(f"{len(sequences)} degree sequences", flush=True)
        jobs = [
            ("exhaustive", s, args.seed + k * args.repeats + r, r)
            for k, s in enumerate(sequences)
            for r in range(args.repeats)
        ]
        random.Random(args.seed).shuffle(jobs)
        yield from jobs
    else:
        for k in range(args.samples):
            yield (args.cls, None, args.seed + k, None)


class Summary:
    """Running minima of the two excesses."""

    def __init__(self) -> None:
        self.samples = 0
        self.elliptic = 0
        self.best = {"excess": None, "strong_excess": None}

    def add(self, record: dict) -> None:
        self.samples += 1
        if not record.get("elliptic"):
            return
        self.elliptic += 1
        for key in self.best:
            value = record[key]
            if self.best[key] is None or value < self.best[key]:
                self.best[key] = value
                print(
                    f"new minimum {key}={value} degrees={record['degrees']} "
                    f"betti={record['betti']} exact={record['exact']}",
                    flush=True,
                )

    def report(self, seconds: float) -> None:
        print(
            f"done: {self.samples} samples, {self.elliptic} elliptic, "
            f"min(dim H - dim V)={self.best['excess']}, "
            f"min(dim H - 2p)={self.best['strong_excess']} ({seconds:.0f}s)"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["random", "exhaustive"], default="random")
    parser.add_argument("--cls", choices=["odd", "small", "mixed", "core"], default="small")
    parser.add_argument("--samples", type=int, default=200, help="number of samples (random mode)")
    parser.add_argument("--max-generators", type=int, default=8, help="exhaustive mode")
    parser.add_argument("--repeats", type=int, default=3, help="samples per sequence (exhaustive mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", type=Path, required=True, help="output file (JSON lines, appended)")
    args = parser.parse_args(argv)

    summary = Summary()
    start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        if args.jobs > 1:
            with Pool(args.jobs) as pool:
                results = pool.imap_unordered(run_task, tasks(args), chunksize=4)
                _consume(results, out, summary)
        else:
            _consume(map(run_task, tasks(args)), out, summary)
    summary.report(time.time() - start)
    return 0


def _consume(results, out, summary: Summary) -> None:
    for record in results:
        summary.add(record)
        if record.get("elliptic"):
            out.write(json.dumps(record) + "\n")
            out.flush()


if __name__ == "__main__":
    sys.exit(main())
