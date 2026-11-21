# Output format

All scripts write [JSON lines](https://jsonlines.org): one JSON object per
line. Output files are opened in append mode, so several runs can be
collected in one file. Output files (`*.jsonl`) in this directory are ignored by git.

## Models (`scripts/search.py`)

One line per elliptic model, produced by `hilali.records.model_record`.

| Field | Meaning |
| --- | --- |
| `degrees` | degrees of the generators, in nondecreasing order |
| `N` | formal dimension `Σ|y_j| − Σ(|x_i| − 1)` |
| `dim_V`, `q`, `p` | `dim V`, `dim V^even`, `dim V^odd` |
| `chi_pi` | homotopy Euler characteristic `q − p` |
| `elliptic` | always `true` in the output files |
| `betti` | `dim H^n` for `0 ≤ n ≤ N` |
| `dim_H` | total dimension of the cohomology |
| `excess` | `dim H − dim V` (Hilali's conjecture: `≥ 0`) |
| `strong_excess` | `dim H − 2 dim V^odd` (strong form: `≥ 0`) |
| `poincare_duality` | `b_n = b_{N−n}` and `b_N = 1` |
| `exact` | `true` if the Betti numbers were computed over `Q`, `false` if they are the minimum of the values modulo `2097143` and `2097133` (upper bounds for the rational ones) |
| `dim_H_pure` | total dimension of the cohomology of the associated pure algebra (modular) |
| `flags` | structural flags, see `hilali/classify.py`: `pure`, `two_stage`, `coformal`, `homogeneous`, `even_closed`, `word_lengths`, `lowest_word_length`, `quadratic_part_elliptic`, `lowest_part_elliptic`, `known_class` |
| `model` | the model itself (see below) |
| `class`, `seed` | degree class and random seed of the sample |

The Betti numbers are recomputed over `Q` whenever `excess ≤ 1` or
`strong_excess ≤ 0`.

The field `model` has the form

```json
{"generators": [["x1", 2], ["y1", 3]],
 "differential": [[], [[[2, 0], 1]]],
 "modulus": null}
```

where `differential[i]` lists the terms `[exponents, coefficient]` of
`d(v_i)`, the exponents being indexed like `generators`. The model is
recovered by `hilali.records.load_model(record)` or
`SullivanAlgebra.from_dict(record["model"])`.

## Degree types (`scripts/generic.py`)

One line per degree type.

| Field | Meaning |
| --- | --- |
| `evens`, `odds`, `N`, `dim_V`, `p` | the degree type and its invariants |
| `chi_pi_zero` | present and `true` if `q = p`; then `h_min = h_generic = ∏(|y_j| + 1) / ∏|x_i|` |
| `h_generic` | smallest `dim H` among the three generic samples (over `F_p`) |
| `h_min` | smallest `dim H` among all samples |
| `best_pattern`, `best_betti` | pattern and Betti numbers of the sample attaining `h_min` |
| `non_elliptic` | number of non-elliptic samples |
| `excess`, `strong_excess` | `h_min − dim V` and `h_min − 2p` |
| `skipped` | present instead of the results when the type is too large; its value is the largest dimension of `(ΛV)^n` |

A pattern has one letter per generator in increasing degree: `g` generic,
`s` sparse, `z` closed, `p` pure (see `hilali.generate.generic_model`).

## The derivation delta (`scripts/chi1.py`)

One line per degree type with `p = q + 1`.

| Field | Meaning |
| --- | --- |
| `evens`, `odds`, `N`, `q` | the degree type |
| `elliptic` | number of elliptic models with a non-closed even generator |
| `delta_nonzero` | number of them with `δ ≠ 0` |
| `min_dim_kernel` | smallest `dim ker δ` (Hilali's conjecture: `≥ q + 1`) |
| `chain_level_d1` | number of models with `d_1(x) ≠ 0` for some even generator |
| `unlinked` | number of models for which the linkage test failed for every relation |
| `higher_koszul` | number of models with `H_k ≠ 0` for some `k ≥ 2` (expected: 0) |
| `checked`, `matched` | number of direct cohomology computations compared with the Betti numbers predicted by `δ`, and how many agreed |
| `examples` | up to three sample models with their pattern, rank and kernel |
| `seconds` | time spent on the type |
