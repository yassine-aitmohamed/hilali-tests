# hilali-tests

Computations with minimal Sullivan algebras supporting the paper
*Hilali's conjecture and negative derivations* by Yassine Ait Mohamed.

The package builds random elliptic minimal Sullivan algebras, computes their
rational cohomology, and tests Hilali's conjecture and its strong form on
them. It also computes the derivation `delta` that controls the cohomology of
spaces with homotopy Euler characteristic `-1`, the weighted Euler
characteristic of a weight grading, and the degree types allowed by the
Friedlander-Halperin criterion.

The code is pure Python 3 and depends only on `numpy`.

## Mathematical background

A simply connected space `X` with rational cohomology of finite type has a
*minimal Sullivan model* `(ΛV, d)`: a free graded commutative algebra on a
graded vector space `V = V^{≥2}`, with a differential satisfying
`d(V) ⊂ Λ^{≥2}V`. Then `V^n ≅ Hom(π_n(X), Q)` and `H(ΛV, d) ≅ H*(X; Q)`.

Write `q = dim V^even`, `p = dim V^odd`, `R = Q[V^even]`, and let `f_j` be the
part of `d(y_j)` in `R` for the odd generators `y_j`. The space is *elliptic*
if `dim V` and `dim H` are both finite. By Halperin, this holds if and only if
`R/(f_1, ..., f_p)` is finite-dimensional. Then `H` is a Poincaré duality
algebra of formal dimension

    N = Σ |y_j| − Σ (|x_i| − 1),

and the homotopy Euler characteristic `χ_π = q − p` is at most zero.

**Hilali's conjecture** states that an elliptic space satisfies
`dim H*(X; Q) ≥ dim π_*(X) ⊗ Q`, that is, `dim H ≥ dim V`. The paper also
studies the **strong form** `dim H ≥ 2 dim V^odd = 2p`, which implies it.
For `χ_π = −1` the two statements are equivalent, and both reduce to the
inequality `dim ker δ ≥ q + 1` for a derivation `δ: A → H_1` from
`A = R/(f_j)` to the first Koszul homology, induced by the component of
lower degree one of `d`.

## Installation

Python 3.9 or later and `numpy` are required. No installation is needed: the
scripts and the tests run from a checkout.

```
git clone https://github.com/yassine-aitmohamed/hilali-tests.git
cd hilali-tests
python3 -m unittest discover -s tests -t .
```

To use the package from other directories, install it with
`pip install -e .`.

## Usage

```python
from hilali import SullivanAlgebra

# Heisenberg model: |a| = |b| = 3, |c| = 5, dc = ab
heis = SullivanAlgebra.from_strings([("a", 3), ("b", 3), ("c", 5)], {"c": "a*b"})
heis.is_minimal_sullivan()          # True
heis.is_elliptic()                  # True
heis.formal_dimension               # 11
heis.betti_numbers(method="exact")  # [1, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 1]

# Gr_2(C^4): Q[c1, c2] modulo the complete symmetric functions h3, h4
gr = SullivanAlgebra.from_strings(
    [("c1", 2), ("c2", 4), ("y5", 5), ("y7", 7)],
    {"y5": "c1^3 - 2*c1*c2", "y7": "c1^4 - 3*c1^2*c2 + c2^2"},
)
gr.cohomology_dimension()           # 6
```

Products in formulas are read from left to right, so `b*a = −a*b` for odd
`a`, `b`. A model can also be given by exponent vectors, as in
`SullivanAlgebra([3, 3, 5], [{}, {}, {(1, 1, 0): 1}])`.

Random models, the derivation `δ`, and the weighted Euler characteristic:

```python
import random
from hilali import random_elliptic_model, delta_analysis, weight_bound
from hilali.weights import weighted_betti_numbers, euler_characteristic

model = random_elliptic_model([2, 2, 3, 5, 7], random.Random(1))
print(model)
print(model.betti_numbers())

chi1 = SullivanAlgebra.from_strings(
    [("x", 2), ("y", 3), ("z", 3), ("w", 6), ("u", 11)],
    {"y": "x^2", "z": "2*x^2", "w": "2*x^2*y - x^2*z", "u": "w^2 + 2*w*y*z + x^6"},
)
result = delta_analysis(chi1, prime=None)
result["rank"], result["dim_kernel"], result["match"]   # (0, 4, True)

weights = [(1,), (1,), (2,)]
euler_characteristic(weighted_betti_numbers(heis, weights))
# {(0,): 1, (1,): -2, (3,): 2, (4,): -1}, that is 1 - 2t + 2t^3 - t^4
weight_bound(heis, weights)                                  # 6
```

### Scripts

| Script | Purpose |
| --- | --- |
| `scripts/examples.py` | Cohomology of the standard examples, checked against their known Betti numbers. |
| `scripts/search.py` | Random search, recording `dim H − dim V` and `dim H − 2 dim V^odd`. |
| `scripts/tables.py` | Summary table of the search output. |
| `scripts/generic.py` | Generic differentials on all elliptic degree types in a range of formal dimensions. |
| `scripts/chi1.py` | Test of the derivation `δ` for `χ_π = −1`. |

Each script prints its options with `--help`. The output format is described
in [`data/README.md`](data/README.md).

## Algorithms

**Representation.** A monomial is a vector of exponents, read as the product
of the generators in increasing index order. The differential is stored on
generators and extended by the Leibniz rule, with a cache on monomials. Since
`d²` is a derivation, `d² = 0` is checked on generators.

**Generation.** Generators are added in nondecreasing degree. When `v` of
degree `k` is added to the algebra `(ΛW, d)` already built, integral cocycles
representing a basis of `H^{k+1}(ΛW)` are computed, and `d(v)` is a random
integral combination of them. All elements of degree `k + 1` of `ΛW` are
decomposable, so the result is a minimal Sullivan algebra, and `d² = 0` holds
by construction. Up to isomorphism only the class of `d(v)` matters, so this
reaches every minimal Sullivan algebra on the given degrees. Kernels are
computed modulo the prime `2^31 − 1`, lifted to integers by rational
reconstruction, and each lifted vector is checked to be a cocycle over `Z`.
The function `generic_model` instead works over `F_p` and takes a random
combination of all cocycles, which models the generic point of each
component of the variety of differentials.

**Ellipticity.** By Halperin's criterion, the algebra is elliptic if and only
if `A = R/(f_1, ..., f_p)` is finite-dimensional. If it is, `A` vanishes above
the formal dimension `N`. Conversely, if `A` vanishes in `D` consecutive
degrees, where `D` is the largest even degree, then every monomial of higher
degree is a multiple of a monomial in this window, so `A` vanishes above it.
Hence the algebra is elliptic if and only if `A^n = 0` for
`N < n ≤ N + D`. These are finitely many rank computations.

**Ranks.** Betti numbers are `dim (ΛV)^n − rk d^n − rk d^{n−1}`. Ranks are
computed modulo the primes `2097143` and `2097133`, close to `2^21`, by a
blocked Gaussian elimination. Entries are below `2^21`, so their products are
below `2^42`, and the panel updates are matrix products of inner dimension at
most 48, which `float64` computes exactly. Exact ranks over `Q` use a sparse
elimination with `Fraction` entries.

**Why modular ranks give upper bounds.** For an integer matrix `M` and a prime
`p`, a minor that is nonzero modulo `p` is a nonzero integer, so
`rk_p(M) ≤ rk_Q(M)`. Hence the Betti numbers computed modulo `p` are upper
bounds for the rational ones, and the minimum over the two primes is still an
upper bound. In the same way, a vanishing of `A^n` modulo `p` implies its
vanishing over `Q`, so a positive ellipticity test modulo `p` is rigorous. A
counterexample to either conjecture found modulo `p` would therefore be a
counterexample over `Q`. Every extremal value, namely every model with
`dim H − dim V ≤ 1` or `dim H − 2p ≤ 0`, is recomputed with exact rational
arithmetic, and Poincaré duality is checked.

**The derivation δ.** The lower degree of an element of `R ⊗ Λ^k V^odd` is `k`,
and `d = d_{−1} + d_1 + d_3 + ...`, where `d_i` raises the lower degree by `i`
and `d_{−1} = d_σ` is the pure differential. The module `hilali.pure`
computes the Koszul homology `H_k` of `(ΛV, d_σ)` degree by degree and the
rank of `δ: A^n → H_1^{n+1}`, as the rank of `d_1` on `R^n` modulo the
boundaries `d_σ(R ⊗ Λ^2 V^odd)`. For `χ_π = −1` it compares the prediction
`dim H^even = dim ker δ`, `dim H^odd = dim coker δ` with the cohomology
computed directly.

**Weights.** For a weight grading in which no odd generator has weight zero,
`χ_T(H) ∏(1 − t^{α_i}) = ∏(1 − t^{β_j})`. The module `hilali.weights`
computes `χ_T(H)` from the weighted cohomology and from this identity, by
exact division of Laurent polynomials, and the bound `dim H ≥ ‖χ_T(H)‖_1`.

**Degree types.** `hilali.degree_types` implements the strong arithmetic
condition of Friedlander and Halperin and enumerates the degree types of
elliptic algebras with formal dimension in a given range.

## Reproducing the tables of the paper

The commands below produce samples of the kind used in the paper. Random
samples depend on the seeds and on the number of samples, so the counts, and
the minima over rare subclasses, can differ slightly from those of the paper.

```
# all degree sequences with at most 8 generators, 3 models each
python3 scripts/search.py --mode exhaustive --max-generators 8 --repeats 3 --jobs 4 --out data/exhaustive.jsonl
# models with an even generator that cannot be killed by a quadratic relation
python3 scripts/search.py --cls core --samples 24000 --jobs 4 --out data/core.jsonl
# formal dimension 21 to 40, and only odd generators
python3 scripts/search.py --cls mixed --samples 5000 --jobs 4 --out data/mixed.jsonl
python3 scripts/search.py --cls odd --samples 5000 --jobs 4 --out data/odd.jsonl

# table of the smallest values of dim H - dim V and dim H - 2p
python3 scripts/tables.py --exact data/exhaustive.jsonl data/core.jsonl data/mixed.jsonl data/odd.jsonl

# generic differentials on all degree types with N <= 20, and 21 <= N <= 30 with |V| <= 6
python3 scripts/generic.py --n-min 2 --n-max 20 --out data/generic_2_20.jsonl
python3 scripts/generic.py --n-min 21 --n-max 30 --max-generators 6 --out data/generic_21_30.jsonl

# the derivation delta for chi_pi = -1, N <= 27
python3 scripts/chi1.py --n-min 2 --n-max 27 --samples 40 --out data/chi1.jsonl
```

The row "Models outside the known classes" of the table uses the flags of
`hilali/classify.py`.

## Running the tests

```
python3 -m unittest discover -s tests -t .
python3 scripts/examples.py
```

The tests check the Betti numbers of `S^2`, `S^4`, `CP^3`, `HP^2`,
`S^3 × S^3`, the Heisenberg model, `S^2 × S^2`, `Gr_2(C^4)`, `U(3)/T` and of
the two generic models on generators of degrees `3, 3, 5, 5, 9, 13, 13`,
which have `dim H = 72` and `dim H = 48`. They also compare modular and exact
ranks, check the generators, the derivation `δ`, the weighted Euler
characteristic, and the enumeration of degree types.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{AitMohamed2026Hilali,
  author = {Ait Mohamed, Yassine},
  title  = {Hilali's conjecture and negative derivations},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/yassine-aitmohamed/hilali-tests}
}
```

## License

MIT License, see [`LICENSE`](LICENSE).
