"""Computations with minimal Sullivan algebras around Hilali's conjecture.

Main entry points:

* :class:`hilali.sullivan.SullivanAlgebra`: models, cohomology, ellipticity;
* :mod:`hilali.generate`: random minimal models built generator by generator;
* :mod:`hilali.pure`: the associated pure algebra and the derivation delta;
* :mod:`hilali.weights`: weight gradings and the weighted Euler characteristic;
* :mod:`hilali.degree_types`: the Friedlander-Halperin criterion;
* :mod:`hilali.classify`: structural flags used in the statistics.
"""

from .linalg import P1, P2, rank_exact, rank_mod
from .sullivan import SullivanAlgebra, is_poincare_duality_sequence
from .generate import generic_model, random_elliptic_model, random_minimal_model
from .pure import delta_analysis, derivation_space_dimension, linkage_test, pure_model
from .weights import weight_bound, weighted_betti_numbers
from .degree_types import elliptic_degree_types, satisfies_sac
from .classify import structural_flags
from .records import model_record

__version__ = "1.0.0"

__all__ = [
    "P1",
    "P2",
    "SullivanAlgebra",
    "delta_analysis",
    "derivation_space_dimension",
    "elliptic_degree_types",
    "generic_model",
    "is_poincare_duality_sequence",
    "linkage_test",
    "model_record",
    "pure_model",
    "random_elliptic_model",
    "random_minimal_model",
    "rank_exact",
    "rank_mod",
    "satisfies_sac",
    "structural_flags",
    "weight_bound",
    "weighted_betti_numbers",
]
