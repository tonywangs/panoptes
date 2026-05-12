"""Semantic entropy (Farquhar, Kossen, Kuhn, Gal — *Nature* 2024).

Token-level entropy over a language model's output distribution conflates
*surface* uncertainty (different ways of saying the same thing) with
*semantic* uncertainty (genuinely different answers). Semantic entropy
clusters temperature samples by meaning first, then takes entropy over the
cluster distribution.

Algorithm
---------
1. Draw `N` temperature-1 samples from the source model (here: a judge).
2. Greedily cluster (Farquhar 2024 §2.2): for each new sample `s_i`,
   test bidirectional NLI entailment against each existing cluster's
   representative. If `entail(s_i → s_rep) AND entail(s_rep → s_i)`, add
   `s_i` to that cluster; otherwise start a new singleton cluster with
   `s_i` as its representative.
3. Let `p_c = |c| / N` for each cluster `c`. The semantic entropy is
        H = -Σ_c p_c log p_c
   bounded in `[0, log N]`.

The greedy clustering is order-dependent but matches the published method;
empirically it produces equivalent results to the (much more expensive)
all-pairs spectral clustering for `N ≤ 20`.

When response probabilities `p(s_i)` are available, Farquhar et al. weight
clusters by `Σ_{s ∈ c} p(s)` instead of `|c| / N`; we keep that formulation
behind an optional `log_probs` argument since most judge providers don't
expose response-level log-probs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from panoptes.uq.nli.base import NLIBackend


@dataclass(frozen=True, slots=True)
class SemanticEntropyResult:
    """One semantic-entropy computation, plus the cluster assignments."""

    entropy: float
    n_clusters: int
    cluster_sizes: tuple[int, ...]
    assignments: tuple[int, ...]
    samples: tuple[str, ...]


async def semantic_entropy(
    samples: list[str],
    *,
    nli: NLIBackend,
    entail_threshold: float = 0.5,
    log_probs: list[float] | None = None,
) -> SemanticEntropyResult:
    """Cluster `samples` by bidirectional NLI and return Shannon entropy.

    Parameters
    ----------
    samples : list[str]
        N temperature-1 responses from the same source.
    nli : NLIBackend
        Backend used for the bidirectional-entailment check. Either the
        local DeBERTa backend or the LLM-as-NLI fallback.
    entail_threshold : float
        Probability above which entailment is considered to hold. 0.5 is
        the Farquhar et al. default; raise to 0.7 for tighter clusters
        with sharper backends.
    log_probs : list[float] or None
        Optional per-sample log-probabilities. When provided, cluster
        weights become normalized exp(log_p) sums; otherwise weights are
        the uniform `|c| / N` (the typical PANOPTES path since most
        judge APIs don't expose token-level log-probs).

    Returns
    -------
    SemanticEntropyResult
    """
    n = len(samples)
    if n < 2:
        raise ValueError(f"need at least 2 samples for semantic entropy; got n={n}")
    if log_probs is not None and len(log_probs) != n:
        raise ValueError(
            f"log_probs length {len(log_probs)} does not match samples length {n}"
        )

    assignments: list[int] = []
    reps: list[int] = []
    # Greedy clustering. For each sample, check against the representative of
    # each existing cluster. Pairs are batched per-sample so the backend can
    # exploit batch inference when available.
    for i, s_i in enumerate(samples):
        if not reps:
            assignments.append(0)
            reps.append(i)
            continue
        pairs: list[tuple[str, str]] = []
        for rep_idx in reps:
            pairs.append((s_i, samples[rep_idx]))
            pairs.append((samples[rep_idx], s_i))
        scores = await nli.classify_pairs(pairs)
        assigned = -1
        for k, _rep_idx in enumerate(reps):
            forward = scores[2 * k]
            backward = scores[2 * k + 1]
            if (
                forward.entailment >= entail_threshold
                and backward.entailment >= entail_threshold
            ):
                assigned = k
                break
        if assigned == -1:
            assignments.append(len(reps))
            reps.append(i)
        else:
            assignments.append(assigned)

    n_clusters = len(reps)
    weights = _cluster_weights(assignments, n_clusters, log_probs=log_probs)
    entropy = _shannon_entropy(weights)
    cluster_sizes = tuple(int(np.sum(np.array(assignments) == c)) for c in range(n_clusters))
    return SemanticEntropyResult(
        entropy=entropy,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        assignments=tuple(assignments),
        samples=tuple(samples),
    )


def _cluster_weights(
    assignments: list[int],
    n_clusters: int,
    *,
    log_probs: list[float] | None,
) -> np.ndarray:
    """Normalized weights per cluster: uniform by default, or log-prob-weighted."""
    if log_probs is None:
        counts = np.zeros(n_clusters, dtype=np.float64)
        for c in assignments:
            counts[c] += 1.0
        return counts / counts.sum()
    weights = np.zeros(n_clusters, dtype=np.float64)
    log_arr = np.asarray(log_probs, dtype=np.float64)
    # Numerically stable: subtract max to avoid overflow on exp.
    shifted = log_arr - log_arr.max()
    p = np.exp(shifted)
    for i, c in enumerate(assignments):
        weights[c] += p[i]
    weights /= weights.sum()
    return weights


def _shannon_entropy(p: np.ndarray) -> float:
    """Natural-log Shannon entropy with 0 log 0 := 0."""
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def max_entropy(n: int) -> float:
    """Upper bound `log(n)` for a discrete distribution on `n` atoms."""
    if n <= 1:
        return 0.0
    return math.log(n)
