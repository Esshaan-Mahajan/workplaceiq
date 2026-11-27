"""Small, dependency-light retrieval evaluation metrics."""

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of known relevant IDs present in the first k retrieved results."""
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    found = set(retrieved[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent."""
    relevant_ids = set(relevant)
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    runs: Iterable[tuple[Sequence[str], Iterable[str]]]
) -> float:
    """Mean reciprocal rank over (retrieved IDs, relevant IDs) query pairs."""
    values = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in runs]
    return sum(values) / len(values) if values else 0.0
