import json
import time
import logging

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import pandas as pd
import hnswlib
import numpy as np

from sentence_transformers import SentenceTransformer

from dataset_processor import build_analysis_records


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# TIMING DECORATOR
# =========================================================

def timed(func: Callable) -> Callable:
    """Measure execution time of a function.

    Logs execution time of the wrapped function using the global logger.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function with execution time logging.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:

        start: float = time.perf_counter()

        result: Any = func(*args, **kwargs)

        elapsed: float = time.perf_counter() - start

        logger.info(
            f"[TIME] {func.__name__} -> {elapsed:.4f} sec"
        )

        return result

    return wrapper


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class DuplicateCandidate:
    """Represent a duplicate candidate pair."""

    source_pk: str
    target_pk: str

    source_original: str
    target_original: str

    source_normalized: str
    target_normalized: str

    similarity: float


# =========================================================
# LOAD DATA
# =========================================================

@timed
def load_records(path: str) -> list[dict[str, Any]]:
    """Load and preprocess source records from JSON file.

    Reads raw JSON data and converts it into analysis records
    using the dataset processor.

    Args:
        path (str): Path to source JSON file.

    Returns:
        list[dict[str, Any]]: List of processed analysis records.
    """

    with open(path, "r", encoding="utf-8") as f:

        raw_data: list[dict[str, Any]] = json.load(f)

    records: list[dict[str, Any]] = build_analysis_records(
        data=raw_data,
        key_fields=["mandt", "kunnr"],
        analysis_fields=["name1"]
    )

    return records


# =========================================================
# CREATE DATAFRAME
# =========================================================

@timed
def create_dataframe(
    records: list[dict[str, Any]]
) -> pd.DataFrame:
    """Create DataFrame with internal identifiers.

    Converts analysis records into a pandas DataFrame and
    assigns sequential internal identifiers.

    Args:
        records (list[dict[str, Any]]): Source analysis records.

    Returns:
        pd.DataFrame: DataFrame with internal_id column.
    """

    df: pd.DataFrame = pd.DataFrame(records)

    df["internal_id"] = range(len(df))

    return df


# =========================================================
# LOAD MODEL
# =========================================================

@timed
def load_model(model_name: str) -> SentenceTransformer:
    """Load sentence transformer model.

    Args:
        model_name (str): HuggingFace model name.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """

    return SentenceTransformer(model_name)


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

@timed
def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str]
) -> np.ndarray:
    """Generate normalized embeddings for input texts.

    Args:
        model (SentenceTransformer): Embedding model.
        texts (list[str]): Source texts.

    Returns:
        np.ndarray: Embedding matrix.
    """

    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    logger.info(
        f"Embeddings shape: {embeddings.shape}"
    )

    return embeddings


# =========================================================
# BUILD HNSW INDEX
# =========================================================

@timed
def build_hnsw_index(
    embeddings: np.ndarray,
    internal_ids: list[int],
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 50
) -> hnswlib.Index:
    """Build and populate HNSW index.

    Args:
        embeddings (np.ndarray): Embedding matrix.
        internal_ids (list[int]): Vector identifiers.
        ef_construction (int): Construction accuracy parameter.
        M (int): Graph connectivity parameter.
        ef_search (int): Search accuracy parameter.

    Returns:
        hnswlib.Index: Initialized HNSW index.
    """

    dim: int = embeddings.shape[1]

    index: hnswlib.Index = hnswlib.Index(
        space="cosine",
        dim=dim
    )

    index.init_index(
        max_elements=len(internal_ids),
        ef_construction=ef_construction,
        M=M
    )

    index.add_items(
        embeddings,
        internal_ids
    )

    index.set_ef(ef_search)

    return index


# =========================================================
# SEARCH NEIGHBORS
# =========================================================

@timed
def search_neighbors(
    index: hnswlib.Index,
    embeddings: np.ndarray,
    k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Search nearest neighbors in HNSW index.

    Args:
        index (hnswlib.Index): HNSW index.
        embeddings (np.ndarray): Query embeddings.
        k (int): Number of neighbors.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Neighbor labels and distances.
    """

    labels: np.ndarray
    distances: np.ndarray

    labels, distances = index.knn_query(
        embeddings,
        k=k
    )

    return labels, distances


# =========================================================
# BUILD ROW LOOKUP
# =========================================================

@timed
def build_row_lookup(
    df: pd.DataFrame
) -> dict[int, pd.Series]:
    """Create fast row lookup dictionary.

    Avoids expensive DataFrame filtering during duplicate search.

    Args:
        df (pd.DataFrame): Source DataFrame.

    Returns:
        dict[int, pd.Series]:
            Mapping internal_id -> DataFrame row.
    """

    return {
        row["internal_id"]: row
        for _, row in df.iterrows()
    }


# =========================================================
# DETECT DUPLICATES
# =========================================================

@timed
def detect_duplicates(
    df: pd.DataFrame,
    labels: np.ndarray,
    distances: np.ndarray,
    rows_by_internal_id: dict[int, pd.Series],
    similarity_threshold: float = 0.7,
    k: int = 5
) -> list[dict[str, Any]]:
    """Detect duplicate candidates using nearest neighbors.

    Compares nearest neighbor pairs and filters them by
    similarity threshold.

    Args:
        df (pd.DataFrame): Source DataFrame.
        labels (np.ndarray): Neighbor labels.
        distances (np.ndarray): Neighbor distances.
        rows_by_internal_id (dict[int, pd.Series]):
            Fast row lookup dictionary.
        similarity_threshold (float):
            Minimum similarity threshold.
        k (int): Number of neighbors.

    Returns:
        list[dict[str, Any]]:
            List of duplicate candidate records.
    """

    duplicate_candidates: list[dict[str, Any]] = []

    seen_pairs: set[tuple[str, str]] = set()

    for row_idx, source_internal_id in enumerate(df["internal_id"]):

        source_row: pd.Series = rows_by_internal_id[
            source_internal_id
        ]

        source_pk: str = source_row["pk_source"]

        for neighbor_pos in range(k):

            target_internal_id: int = labels[row_idx][neighbor_pos]

            distance: float = distances[row_idx][neighbor_pos]

            similarity: float = 1 - distance

            # =============================================
            # Skip self-match
            # =============================================

            if source_internal_id == target_internal_id:
                continue

            # =============================================
            # Apply threshold
            # =============================================

            if similarity < similarity_threshold:
                continue

            target_row: pd.Series = rows_by_internal_id[
                target_internal_id
            ]

            target_pk: str = target_row["pk_source"]

            # =============================================
            # Avoid reverse duplicates
            # =============================================

            pair_key: tuple[str, str]

            if source_pk < target_pk:
                pair_key = (source_pk, target_pk)
            else:
                pair_key = (target_pk, source_pk)

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            candidate: DuplicateCandidate = DuplicateCandidate(
                source_pk=source_pk,
                target_pk=target_pk,

                source_original=source_row["fields_source"],
                target_original=target_row["fields_source"],

                source_normalized=source_row["fields_normalized"],
                target_normalized=target_row["fields_normalized"],

                similarity=round(float(similarity), 4)
            )

            duplicate_candidates.append(
                candidate.__dict__
            )

    return duplicate_candidates


# =========================================================
# CREATE RESULT DATAFRAME
# =========================================================

@timed
def create_duplicates_dataframe(
    duplicate_candidates: list[dict[str, Any]]
) -> pd.DataFrame:
    """Create sorted duplicate candidates DataFrame.

    Args:
        duplicate_candidates (list[dict[str, Any]]):
            Duplicate candidate records.

    Returns:
        pd.DataFrame:
            Sorted duplicate candidates DataFrame.
    """

    duplicates_df: pd.DataFrame = pd.DataFrame(
        duplicate_candidates
    )

    if duplicates_df.empty:
        return duplicates_df

    duplicates_df = duplicates_df.sort_values(
        by="similarity",
        ascending=False
    )

    return duplicates_df


# =========================================================
# MAIN
# =========================================================

@timed
def main() -> None:
    """Execute duplicate detection pipeline."""

    # =====================================================
    # Load records
    # =====================================================

    records: list[dict[str, Any]] = load_records(
        "./datasets/kna1.json"
    )

    logger.info(
        f"Loaded records: {len(records)}"
    )

    # =====================================================
    # DataFrame
    # =====================================================

    df: pd.DataFrame = create_dataframe(records)

    # =====================================================
    # Model
    # =====================================================

    model: SentenceTransformer = load_model(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # =====================================================
    # Embeddings
    # =====================================================

    texts: list[str] = df[
        "fields_normalized"
    ].tolist()

    embeddings: np.ndarray = generate_embeddings(
        model,
        texts
    )

    # =====================================================
    # HNSW
    # =====================================================

    index: hnswlib.Index = build_hnsw_index(
        embeddings,
        df["internal_id"].tolist() # ids which used as labels
    )

    # =====================================================
    # KNN Search
    # =====================================================

    neighbors_count: int = 5

    labels: np.ndarray      # [[0 11  3 13 14], ... ]
    distances: np.ndarray   # [[-3.5762787e-07  6.2004298e-01  6.3889337e-01  7.6401603e-01   7.6401603e-01], ... ]

    labels, distances = search_neighbors(
        index,
        embeddings,
        k=neighbors_count
    )

    # =====================================================
    # Fast lookup
    # =====================================================

    rows_by_internal_id: dict[int, pd.Series]

    rows_by_internal_id = build_row_lookup(df)

    # =====================================================
    # Duplicate detection
    # =====================================================

    duplicate_candidates: list[dict[str, Any]]

    duplicate_candidates = detect_duplicates(
        df=df,
        labels=labels,
        distances=distances,
        rows_by_internal_id=rows_by_internal_id,
        similarity_threshold=0.7,
        k=neighbors_count
    )

    # =====================================================
    # Result DataFrame
    # =====================================================

    duplicates_df: pd.DataFrame

    duplicates_df = create_duplicates_dataframe(
        duplicate_candidates
    )

    # =====================================================
    # Output
    # =====================================================

    print("\n================ DUPLICATE CANDIDATES ================\n")

    if duplicates_df.empty:

        print("No duplicates found")

    else:

        print(
            duplicates_df.to_string(index=False)
        )


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    main()