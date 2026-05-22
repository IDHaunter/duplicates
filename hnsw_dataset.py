import json
import pandas as pd
import hnswlib

from sentence_transformers import SentenceTransformer

from dataset_processor import build_analysis_records


# =========================================================
# 1. Load real data
# =========================================================

with open("./datasets/kna1.json", "r", encoding="utf-8") as f:

    raw_data = json.load(f)

    records = build_analysis_records(
        data=raw_data,
        key_fields=["mandt", "kunnr"],
        analysis_fields=["name1"]
    )

print(json.dumps(records, indent=4))

# =========================================================
# records example:
#
# {
#     "pk_source": "800|0000000001",
#     "fields_source": "Patron Automotive",
#     "fields_normalized": "patron automotive"
# }
# =========================================================


# =========================================================
# 2. Convert to DataFrame
# =========================================================

df = pd.DataFrame(records)

print(df.head())


# =========================================================
# 3. Create internal integer ids
#
# HNSW works best with integer ids
# =========================================================

df["internal_id"] = range(len(df))


# =========================================================
# 4. Create mappings
# =========================================================

internal_to_pk = dict(
    zip(df["internal_id"], df["pk_source"])
)

pk_to_internal = dict(
    zip(df["pk_source"], df["internal_id"])
)


# =========================================================
# 5. Load embedding model
# =========================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 6. Generate embeddings
#
# IMPORTANT:
# We use normalized text for embeddings
# =========================================================

texts = df["fields_normalized"].tolist()

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)


# =========================================================
# 7. Create HNSW index
# =========================================================

dim = embeddings.shape[1]

index = hnswlib.Index(
    space='cosine',
    dim=dim
)

index.init_index(
    max_elements=len(df),
    ef_construction=200,
    M=16
)


# =========================================================
# 8. Add vectors into HNSW
# =========================================================

index.add_items(
    embeddings,
    df["internal_id"].tolist()
)


# =========================================================
# 9. Search quality
#
# Higher ef:
#   better recall
#   slower search
# =========================================================

index.set_ef(50)


# =========================================================
# 10. Find nearest neighbors
# =========================================================

K = 5

labels, distances = index.knn_query(
    embeddings,
    k=K
)


# =========================================================
# 11. Duplicate detection
# =========================================================

SIMILARITY_THRESHOLD = 0.7

duplicate_candidates = []


# =========================================================
# Iterate through all source rows
# =========================================================

for row_idx, source_internal_id in enumerate(df["internal_id"]):

    source_row = df.iloc[row_idx]

    source_pk = source_row["pk_source"]

    source_original = source_row["fields_source"]

    source_normalized = source_row["fields_normalized"]


    # =====================================================
    # Iterate through neighbors
    # =====================================================

    for neighbor_pos in range(K):

        target_internal_id = labels[row_idx][neighbor_pos]

        distance = distances[row_idx][neighbor_pos]

        similarity = 1 - distance


        # =================================================
        # Skip self-match
        # =================================================

        if source_internal_id == target_internal_id:
            continue


        # =================================================
        # Apply similarity threshold
        # =================================================

        if similarity < SIMILARITY_THRESHOLD:
            continue


        # =================================================
        # Get target row
        # =================================================

        target_row = df[
            df["internal_id"] == target_internal_id
        ].iloc[0]

        target_pk = target_row["pk_source"]

        target_original = target_row["fields_source"]

        target_normalized = target_row["fields_normalized"]


        # =================================================
        # Avoid duplicate reverse pairs
        #
        # (A,B) == (B,A)
        # =================================================

        pair_key = tuple(
            sorted([source_pk, target_pk])
        )


        duplicate_candidates.append({
            "pair_key": pair_key,

            "source_pk": source_pk,
            "target_pk": target_pk,

            "source_original": source_original,
            "target_original": target_original,

            "source_normalized": source_normalized,
            "target_normalized": target_normalized,

            "similarity": round(float(similarity), 4)
        })


# =========================================================
# 12. Convert to DataFrame
# =========================================================

duplicates_df = pd.DataFrame(
    duplicate_candidates
)


# =========================================================
# 13. Remove duplicate reverse pairs
# =========================================================

duplicates_df = duplicates_df.drop_duplicates(
    subset=["pair_key"]
)

duplicates_df = duplicates_df.drop(
    columns=["pair_key"]
)


# =========================================================
# 14. Sort by similarity
# =========================================================

duplicates_df = duplicates_df.sort_values(
    by="similarity",
    ascending=False
)


# =========================================================
# 15. Output results
# =========================================================

print("\n================ DUPLICATE CANDIDATES ================\n")

print(
    duplicates_df.to_string(index=False)
)