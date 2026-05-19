import pandas as pd
import hnswlib

from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Table
# =========================================================

df = pd.DataFrame([
    {"id": 1, "value": "Apple iPhone 15"},
    {"id": 2, "value": "iPhone 15 by Apple"},
    {"id": 3, "value": "Samsung Galaxy S24"},
    {"id": 4, "value": "Apple iphone15"},
    {"id": 5, "value": "Galaxy Samsung S24"},
    {"id": 6, "value": "Wooden table"},
])

print(df)


# =========================================================
# 2. Embedding model
# =========================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 3. Generate embeddings
# =========================================================

texts = df["value"].tolist()

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embeddings shape:", embeddings.shape)


# =========================================================
# 4. Create HNSW
# =========================================================

dim = embeddings.shape[1]

index = hnswlib.Index(
    space='cosine',
    dim=dim
)

# max_elements = How many objects in total
# ef_construction = index quality
# M = graph connectivity

index.init_index(
    max_elements=len(df),
    ef_construction=200,
    M=16
)

# Add vectors
index.add_items(
    embeddings,
    df["id"].tolist()
)

# ef = search quality
index.set_ef(50)


# =========================================================
# 5. Finding Nearest Neighbors
# =========================================================

K = 3

labels, distances = index.knn_query(
    embeddings,
    k=K
)

print("\nNearest neighbors:\n")

for row_idx, source_id in enumerate(df["id"]):

    print(f"\nSOURCE: {source_id} | {df.iloc[row_idx]['value']}")

    for neighbor_pos in range(K):

        target_id = labels[row_idx][neighbor_pos]

        distance = distances[row_idx][neighbor_pos]

        target_row = df[df["id"] == target_id].iloc[0]

        similarity = 1 - distance

        print(
            f"    neighbor={target_id}"
            f" similarity={similarity:.4f}"
            f" text={target_row['value']}"
        )