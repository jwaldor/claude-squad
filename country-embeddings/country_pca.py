"""
Embed every country in the world with a local sentence-transformer model,
reduce to 2D with PCA, visualize, and report on Israel's nearest neighbors
and the main clusters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycountry
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

OUT = Path(__file__).parent
MODEL_NAME = "all-MiniLM-L6-v2"
N_CLUSTERS = 8
N_NEIGHBORS = 10


def country_list() -> list[str]:
    # pycountry includes some long official names; use common_name where possible.
    names = []
    for c in pycountry.countries:
        name = getattr(c, "common_name", None) or c.name
        names.append(name)
    return sorted(set(names))


def main() -> None:
    countries = country_list()
    print(f"Loaded {len(countries)} countries")

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding country names...")
    emb = model.encode(countries, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb)
    print(f"Embeddings shape: {emb.shape}")

    print("Running PCA -> 2 components...")
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(emb)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_} "
          f"(total {pca.explained_variance_ratio_.sum():.3f})")

    print(f"Clustering into {N_CLUSTERS} groups (KMeans on full embeddings)...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    cluster_ids = km.fit_predict(emb)

    # --- Israel analysis ---
    try:
        israel_idx = countries.index("Israel")
    except ValueError:
        raise SystemExit("Israel not found in country list")
    sims = cosine_similarity(emb[israel_idx : israel_idx + 1], emb)[0]
    # Exclude self
    order = np.argsort(-sims)
    neighbors = [(countries[i], float(sims[i])) for i in order if i != israel_idx][:N_NEIGHBORS]

    print("\n=== Israel's nearest neighbors (cosine similarity, embedding space) ===")
    for name, s in neighbors:
        print(f"  {s:.3f}  {name}")

    # --- Per-cluster summary ---
    print("\n=== Clusters (KMeans on embeddings) ===")
    clusters: dict[int, list[str]] = {i: [] for i in range(N_CLUSTERS)}
    for name, cid in zip(countries, cluster_ids):
        clusters[cid].append(name)
    # Sort clusters by size descending for readability
    cluster_order = sorted(clusters.keys(), key=lambda k: -len(clusters[k]))
    for cid in cluster_order:
        members = clusters[cid]
        sample = ", ".join(members[:15])
        more = f" ... (+{len(members) - 15} more)" if len(members) > 15 else ""
        print(f"\nCluster {cid} ({len(members)} countries): {sample}{more}")

    israel_cluster = int(cluster_ids[israel_idx])
    print(f"\nIsrael is in cluster {israel_cluster}: "
          f"{', '.join(clusters[israel_cluster])}")

    # --- Plot ---
    print("\nRendering plot...")
    fig, ax = plt.subplots(figsize=(20, 14))
    cmap = plt.get_cmap("tab10")

    for cid in range(N_CLUSTERS):
        mask = cluster_ids == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=30,
            color=cmap(cid % 10),
            alpha=0.75,
            label=f"Cluster {cid} (n={mask.sum()})",
            edgecolors="none",
        )

    # Label every country (small font); highlight Israel and its neighbors.
    neighbor_names = {n for n, _ in neighbors}
    for i, name in enumerate(countries):
        is_israel = i == israel_idx
        is_neighbor = name in neighbor_names
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            fontsize=8 if (is_israel or is_neighbor) else 5,
            color="black" if (is_israel or is_neighbor) else "#555555",
            weight="bold" if is_israel else ("semibold" if is_neighbor else "normal"),
            alpha=1.0 if (is_israel or is_neighbor) else 0.7,
            xytext=(3, 2),
            textcoords="offset points",
        )

    # Highlight Israel with a red ring.
    ax.scatter(
        coords[israel_idx, 0],
        coords[israel_idx, 1],
        s=240,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="Israel",
        zorder=5,
    )
    # Highlight neighbors with orange rings.
    for name, _ in neighbors:
        j = countries.index(name)
        ax.scatter(
            coords[j, 0],
            coords[j, 1],
            s=120,
            facecolors="none",
            edgecolors="orange",
            linewidths=1.5,
            zorder=4,
        )

    ax.set_title(
        f"Country name embeddings ({MODEL_NAME}) -> PCA(2)\n"
        f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}. "
        f"Israel circled red; top-{N_NEIGHBORS} neighbors circled orange.",
        fontsize=13,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()

    png_path = OUT / "country_pca.png"
    fig.savefig(png_path, dpi=160)
    print(f"Saved plot -> {png_path}")

    # --- Save artifacts ---
    csv_path = OUT / "country_pca.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "pc1", "pc2", "cluster"])
        for name, (x, y), cid in zip(countries, coords, cluster_ids):
            w.writerow([name, f"{x:.6f}", f"{y:.6f}", int(cid)])
    print(f"Saved coords -> {csv_path}")

    summary = {
        "model": MODEL_NAME,
        "n_countries": len(countries),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "israel_neighbors": [
            {"country": n, "cosine_similarity": s} for n, s in neighbors
        ],
        "israel_cluster_id": israel_cluster,
        "clusters": {
            str(cid): clusters[cid] for cid in range(N_CLUSTERS)
        },
    }
    json_path = OUT / "country_pca_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {json_path}")


if __name__ == "__main__":
    main()
