"""
Compare country-name embeddings across embedding models from 2014 to 2025.

For each model: embed all 249 country names, find Israel's top-10 cosine
neighbors, run PCA -> 2D, render a per-model chart. Then build a combined
grid figure and a markdown table of Israel's neighbors over time.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pycountry
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

OUT = Path(__file__).parent
MODELS_DIR = OUT / "models"
MODELS_DIR.mkdir(exist_ok=True)
N_NEIGHBORS = 10


def country_list() -> list[str]:
    names = []
    for c in pycountry.countries:
        name = getattr(c, "common_name", None) or c.name
        names.append(name)
    return sorted(set(names))


# -----------------------------
# Encoders for each era
# -----------------------------

def encode_with_gensim_keyed_vectors(name: str, countries: list[str]) -> np.ndarray:
    """For static word vectors (GloVe, fastText, word2vec).

    Country names are often multi-word. We tokenize on whitespace + punctuation,
    lowercase, and average the in-vocab token vectors. Unknown tokens are
    skipped; if no token is in vocab we fall back to a zero vector and warn.
    """
    import gensim.downloader as gdl

    print(f"  Loading gensim keyed vectors '{name}' (may download)...")
    kv = gdl.load(name)
    dim = kv.vector_size
    out = np.zeros((len(countries), dim), dtype=np.float32)
    misses_total = 0
    no_hit = []
    for i, c in enumerate(countries):
        toks = [
            t.strip(",.'\"()[]")
            for t in c.lower().replace("-", " ").replace(",", " ").split()
        ]
        toks = [t for t in toks if t]
        vecs = []
        for t in toks:
            if t in kv:
                vecs.append(kv[t])
            else:
                misses_total += 1
        if vecs:
            out[i] = np.mean(vecs, axis=0)
        else:
            no_hit.append(c)
    if no_hit:
        print(f"  WARN: {len(no_hit)} countries had no in-vocab tokens (first few: {no_hit[:5]})")
    # L2 normalize
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = out / norms
    return out


def encode_with_sentence_transformer(model_id: str, countries: list[str],
                                     prompt: Optional[str] = None,
                                     trust_remote: bool = False) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading sentence-transformer '{model_id}'...")
    kwargs = {}
    if trust_remote:
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_id, **kwargs)
    inputs = countries
    if prompt:
        inputs = [f"{prompt}{c}" for c in countries]
    emb = model.encode(inputs, show_progress_bar=False, normalize_embeddings=True,
                       batch_size=32)
    return np.asarray(emb)


def encode_with_hf_mean_pool(model_id: str, countries: list[str]) -> np.ndarray:
    """Generic HF encoder model with mean pooling. Used for raw BERT-base 2018."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"  Loading HF model '{model_id}'...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    out = []
    with torch.no_grad():
        batch = 16
        for i in range(0, len(countries), batch):
            chunk = countries[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, return_tensors="pt")
            res = model(**enc)
            # mean pool over non-pad tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (res.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean = summed / counts
            out.append(mean.cpu().numpy())
    arr = np.concatenate(out, axis=0)
    arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
    return arr


# -----------------------------
# Model registry
# -----------------------------

@dataclass
class ModelSpec:
    year: int
    label: str           # short display label
    description: str     # one-liner
    encode: Callable[[list[str]], np.ndarray]

MODELS: list[ModelSpec] = [
    ModelSpec(
        2014, "GloVe-300", "GloVe 6B 300d (Stanford, 2014; Wikipedia+Gigaword)",
        lambda c: encode_with_gensim_keyed_vectors("glove-wiki-gigaword-300", c),
    ),
    ModelSpec(
        2017, "fastText-300", "fastText subword 300d (Facebook, 2017; wiki-news)",
        lambda c: encode_with_gensim_keyed_vectors("fasttext-wiki-news-subwords-300", c),
    ),
    ModelSpec(
        2018, "BERT-base", "bert-base-uncased mean-pool (Google, 2018)",
        lambda c: encode_with_hf_mean_pool("bert-base-uncased", c),
    ),
    ModelSpec(
        2019, "SBERT-NLI", "SBERT bert-base-nli-mean-tokens (Reimers, 2019)",
        lambda c: encode_with_sentence_transformer(
            "sentence-transformers/bert-base-nli-mean-tokens", c),
    ),
    ModelSpec(
        2021, "MiniLM-L6", "all-MiniLM-L6-v2 (2021)",
        lambda c: encode_with_sentence_transformer("all-MiniLM-L6-v2", c),
    ),
    ModelSpec(
        2022, "MPNet", "all-mpnet-base-v2 (2021/2022 SBERT family)",
        lambda c: encode_with_sentence_transformer("all-mpnet-base-v2", c),
    ),
    ModelSpec(
        2023, "BGE-small", "BAAI/bge-small-en-v1.5 (2023)",
        lambda c: encode_with_sentence_transformer("BAAI/bge-small-en-v1.5", c),
    ),
    ModelSpec(
        2024, "mxbai-large", "mixedbread-ai/mxbai-embed-large-v1 (2024)",
        lambda c: encode_with_sentence_transformer("mixedbread-ai/mxbai-embed-large-v1", c),
    ),
    ModelSpec(
        2025, "nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5 (2024/2025)",
        # nomic expects a task prefix; use search_document for general semantic embed
        lambda c: encode_with_sentence_transformer(
            "nomic-ai/nomic-embed-text-v1.5", c,
            prompt="search_document: ", trust_remote=True),
    ),
]


# -----------------------------
# Per-model analysis
# -----------------------------

@dataclass
class ModelResult:
    spec: ModelSpec
    coords: np.ndarray        # (n, 2)
    explained: np.ndarray     # (2,)
    israel_neighbors: list[tuple[str, float]] = field(default_factory=list)
    error: Optional[str] = None


def analyze(spec: ModelSpec, countries: list[str]) -> ModelResult:
    print(f"\n=== {spec.year} {spec.label} : {spec.description} ===")
    try:
        emb = spec.encode(countries)
        if emb.ndim != 2 or emb.shape[0] != len(countries):
            raise RuntimeError(f"bad emb shape: {emb.shape}")
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(emb)

        israel_idx = countries.index("Israel")
        sims = cosine_similarity(emb[israel_idx:israel_idx + 1], emb)[0]
        order = np.argsort(-sims)
        neighbors = [(countries[i], float(sims[i])) for i in order
                     if i != israel_idx][:N_NEIGHBORS]
        print("  Israel neighbors:")
        for n, s in neighbors:
            print(f"    {s:.3f}  {n}")
        return ModelResult(
            spec=spec, coords=coords,
            explained=pca.explained_variance_ratio_,
            israel_neighbors=neighbors,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return ModelResult(
            spec=spec,
            coords=np.zeros((len(countries), 2)),
            explained=np.zeros(2),
            error=str(e),
        )


# -----------------------------
# Plotting
# -----------------------------

def plot_single(result: ModelResult, countries: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    israel_idx = countries.index("Israel")
    neighbor_names = {n for n, _ in result.israel_neighbors}

    ax.scatter(result.coords[:, 0], result.coords[:, 1],
               s=14, alpha=0.5, color="#888", edgecolors="none")
    # Highlight Israel + neighbors
    for i, name in enumerate(countries):
        if i == israel_idx:
            ax.scatter(result.coords[i, 0], result.coords[i, 1],
                       s=180, facecolors="none", edgecolors="red", linewidths=2.0, zorder=5)
        elif name in neighbor_names:
            ax.scatter(result.coords[i, 0], result.coords[i, 1],
                       s=80, facecolors="none", edgecolors="orange", linewidths=1.5, zorder=4)
        if i == israel_idx or name in neighbor_names:
            ax.annotate(name, (result.coords[i, 0], result.coords[i, 1]),
                        fontsize=9, weight="bold" if i == israel_idx else "semibold",
                        xytext=(4, 3), textcoords="offset points")

    ev = result.explained.sum()
    ax.set_title(f"{result.spec.year} - {result.spec.label}\n"
                 f"{result.spec.description}\n"
                 f"PCA explained: {ev:.1%}", fontsize=10)
    ax.set_xlabel(f"PC1 ({result.explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({result.explained[1]:.1%})")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_grid(results: list[ModelResult], countries: list[str], path: Path) -> None:
    ok = [r for r in results if r.error is None]
    n = len(ok)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    israel_idx = countries.index("Israel")

    for ax, r in zip(axes_flat, ok):
        neighbor_names = {n for n, _ in r.israel_neighbors}
        ax.scatter(r.coords[:, 0], r.coords[:, 1],
                   s=8, alpha=0.4, color="#888", edgecolors="none")
        # Plot Israel + neighbors
        for i, name in enumerate(countries):
            if i == israel_idx:
                ax.scatter(r.coords[i, 0], r.coords[i, 1],
                           s=120, facecolors="none", edgecolors="red", linewidths=1.6, zorder=5)
                ax.annotate("Israel", (r.coords[i, 0], r.coords[i, 1]),
                            fontsize=8, weight="bold", color="red",
                            xytext=(4, 3), textcoords="offset points")
            elif name in neighbor_names:
                ax.scatter(r.coords[i, 0], r.coords[i, 1],
                           s=40, facecolors="none", edgecolors="orange", linewidths=1.0, zorder=4)
                ax.annotate(name, (r.coords[i, 0], r.coords[i, 1]),
                            fontsize=6, color="darkorange",
                            xytext=(3, 2), textcoords="offset points")
        ax.set_title(f"{r.spec.year}  {r.spec.label}\n"
                     f"PCA exp. {r.explained.sum():.1%}",
                     fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.3)

    # Empty axes
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle("Country-name embeddings, 2014 -> 2025\n"
                 "Israel circled red; top-10 cosine neighbors circled orange.",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_neighbors_table(results: list[ModelResult], path: Path) -> None:
    ok = [r for r in results if r.error is None]
    lines = ["# Israel's top-10 nearest neighbors per model",
             "",
             "Cosine similarity in the model's full embedding space.",
             "Country names embedded; multi-token names averaged for static-vector models.",
             ""]
    # Per-model bullet lists
    for r in ok:
        lines.append(f"## {r.spec.year} - {r.spec.label}")
        lines.append(f"_{r.spec.description}_")
        lines.append("")
        lines.append("| # | Country | cosine |")
        lines.append("|---|---------|--------|")
        for i, (n, s) in enumerate(r.israel_neighbors, 1):
            lines.append(f"| {i} | {n} | {s:.3f} |")
        lines.append("")
    # Compact side-by-side rank table (rank x model)
    lines.append("## Rank table - Israel neighbors across models")
    lines.append("")
    header = "| Rank | " + " | ".join(f"{r.spec.year} {r.spec.label}" for r in ok) + " |"
    sep = "|------|" + "|".join(["------"] * len(ok)) + "|"
    lines.append(header)
    lines.append(sep)
    for i in range(N_NEIGHBORS):
        cells = []
        for r in ok:
            if i < len(r.israel_neighbors):
                n, s = r.israel_neighbors[i]
                cells.append(f"{n} ({s:.2f})")
            else:
                cells.append("-")
        lines.append(f"| {i+1} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines))


def main() -> None:
    countries = country_list()
    print(f"Loaded {len(countries)} countries")

    results: list[ModelResult] = []
    for spec in MODELS:
        r = analyze(spec, countries)
        results.append(r)
        if r.error is None:
            slug = f"{r.spec.year}_{r.spec.label.replace('/', '-')}"
            mdir = MODELS_DIR / slug
            mdir.mkdir(exist_ok=True)
            plot_single(r, countries, mdir / "pca.png")
            np.savetxt(mdir / "coords.csv", r.coords, delimiter=",",
                       header="pc1,pc2", comments="")
            (mdir / "neighbors.json").write_text(json.dumps({
                "year": r.spec.year,
                "label": r.spec.label,
                "description": r.spec.description,
                "explained_variance": r.explained.tolist(),
                "israel_neighbors": [
                    {"country": n, "cosine": s} for n, s in r.israel_neighbors
                ],
            }, indent=2))

    print("\nWriting combined outputs...")
    plot_grid(results, countries, OUT / "comparison.png")
    write_neighbors_table(results, OUT / "neighbors_over_time.md")

    summary = {
        "n_countries": len(countries),
        "models": [
            {
                "year": r.spec.year,
                "label": r.spec.label,
                "description": r.spec.description,
                "error": r.error,
                "explained_variance": r.explained.tolist() if r.error is None else None,
                "israel_neighbors": [
                    {"country": n, "cosine": s} for n, s in r.israel_neighbors
                ] if r.error is None else None,
            }
            for r in results
        ],
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
