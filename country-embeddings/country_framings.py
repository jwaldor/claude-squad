"""
Probe an LM with multiple contrastive prompt-pair framings to see how the
'most hated' country list shifts when the framing changes (year fixed).

Same scoring strategy as country_hate_lm.py: log P(c|negative prompt) -
log P(c|positive prompt). Length effects cancel because country tokens
are identical on both sides.

Caveat: This measures TRAINING-CORPUS associations, not truth or sentiment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycountry
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen3-4B-Base"
YEAR = 2026
TOP_K = 15
# Filter: only consider countries with raw negative-prompt log-prob above
# this threshold (everything below is in the model's noise floor).
NOISE_FLOOR = -16.0

# Each framing is a (label, negative_prompt, positive_prompt) tuple. Both
# prompts must end with a trailing space so the country tokens attach cleanly.
FRAMINGS = [
    ("hate_vs_love",
     "In 2026 we hate ",
     "In 2026 we love "),
    ("fear_vs_admire",
     "In 2026 we fear ",
     "In 2026 we admire "),
    ("dangerous_vs_peaceful",
     "In 2026 the most dangerous country in the world is ",
     "In 2026 the most peaceful country in the world is "),
    ("enemy_vs_ally",
     "In 2026 America's biggest enemy is ",
     "In 2026 America's biggest ally is "),
    ("criticize_vs_praise",
     "In 2026 people often criticize ",
     "In 2026 people often praise "),
]


@dataclass
class CountryTokens:
    name: str
    token_ids: list[int]


def country_list() -> list[str]:
    names = []
    for c in pycountry.countries:
        name = getattr(c, "common_name", None) or c.name
        names.append(name)
    return sorted(set(names))


def tokenize_countries(tok, countries):
    return [CountryTokens(name=c,
                          token_ids=tok(" " + c, add_special_tokens=False)["input_ids"])
            for c in countries]


@torch.no_grad()
def score_prompt(model, tok, prompt, countries, device):
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    by_len = {}
    for ct in countries:
        if ct.token_ids:
            by_len.setdefault(len(ct.token_ids), []).append(ct)
    results = {}
    for ntoks, items in sorted(by_len.items()):
        rows = [prompt_ids + ct.token_ids for ct in items]
        input_ids = torch.tensor(rows, device=device)
        out = model(input_ids=input_ids)
        lp = torch.log_softmax(out.logits, dim=-1)
        country_ids = torch.tensor([ct.token_ids for ct in items], device=device)
        slice_ = lp[:, prompt_len - 1:prompt_len - 1 + ntoks, :]
        gathered = torch.gather(slice_, 2, country_ids.unsqueeze(-1)).squeeze(-1)
        sums = gathered.sum(dim=1).cpu().tolist()
        for ct, s in zip(items, sums):
            results[ct.name] = float(s)
    return results


def main():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded in {time.time()-t0:.1f}s on {device}")

    countries = country_list()
    ct = tokenize_countries(tok, countries)
    names = [x.name for x in ct]

    all_results = {}
    for label, neg, pos in FRAMINGS:
        print(f"\n=== {label} ===  neg={neg!r}  pos={pos!r}")
        t0 = time.time()
        neg_lp = score_prompt(model, tok, neg, ct, device)
        t1 = time.time()
        pos_lp = score_prompt(model, tok, pos, ct, device)
        t2 = time.time()
        print(f"  scored in {t1-t0:.1f}s + {t2-t1:.1f}s")

        neg_arr = np.array([neg_lp[n] for n in names])
        pos_arr = np.array([pos_lp[n] for n in names])
        score = neg_arr - pos_arr

        # Filter to countries above noise floor (in the negative-prompt direction)
        valid = neg_arr > NOISE_FLOOR
        valid_idx = np.where(valid)[0]
        order = valid_idx[np.argsort(-score[valid_idx])]
        order_pos = valid_idx[np.argsort(score[valid_idx])]

        print(f"  Top-{TOP_K} {label.split('_')[0]}-biased (n_valid={valid.sum()}):")
        for i in order[:TOP_K]:
            print(f"    {score[i]:+.3f}  neg={neg_arr[i]:.2f}  pos={pos_arr[i]:.2f}  {names[i]}")
        print(f"  Bottom-{TOP_K} (positive-biased):")
        for i in order_pos[:TOP_K]:
            print(f"    {score[i]:+.3f}  neg={neg_arr[i]:.2f}  pos={pos_arr[i]:.2f}  {names[i]}")

        all_results[label] = {
            "negative_prompt": neg,
            "positive_prompt": pos,
            "per_country": {names[i]: {"neg": float(neg_arr[i]),
                                       "pos": float(pos_arr[i]),
                                       "score": float(score[i])}
                            for i in range(len(names))},
            "top_negative": [
                {"country": names[i], "score": float(score[i]),
                 "neg_logp": float(neg_arr[i]), "pos_logp": float(pos_arr[i])}
                for i in order[:TOP_K]
            ],
            "top_positive": [
                {"country": names[i], "score": float(score[i]),
                 "neg_logp": float(neg_arr[i]), "pos_logp": float(pos_arr[i])}
                for i in order_pos[:TOP_K]
            ],
        }

    # Save JSON
    json_path = OUT / "country_framings.json"
    json_path.write_text(json.dumps({
        "model_id": MODEL_ID, "year": YEAR, "noise_floor": NOISE_FLOOR,
        "framings": all_results,
    }, indent=2))
    print(f"\nSaved JSON -> {json_path}")

    # Build a heatmap of per-framing scores for the union of top-10 across framings
    union = []
    seen = set()
    for label, _, _ in FRAMINGS:
        for d in all_results[label]["top_negative"][:10]:
            if d["country"] not in seen:
                union.append(d["country"])
                seen.add(d["country"])
    mat = np.zeros((len(union), len(FRAMINGS)))
    for j, (label, _, _) in enumerate(FRAMINGS):
        per = all_results[label]["per_country"]
        for i, c in enumerate(union):
            mat[i, j] = per[c]["score"]

    fig, ax = plt.subplots(figsize=(11, max(8, len(union) * 0.35)))
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(FRAMINGS)))
    ax.set_xticklabels([f[0] for f in FRAMINGS], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(union)))
    ax.set_yticklabels(union, fontsize=9)
    ax.set_title(f"Country log-odds score across 5 prompt framings (year {YEAR})\n"
                 f"Model: {MODEL_ID} | red=negative-biased, blue=positive-biased",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="neg_logp - pos_logp (nats)")
    fig.tight_layout()
    fig.savefig(OUT / "country_framings_heatmap.png", dpi=140)
    plt.close(fig)
    print("Saved heatmap.")

    # Markdown table
    md = [f"# Country sentiment across 5 framings ({YEAR}) - {MODEL_ID}",
          "",
          f"Noise floor: only countries with `log P(c|negative prompt) > {NOISE_FLOOR}` "
          "are considered, to avoid statistical noise on rare multi-token names.",
          "",
          "## Top-10 most-negative-biased country per framing",
          "",
          "| Rank | " + " | ".join(label for label, _, _ in FRAMINGS) + " |",
          "|------|" + "|".join(["------"] * len(FRAMINGS)) + "|"]
    for i in range(10):
        cells = []
        for label, _, _ in FRAMINGS:
            top = all_results[label]["top_negative"]
            if i < len(top):
                d = top[i]
                cells.append(f"{d['country']} ({d['score']:+.2f})")
            else:
                cells.append("-")
        md.append(f"| {i+1} | " + " | ".join(cells) + " |")

    md.extend(["", "## Bottom-10 (positive-biased) per framing", "",
               "| Rank | " + " | ".join(label for label, _, _ in FRAMINGS) + " |",
               "|------|" + "|".join(["------"] * len(FRAMINGS)) + "|"])
    for i in range(10):
        cells = []
        for label, _, _ in FRAMINGS:
            bot = all_results[label]["top_positive"]
            if i < len(bot):
                d = bot[i]
                cells.append(f"{d['country']} ({d['score']:+.2f})")
            else:
                cells.append("-")
        md.append(f"| {i+1} | " + " | ".join(cells) + " |")

    (OUT / "country_framings_table.md").write_text("\n".join(md))
    print("Saved markdown table.")


if __name__ == "__main__":
    main()
