"""
Probe an LM with multiple contrastive prompt-pair framings across multiple
years, to see how 'most hated / most-X' rankings shift by framing and by
the year in the prompt.

Scoring: log P(c | negative prompt) - log P(c | positive prompt). The
country tokens are identical on both sides, so name length and tokenizer
frequency effects cancel.

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
YEARS = [1970, 2000, 2020, 2026]
TOP_K = 15
NOISE_FLOOR = -16.0

# Each framing: (label, negative_prompt_template, positive_prompt_template).
# Use {year} placeholder; both prompts must end with a trailing space.
FRAMINGS = [
    ("hate_vs_love",
     "In {year} we hate ",
     "In {year} we love "),
    ("fear_vs_admire",
     "In {year} we fear ",
     "In {year} we admire "),
    ("dangerous_vs_peaceful",
     "In {year} the most dangerous country in the world is ",
     "In {year} the most peaceful country in the world is "),
    ("enemy_vs_ally",
     "In {year} America's biggest enemy is ",
     "In {year} America's biggest ally is "),
    ("criticize_vs_praise",
     "In {year} people often criticize ",
     "In {year} people often praise "),
    ("college_hate_vs_love",
     "In {year} college students hate ",
     "In {year} college students love "),
]


@dataclass
class CountryTokens:
    name: str
    token_ids: list[int]


def country_list():
    names = []
    for c in pycountry.countries:
        names.append(getattr(c, "common_name", None) or c.name)
    return sorted(set(names))


def tokenize_countries(tok, countries):
    return [CountryTokens(name=c,
                          token_ids=tok(" " + c, add_special_tokens=False)["input_ids"])
            for c in countries]


@torch.no_grad()
def score_prompt(model, tok, prompt, cts, device):
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    by_len = {}
    for ct in cts:
        if ct.token_ids:
            by_len.setdefault(len(ct.token_ids), []).append(ct)
    results = {}
    for ntoks, items in sorted(by_len.items()):
        rows = [prompt_ids + ct.token_ids for ct in items]
        ids = torch.tensor(rows, device=device)
        out = model(input_ids=ids)
        lp = torch.log_softmax(out.logits, dim=-1)
        cids = torch.tensor([ct.token_ids for ct in items], device=device)
        slc = lp[:, prompt_len - 1:prompt_len - 1 + ntoks, :]
        g = torch.gather(slc, 2, cids.unsqueeze(-1)).squeeze(-1)
        s = g.sum(dim=1).cpu().tolist()
        for ct, v in zip(items, s):
            results[ct.name] = float(v)
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
    cts = tokenize_countries(tok, countries)
    names = [c.name for c in cts]

    # results[year][framing_label] = {...}
    all_results: dict = {y: {} for y in YEARS}

    total_runs = len(YEARS) * len(FRAMINGS)
    counter = 0
    for year in YEARS:
        for label, neg_tpl, pos_tpl in FRAMINGS:
            counter += 1
            neg = neg_tpl.format(year=year)
            pos = pos_tpl.format(year=year)
            print(f"\n[{counter}/{total_runs}] Year {year} | {label}")
            print(f"  neg={neg!r}")
            print(f"  pos={pos!r}")
            t0 = time.time()
            neg_lp = score_prompt(model, tok, neg, cts, device)
            t1 = time.time()
            pos_lp = score_prompt(model, tok, pos, cts, device)
            t2 = time.time()
            print(f"  neg {t1-t0:.1f}s, pos {t2-t1:.1f}s")

            neg_arr = np.array([neg_lp[n] for n in names])
            pos_arr = np.array([pos_lp[n] for n in names])
            score = neg_arr - pos_arr
            valid = neg_arr > NOISE_FLOOR
            valid_idx = np.where(valid)[0]
            order = valid_idx[np.argsort(-score[valid_idx])]
            order_pos = valid_idx[np.argsort(score[valid_idx])]

            print(f"  Top-{TOP_K} negative-biased (above noise floor; n_valid={int(valid.sum())}):")
            for i in order[:TOP_K]:
                print(f"    {score[i]:+.3f}  neg={neg_arr[i]:.2f}  pos={pos_arr[i]:.2f}  {names[i]}")

            all_results[year][label] = {
                "negative_prompt": neg,
                "positive_prompt": pos,
                "per_country": {
                    names[i]: {
                        "neg_logp": float(neg_arr[i]),
                        "pos_logp": float(pos_arr[i]),
                        "score": float(score[i]),
                    } for i in range(len(names))
                },
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
        "model_id": MODEL_ID, "years": YEARS, "noise_floor": NOISE_FLOOR,
        "framings": [f[0] for f in FRAMINGS],
        "framing_templates": {f[0]: {"neg": f[1], "pos": f[2]} for f in FRAMINGS},
        "results": all_results,
    }, indent=2))
    print(f"\nSaved JSON -> {json_path}")

    # ---- Per-framing heatmap (across years) ----
    for label, _, _ in FRAMINGS:
        union = []
        seen = set()
        for y in YEARS:
            for d in all_results[y][label]["top_negative"][:8]:
                if d["country"] not in seen:
                    union.append(d["country"])
                    seen.add(d["country"])
        if not union:
            continue
        mat = np.zeros((len(union), len(YEARS)))
        for j, y in enumerate(YEARS):
            per = all_results[y][label]["per_country"]
            for i, c in enumerate(union):
                mat[i, j] = per[c]["score"]
        fig, ax = plt.subplots(figsize=(8, max(6, len(union) * 0.32)))
        vmax = max(abs(mat.min()), abs(mat.max()))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(YEARS)))
        ax.set_xticklabels(YEARS)
        ax.set_yticks(range(len(union)))
        ax.set_yticklabels(union, fontsize=8)
        ax.set_title(f"{label}\n{MODEL_ID}", fontsize=10)
        fig.colorbar(im, ax=ax, label="neg - pos (nats)")
        fig.tight_layout()
        fig.savefig(OUT / f"framings_{label}_heatmap.png", dpi=140)
        plt.close(fig)
    print("Saved per-framing heatmaps.")

    # ---- Cross-framing heatmap for the final year ----
    final_year = YEARS[-1]
    union = []
    seen = set()
    for label, _, _ in FRAMINGS:
        for d in all_results[final_year][label]["top_negative"][:8]:
            if d["country"] not in seen:
                union.append(d["country"])
                seen.add(d["country"])
    mat = np.zeros((len(union), len(FRAMINGS)))
    for j, (label, _, _) in enumerate(FRAMINGS):
        per = all_results[final_year][label]["per_country"]
        for i, c in enumerate(union):
            mat[i, j] = per[c]["score"]
    fig, ax = plt.subplots(figsize=(11, max(8, len(union) * 0.33)))
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(FRAMINGS)))
    ax.set_xticklabels([f[0] for f in FRAMINGS], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(union)))
    ax.set_yticklabels(union, fontsize=9)
    ax.set_title(f"Country log-odds score across 6 framings (year {final_year})\n"
                 f"{MODEL_ID} | red = negative-biased, blue = positive-biased",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="neg - pos (nats)")
    fig.tight_layout()
    fig.savefig(OUT / "country_framings_heatmap.png", dpi=140)
    plt.close(fig)
    print("Saved cross-framing heatmap.")

    # ---- Markdown table: top-10 per (framing x year) ----
    md = [f"# Country sentiment across framings x years - {MODEL_ID}",
          "",
          f"Years: {YEARS}",
          "",
          f"Noise floor: only countries with `log P(c|negative) > {NOISE_FLOOR}` "
          "considered, to avoid statistical noise from rare multi-token names.",
          "",
          "Metric: `score = log P(c|negative prompt) - log P(c|positive prompt)`. "
          "Country tokens identical on both sides so length cancels. Positive = "
          "model finds country more likely under the negative prompt.",
          ""]
    for label, neg_tpl, pos_tpl in FRAMINGS:
        md.append(f"## {label}")
        md.append(f"_neg_: `{neg_tpl}` | _pos_: `{pos_tpl}`")
        md.append("")
        header = "| Rank | " + " | ".join(str(y) for y in YEARS) + " |"
        sep = "|------|" + "|".join(["------"] * len(YEARS)) + "|"
        md.append(header)
        md.append(sep)
        for i in range(10):
            cells = []
            for y in YEARS:
                top = all_results[y][label]["top_negative"]
                if i < len(top):
                    d = top[i]
                    cells.append(f"{d['country']} ({d['score']:+.2f})")
                else:
                    cells.append("-")
            md.append(f"| {i+1} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "country_framings_table.md").write_text("\n".join(md))
    print("Saved markdown.")


if __name__ == "__main__":
    main()
