"""
Probe an LM for country sentiment bias via a contrastive prompt pair.

For each country c and year y, compute:
    hate_logp(c, y) = log P(c | "In <y> we hate ")
    love_logp(c, y) = log P(c | "In <y> we love ")
    score(c, y)     = hate_logp - love_logp     [log-odds of hate vs love]

Because the country tokens are identical in both forwards, any
length / tokenizer-frequency effect cancels exactly. The score is the
direction (+ hate-biased / - love-biased) that the prompt frame pushes
each country.

NOTE: This measures what associations the training corpus encoded. It is
NOT a measurement of truth, of public sentiment, or of any moral fact
about any country.
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
YEARS = [1950, 1970, 1990, 2000, 2010, 2020, 2024, 2026]
HATE_PROMPT = "In {year} we hate "
LOVE_PROMPT = "In {year} we love "
TOP_K = 20


def country_list() -> list[str]:
    names = []
    for c in pycountry.countries:
        name = getattr(c, "common_name", None) or c.name
        names.append(name)
    return sorted(set(names))


@dataclass
class CountryTokens:
    name: str
    token_ids: list[int]


def tokenize_countries(tok, countries: list[str]) -> list[CountryTokens]:
    out = []
    for c in countries:
        ids = tok(" " + c, add_special_tokens=False)["input_ids"]
        out.append(CountryTokens(name=c, token_ids=ids))
    return out


@torch.no_grad()
def score_prompt(
    model,
    tok,
    prompt: str,
    countries: list[CountryTokens],
    device: str,
    pad_id: int,
) -> dict[str, float]:
    """Return {country_name: sum_log_prob} for the given prompt.

    Batched by total sequence length so countries of the same length run
    in a single forward pass.
    """
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    by_len: dict[int, list[CountryTokens]] = {}
    for ct in countries:
        if not ct.token_ids:
            continue
        by_len.setdefault(len(ct.token_ids), []).append(ct)

    results: dict[str, float] = {}
    for ntoks, items in sorted(by_len.items()):
        # Build (B, prompt_len + ntoks) input tensor: prompt + country tokens
        rows = [prompt_ids + ct.token_ids for ct in items]
        input_ids = torch.tensor(rows, device=device)
        out = model(input_ids=input_ids)
        log_probs = torch.log_softmax(out.logits, dim=-1)  # (B, L, V)

        # For each item, position prompt_len-1+k produces a logit predicting
        # token ct.token_ids[k] (for k in 0..ntoks-1).
        # Gather log-probs of country tokens.
        country_ids = torch.tensor([ct.token_ids for ct in items],
                                   device=device)  # (B, ntoks)
        # Positions to read: prompt_len-1, prompt_len, ..., prompt_len-1+ntoks-1
        pos_start = prompt_len - 1
        # log_probs at positions [pos_start : pos_start + ntoks]
        # slice -> (B, ntoks, V)
        log_probs_slice = log_probs[:, pos_start:pos_start + ntoks, :]
        # gather along last dim
        gathered = torch.gather(
            log_probs_slice, 2, country_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, ntoks)
        sum_lp = gathered.sum(dim=1).cpu().tolist()  # length B
        for ct, lp in zip(items, sum_lp):
            results[ct.name] = float(lp)
    return results


def main() -> None:
    print(f"Loading model {MODEL_ID}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Loaded in {time.time()-t0:.1f}s on {device}, dtype={dtype}")

    countries = country_list()
    print(f"{len(countries)} countries")
    ct = tokenize_countries(tok, countries)
    lengths = [len(x.token_ids) for x in ct]
    print(f"Token lengths: mean {np.mean(lengths):.1f}, max {max(lengths)}, "
          f"unique buckets: {len(set(lengths))}")

    all_year_results = {}
    for year in YEARS:
        hp = HATE_PROMPT.format(year=year)
        lp_str = LOVE_PROMPT.format(year=year)
        print(f"\n=== Year {year} ===  hate={hp!r}  love={lp_str!r}")

        t0 = time.time()
        hate_logp = score_prompt(model, tok, hp, ct, device, pad_id)
        t1 = time.time()
        love_logp = score_prompt(model, tok, lp_str, ct, device, pad_id)
        t2 = time.time()
        print(f"  hate scored in {t1-t0:.1f}s, love in {t2-t1:.1f}s")

        names = [x.name for x in ct]
        hate_arr = np.array([hate_logp[n] for n in names])
        love_arr = np.array([love_logp[n] for n in names])
        score = hate_arr - love_arr  # log-odds hate vs love

        # Rankings
        order_hate_bias = np.argsort(-score)
        order_love_bias = np.argsort(score)
        order_raw_hate = np.argsort(-hate_arr)

        print(f"  Top-{TOP_K} HATE-biased (score = log P(c|hate) - log P(c|love)):")
        for i in order_hate_bias[:15]:
            print(f"    {score[i]:+.3f}  hate={hate_arr[i]:.2f}  love={love_arr[i]:.2f}  {names[i]}")

        print(f"  Top-10 LOVE-biased:")
        for i in order_love_bias[:10]:
            print(f"    {score[i]:+.3f}  hate={hate_arr[i]:.2f}  love={love_arr[i]:.2f}  {names[i]}")

        all_year_results[year] = {
            "hate_prompt": hp,
            "love_prompt": lp_str,
            "per_country": {
                names[i]: {
                    "hate_logp": float(hate_arr[i]),
                    "love_logp": float(love_arr[i]),
                    "hate_minus_love": float(score[i]),
                    "n_tokens": int(lengths[i]),
                } for i in range(len(names))
            },
            "top_hate_biased": [
                {"country": names[i], "score": float(score[i]),
                 "hate_logp": float(hate_arr[i]), "love_logp": float(love_arr[i])}
                for i in order_hate_bias[:TOP_K]
            ],
            "top_love_biased": [
                {"country": names[i], "score": float(score[i]),
                 "hate_logp": float(hate_arr[i]), "love_logp": float(love_arr[i])}
                for i in order_love_bias[:TOP_K]
            ],
            "top_raw_hate": [
                {"country": names[i], "hate_logp": float(hate_arr[i]),
                 "n_tokens": lengths[i]}
                for i in order_raw_hate[:TOP_K]
            ],
        }

    # ---- Save JSON ----
    json_path = OUT / "country_hate_lm.json"
    json_path.write_text(json.dumps({
        "model_id": MODEL_ID,
        "hate_prompt_template": HATE_PROMPT,
        "love_prompt_template": LOVE_PROMPT,
        "years": YEARS,
        "n_countries": len(countries),
        "metric": "hate_minus_love = log P(country | 'we hate') - log P(country | 'we love')",
        "results": all_year_results,
    }, indent=2))
    print(f"\nSaved JSON -> {json_path}")

    # ---- Heatmap: union of top-10 hate-biased per year, score over years ----
    union = []
    seen = set()
    for y in YEARS:
        for d in all_year_results[y]["top_hate_biased"][:10]:
            if d["country"] not in seen:
                union.append(d["country"])
                seen.add(d["country"])
    mat = np.zeros((len(union), len(YEARS)))
    for j, y in enumerate(YEARS):
        for i, c in enumerate(union):
            mat[i, j] = all_year_results[y]["per_country"][c]["hate_minus_love"]

    fig, ax = plt.subplots(figsize=(13, max(8, len(union) * 0.35)))
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS)
    ax.set_yticks(range(len(union)))
    ax.set_yticklabels(union, fontsize=9)
    ax.set_xlabel("Year in prompt")
    ax.set_title(f"log P(c | 'In <year> we hate ') - log P(c | 'In <year> we love ')\n"
                 f"Model: {MODEL_ID}.  Red = hate-biased, blue = love-biased.\n"
                 f"This shows TRAINING-DATA BIAS, not truth or sentiment.",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="hate - love (nats)")
    fig.tight_layout()
    png_path = OUT / "country_hate_heatmap.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved heatmap -> {png_path}")

    # ---- Year-over-year rank tracking (top-12 of final year) ----
    final = YEARS[-1]
    plot_set = [d["country"]
                for d in all_year_results[final]["top_hate_biased"][:12]]
    rank_mat = np.zeros((len(plot_set), len(YEARS)))
    for j, y in enumerate(YEARS):
        per = all_year_results[y]["per_country"]
        ordered = sorted(per.keys(),
                         key=lambda c: -per[c]["hate_minus_love"])
        rank_of = {c: i + 1 for i, c in enumerate(ordered)}
        for i, c in enumerate(plot_set):
            rank_mat[i, j] = rank_of[c]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(plot_set):
        ax.plot(YEARS, rank_mat[i], marker="o", color=cmap(i % 20), label=c)
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_xlabel("Year in prompt")
    ax.set_ylabel("Hate-bias rank (1 = highest), log scale")
    ax.set_title(f"Top-12 hate-biased countries of {final}: rank trajectory across years\n"
                 f"{MODEL_ID}", fontsize=11)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    rank_path = OUT / "country_hate_rank_evolution.png"
    fig.savefig(rank_path, dpi=140)
    plt.close(fig)
    print(f"Saved rank chart -> {rank_path}")

    # ---- Markdown table ----
    md = [
        "# 'In <year> we hate <country>' vs 'we love <country>' - Qwen3-4B-Base",
        "",
        f"Model: `{MODEL_ID}`  (base / not instruction-tuned)",
        "",
        "**Metric**: `score = log P(c | 'In <year> we hate ') - log P(c | 'In <year> we love ')`. "
        "Country tokens are identical in both terms, so name-length and tokenizer-frequency "
        "effects cancel. Positive = model finds the country more likely after 'we hate'; "
        "negative = more likely after 'we love'.",
        "",
        "**Caveat (important)**: This shows what associations the training corpus "
        "encoded; it is NOT a measurement of truth, of public sentiment, or of any moral "
        "fact about any country.",
        "",
        "## Top-10 hate-biased per year",
        "",
    ]
    header = "| Rank | " + " | ".join(str(y) for y in YEARS) + " |"
    sep = "|------|" + "|".join(["------"] * len(YEARS)) + "|"
    md.append(header)
    md.append(sep)
    for i in range(10):
        cells = []
        for y in YEARS:
            d = all_year_results[y]["top_hate_biased"][i]
            cells.append(f"{d['country']} ({d['score']:+.2f})")
        md.append(f"| {i+1} | " + " | ".join(cells) + " |")

    md.extend([
        "",
        "## Top-10 love-biased per year (lowest score = most love-biased)",
        "",
        header,
        sep,
    ])
    for i in range(10):
        cells = []
        for y in YEARS:
            d = all_year_results[y]["top_love_biased"][i]
            cells.append(f"{d['country']} ({d['score']:+.2f})")
        md.append(f"| {i+1} | " + " | ".join(cells) + " |")

    (OUT / "country_hate_table.md").write_text("\n".join(md))
    print("Saved markdown table.")


if __name__ == "__main__":
    main()
