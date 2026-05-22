"""
Post-process the hate-vs-love probe results: filter out countries below the
noise floor (where the model assigns vanishing probability in both prompts),
then re-render rankings, heatmap, and rank-evolution chart.

Noise-floor cutoff: hate_logp > -15. Below that the absolute probability is so
small that a half-nat fluctuation produces nonsense top-K spots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
HATE_FLOOR = -15.0
TOP_K = 15

data = json.loads((OUT / "country_hate_lm.json").read_text())
years = data["years"]
results = data["results"]

filtered_per_year = {}
for y in years:
    per = results[str(y)]["per_country"]
    above_floor = {
        c: v for c, v in per.items() if v["hate_logp"] > HATE_FLOOR
    }
    # Rank by hate_minus_love
    ranked = sorted(above_floor.items(),
                    key=lambda kv: -kv[1]["hate_minus_love"])
    filtered_per_year[y] = ranked
    print(f"\n=== Year {y} (filtered, hate_logp > {HATE_FLOOR}) ===")
    print(f"  Survived noise floor: {len(above_floor)}/{len(per)}")
    print(f"  Top-{TOP_K} hate-biased:")
    for i, (c, v) in enumerate(ranked[:TOP_K]):
        print(f"    {i+1:2d}. {v['hate_minus_love']:+.3f}  "
              f"hate={v['hate_logp']:.2f}  love={v['love_logp']:.2f}  {c}")

# ---- Build union list across years (top-10 of each year, dedup) ----
union: list[str] = []
seen: set[str] = set()
for y in years:
    for c, _ in filtered_per_year[y][:10]:
        if c not in seen:
            union.append(c)
            seen.add(c)
print(f"\nUnion of top-10 hate-biased per year: {len(union)} countries")

# ---- Heatmap ----
mat = np.zeros((len(union), len(years)))
for j, y in enumerate(years):
    per = results[str(y)]["per_country"]
    for i, c in enumerate(union):
        # If the country was below the noise floor that year, gray it out
        # (we still display the raw score for completeness)
        mat[i, j] = per[c]["hate_minus_love"]

# Mask cells where hate_logp <= floor (to show noise floor)
mask = np.zeros_like(mat, dtype=bool)
for j, y in enumerate(years):
    per = results[str(y)]["per_country"]
    for i, c in enumerate(union):
        mask[i, j] = per[c]["hate_logp"] <= HATE_FLOOR

fig, ax = plt.subplots(figsize=(13, max(8, len(union) * 0.38)))
vmax = max(abs(mat.min()), abs(mat.max()))
masked = np.ma.array(mat, mask=mask)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#dddddd")
im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)
ax.set_yticks(range(len(union)))
ax.set_yticklabels(union, fontsize=9)
ax.set_xlabel("Year in prompt")
ax.set_title(
    "Hate-vs-love bias per country, by year (noise floor filtered)\n"
    "score = log P(c | 'In <year> we hate ') - log P(c | 'In <year> we love ')\n"
    f"Gray = log P(c|hate) <= {HATE_FLOOR} (model assigns vanishing prob, "
    "result is noise).\n"
    "Red = hate-biased, blue = love-biased.  Measures TRAINING-DATA BIAS, not truth.",
    fontsize=11,
)
fig.colorbar(im, ax=ax, label="hate - love (nats)")
fig.tight_layout()
fig.savefig(OUT / "country_hate_heatmap_filtered.png", dpi=140)
plt.close(fig)
print(f"\nSaved filtered heatmap.")

# ---- Rank-evolution chart on filtered top-12 of final year ----
final = years[-1]
top_final = [c for c, _ in filtered_per_year[final][:12]]
rank_mat = np.zeros((len(top_final), len(years)))
for j, y in enumerate(years):
    ranked = filtered_per_year[y]
    rank_of = {c: i + 1 for i, (c, _) in enumerate(ranked)}
    for i, c in enumerate(top_final):
        # If a country is not in filtered list that year (below floor), give NaN
        rank_mat[i, j] = rank_of.get(c, np.nan)

fig, ax = plt.subplots(figsize=(13, 7))
cmap_lines = plt.get_cmap("tab20")
for i, c in enumerate(top_final):
    ys = rank_mat[i]
    ax.plot(years, ys, marker="o", color=cmap_lines(i % 20), label=c, linewidth=1.5)
ax.invert_yaxis()
ax.set_yscale("log")
ax.set_xlabel("Year in prompt")
ax.set_ylabel("Hate-bias rank among above-noise-floor countries (log scale)")
ax.set_title(
    f"Top-12 hate-biased countries of {final}: rank trajectory across years\n"
    "Filtered to countries with hate_logp > -15 (above model's noise floor)",
    fontsize=11,
)
ax.legend(loc="best", fontsize=8, ncol=2)
ax.grid(True, linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "country_hate_rank_evolution_filtered.png", dpi=140)
plt.close(fig)
print(f"Saved filtered rank chart.")

# ---- Markdown table (filtered) ----
md = [
    "# 'In <year> we hate <country>' vs 'we love <country>' - Qwen3-4B-Base",
    "",
    "**Model**: `Qwen/Qwen3-4B-Base` (pretraining-only, no SFT/RLHF)",
    "",
    "**Metric**:  `score = log P(c | 'In <year> we hate ') - log P(c | 'In <year> we love ')`. ",
    "Country tokens are identical in both terms, so name-length and tokenizer-",
    "frequency effects cancel. Positive = hate-biased, negative = love-biased.",
    "",
    f"**Noise-floor filter**: only includes countries with hate_logp > {HATE_FLOOR}. ",
    "Below that the absolute probability is so small that the metric is dominated ",
    "by floor noise (Svalbard, Kiribati, Micronesia repeatedly topped the unfiltered ",
    "list for that reason). Filtering away the noise leaves the countries the ",
    "model treats as plausible completions in either context.",
    "",
    "**Caveat (important)**: This shows what associations the training corpus ",
    "encoded; it is NOT a measurement of truth, of public sentiment, or of any ",
    "moral fact about any country.",
    "",
    "## Top-10 hate-biased per year (filtered)",
    "",
]
header = "| Rank | " + " | ".join(str(y) for y in years) + " |"
sep = "|------|" + "|".join(["------"] * len(years)) + "|"
md.append(header)
md.append(sep)
for i in range(10):
    cells = []
    for y in years:
        ranked = filtered_per_year[y]
        if i < len(ranked):
            c, v = ranked[i]
            cells.append(f"{c} ({v['hate_minus_love']:+.2f})")
        else:
            cells.append("-")
    md.append(f"| {i+1} | " + " | ".join(cells) + " |")

(OUT / "country_hate_table_filtered.md").write_text("\n".join(md))
print("Saved filtered markdown table.")
