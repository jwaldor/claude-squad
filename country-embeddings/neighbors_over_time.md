# Israel's top-10 nearest neighbors per model

Cosine similarity in the model's full embedding space.
Country names embedded; multi-token names averaged for static-vector models.

## 2014 - GloVe-300
_GloVe 6B 300d (Stanford, 2014; Wikipedia+Gigaword)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Lebanon | 0.665 |
| 2 | Syria | 0.663 |
| 3 | Egypt | 0.581 |
| 4 | United Arab Emirates | 0.535 |
| 5 | Palestine, State of | 0.520 |
| 6 | Jordan | 0.511 |
| 7 | Iran | 0.475 |
| 8 | United States | 0.432 |
| 9 | French Southern Territories | 0.415 |
| 10 | Libya | 0.408 |

## 2017 - fastText-300
_fastText subword 300d (Facebook, 2017; wiki-news)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Lebanon | 0.685 |
| 2 | Syria | 0.656 |
| 3 | Iraq | 0.622 |
| 4 | Iran | 0.620 |
| 5 | Egypt | 0.614 |
| 6 | Jordan | 0.599 |
| 7 | Saudi Arabia | 0.590 |
| 8 | Pakistan | 0.577 |
| 9 | Ukraine | 0.553 |
| 10 | United Arab Emirates | 0.553 |

## 2018 - BERT-base
_bert-base-uncased mean-pool (Google, 2018)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Lebanon | 0.883 |
| 2 | Egypt | 0.880 |
| 3 | Oman | 0.866 |
| 4 | Iran | 0.860 |
| 5 | Sudan | 0.849 |
| 6 | India | 0.848 |
| 7 | Yemen | 0.846 |
| 8 | Morocco | 0.846 |
| 9 | Tunisia | 0.845 |
| 10 | China | 0.841 |

## 2019 - SBERT-NLI
_SBERT bert-base-nli-mean-tokens (Reimers, 2019)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Palestine, State of | 0.823 |
| 2 | Lebanon | 0.702 |
| 3 | Egypt | 0.686 |
| 4 | Syria | 0.670 |
| 5 | Jordan | 0.661 |
| 6 | Oman | 0.651 |
| 7 | India | 0.643 |
| 8 | Eritrea | 0.641 |
| 9 | Netherlands | 0.640 |
| 10 | Malaysia | 0.637 |

## 2021 - MiniLM-L6
_all-MiniLM-L6-v2 (2021)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Iran | 0.723 |
| 2 | Palestine, State of | 0.720 |
| 3 | Lebanon | 0.703 |
| 4 | Syria | 0.673 |
| 5 | Switzerland | 0.665 |
| 6 | Egypt | 0.661 |
| 7 | Saudi Arabia | 0.656 |
| 8 | Iraq | 0.626 |
| 9 | China | 0.626 |
| 10 | North Korea | 0.614 |

## 2022 - MPNet
_all-mpnet-base-v2 (2021/2022 SBERT family)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Lebanon | 0.749 |
| 2 | Syria | 0.714 |
| 3 | Egypt | 0.702 |
| 4 | Iran | 0.700 |
| 5 | Palestine, State of | 0.690 |
| 6 | Iraq | 0.687 |
| 7 | Kuwait | 0.660 |
| 8 | Libya | 0.658 |
| 9 | Tunisia | 0.657 |
| 10 | Morocco | 0.640 |

## 2023 - BGE-small
_BAAI/bge-small-en-v1.5 (2023)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Lebanon | 0.716 |
| 2 | Palestine, State of | 0.705 |
| 3 | Iran | 0.697 |
| 4 | North Korea | 0.697 |
| 5 | Syria | 0.697 |
| 6 | Armenia | 0.694 |
| 7 | Switzerland | 0.693 |
| 8 | Jordan | 0.690 |
| 9 | Egypt | 0.689 |
| 10 | Argentina | 0.685 |

## 2024 - mxbai-large
_mixedbread-ai/mxbai-embed-large-v1 (2024)_

| # | Country | cosine |
|---|---------|--------|
| 1 | Jordan | 0.728 |
| 2 | Lebanon | 0.719 |
| 3 | Palestine, State of | 0.707 |
| 4 | Ireland | 0.704 |
| 5 | Egypt | 0.701 |
| 6 | Iran | 0.698 |
| 7 | India | 0.689 |
| 8 | Syria | 0.684 |
| 9 | Armenia | 0.682 |
| 10 | Iceland | 0.677 |

## 2025 - nomic-v1.5
_nomic-ai/nomic-embed-text-v1.5 (2024/2025)_

| # | Country | cosine |
|---|---------|--------|
| 1 | India | 0.831 |
| 2 | Egypt | 0.826 |
| 3 | Iran | 0.826 |
| 4 | Kenya | 0.821 |
| 5 | Indonesia | 0.812 |
| 6 | Oman | 0.804 |
| 7 | Syria | 0.799 |
| 8 | China | 0.799 |
| 9 | Ethiopia | 0.798 |
| 10 | Eritrea | 0.797 |

## Rank table - Israel neighbors across models

| Rank | 2014 GloVe-300 | 2017 fastText-300 | 2018 BERT-base | 2019 SBERT-NLI | 2021 MiniLM-L6 | 2022 MPNet | 2023 BGE-small | 2024 mxbai-large | 2025 nomic-v1.5 |
|------|------|------|------|------|------|------|------|------|------|
| 1 | Lebanon (0.66) | Lebanon (0.68) | Lebanon (0.88) | Palestine, State of (0.82) | Iran (0.72) | Lebanon (0.75) | Lebanon (0.72) | Jordan (0.73) | India (0.83) |
| 2 | Syria (0.66) | Syria (0.66) | Egypt (0.88) | Lebanon (0.70) | Palestine, State of (0.72) | Syria (0.71) | Palestine, State of (0.71) | Lebanon (0.72) | Egypt (0.83) |
| 3 | Egypt (0.58) | Iraq (0.62) | Oman (0.87) | Egypt (0.69) | Lebanon (0.70) | Egypt (0.70) | Iran (0.70) | Palestine, State of (0.71) | Iran (0.83) |
| 4 | United Arab Emirates (0.54) | Iran (0.62) | Iran (0.86) | Syria (0.67) | Syria (0.67) | Iran (0.70) | North Korea (0.70) | Ireland (0.70) | Kenya (0.82) |
| 5 | Palestine, State of (0.52) | Egypt (0.61) | Sudan (0.85) | Jordan (0.66) | Switzerland (0.67) | Palestine, State of (0.69) | Syria (0.70) | Egypt (0.70) | Indonesia (0.81) |
| 6 | Jordan (0.51) | Jordan (0.60) | India (0.85) | Oman (0.65) | Egypt (0.66) | Iraq (0.69) | Armenia (0.69) | Iran (0.70) | Oman (0.80) |
| 7 | Iran (0.47) | Saudi Arabia (0.59) | Yemen (0.85) | India (0.64) | Saudi Arabia (0.66) | Kuwait (0.66) | Switzerland (0.69) | India (0.69) | Syria (0.80) |
| 8 | United States (0.43) | Pakistan (0.58) | Morocco (0.85) | Eritrea (0.64) | Iraq (0.63) | Libya (0.66) | Jordan (0.69) | Syria (0.68) | China (0.80) |
| 9 | French Southern Territories (0.41) | Ukraine (0.55) | Tunisia (0.84) | Netherlands (0.64) | China (0.63) | Tunisia (0.66) | Egypt (0.69) | Armenia (0.68) | Ethiopia (0.80) |
| 10 | Libya (0.41) | United Arab Emirates (0.55) | China (0.84) | Malaysia (0.64) | North Korea (0.61) | Morocco (0.64) | Argentina (0.68) | Iceland (0.68) | Eritrea (0.80) |