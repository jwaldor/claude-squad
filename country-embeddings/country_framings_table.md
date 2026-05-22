# Country sentiment across framings x years - Qwen/Qwen3-4B-Base

Years: [1970, 2000, 2020, 2026]

Noise floor: only countries with `log P(c|negative) > -16.0` considered, to avoid statistical noise from rare multi-token names.

Metric: `score = log P(c|negative prompt) - log P(c|positive prompt)`. Country tokens identical on both sides so length cancels. Positive = model finds country more likely under the negative prompt.

## hate_vs_love
_neg_: `In {year} we hate ` | _pos_: `In {year} we love `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | North Korea (+1.88) | Namibia (+9.50) | North Korea (+2.31) | Iraq (+2.56) |
| 2 | Iraq (+1.75) | North Korea (+1.19) | China (+1.81) | Iran (+2.50) |
| 3 | China (+1.50) | Colombia (+1.06) | Iraq (+1.75) | North Korea (+1.88) |
| 4 | Iran (+1.50) | Kazakhstan (+1.06) | Israel (+1.69) | Israel (+1.62) |
| 5 | Rwanda (+1.50) | Bangladesh (+1.00) | Iran (+1.38) | South Korea (+1.62) |
| 6 | Malaysia (+1.44) | Canada (+0.94) | Qatar (+1.12) | China (+1.44) |
| 7 | Bangladesh (+1.44) | Taiwan (+0.94) | Ukraine (+0.94) | Syria (+1.38) |
| 8 | Canada (+1.38) | China (+0.81) | United States (+0.94) | Yemen (+1.38) |
| 9 | Libya (+1.38) | Venezuela (+0.81) | Malaysia (+0.88) | Saudi Arabia (+1.12) |
| 10 | Haiti (+1.25) | Iraq (+0.75) | Canada (+0.81) | Cuba (+1.12) |

## fear_vs_admire
_neg_: `In {year} we fear ` | _pos_: `In {year} we admire `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | Libya (+2.56) | Yemen (+3.06) | North Korea (+2.62) | Ukraine (+2.88) |
| 2 | Iraq (+2.25) | Iraq (+2.56) | Iraq (+2.12) | North Korea (+2.50) |
| 3 | Somalia (+2.06) | Botswana (+2.06) | Syria (+1.50) | Afghanistan (+2.44) |
| 4 | Bangladesh (+2.00) | Bangladesh (+2.06) | China (+1.44) | Pakistan (+2.38) |
| 5 | Pakistan (+1.94) | Pakistan (+2.06) | Egypt (+1.38) | Nigeria (+2.38) |
| 6 | Puerto Rico (+1.88) | Vietnam (+2.00) | Afghanistan (+1.19) | Venezuela (+2.38) |
| 7 | Greenland (+1.75) | Canada (+2.00) | Puerto Rico (+1.06) | Yemen (+2.38) |
| 8 | Uganda (+1.69) | North Korea (+1.94) | Cuba (+0.88) | Iraq (+2.31) |
| 9 | Ukraine (+1.62) | Libya (+1.88) | El Salvador (+0.75) | Malaysia (+2.31) |
| 10 | Indonesia (+1.50) | Afghanistan (+1.81) | Mexico (+0.75) | Iran (+2.19) |

## dangerous_vs_peaceful
_neg_: `In {year} the most dangerous country in the world is ` | _pos_: `In {year} the most peaceful country in the world is `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | Faroe Islands (+7.12) | Burundi (+10.19) | Iraq (+5.25) | Libya (+2.62) |
| 2 | South Africa (+1.75) | South Africa (+4.06) | Pakistan (+5.03) | Yemen (+2.59) |
| 3 | Cameroon (+1.25) | Pakistan (+3.09) | Yemen (+4.88) | South Sudan (+2.06) |
| 4 | North Korea (+1.12) | Iraq (+3.06) | Syria (+4.28) | Iraq (+2.06) |
| 5 | Pakistan (+1.09) | Syria (+2.94) | South Sudan (+4.25) | Iran (+2.00) |
| 6 | Iran (+1.06) | South Sudan (+2.69) | Libya (+4.09) | Pakistan (+2.00) |
| 7 | Libya (+1.06) | Mali (+2.69) | Russian Federation (+4.06) | Syria (+1.75) |
| 8 | Zimbabwe (+1.00) | Sudan (+2.66) | Iran (+4.03) | Somalia (+1.56) |
| 9 | Syria (+0.94) | Russian Federation (+2.56) | Niger (+4.00) | Afghanistan (+1.53) |
| 10 | Iraq (+0.88) | Yemen (+2.50) | Somalia (+3.94) | Sudan (+1.41) |

## enemy_vs_ally
_neg_: `In {year} America's biggest enemy is ` | _pos_: `In {year} America's biggest ally is `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | Jersey (+1.50) | Jersey (+0.81) | Honduras (-1.00) | Montserrat (+0.88) |
| 2 | Antarctica (+1.25) | Antarctica (+0.19) | Guinea (-1.31) | Jersey (+0.25) |
| 3 | United States (+1.25) | North Korea (+0.19) | North Korea (-1.44) | Anguilla (+0.19) |
| 4 | North Korea (+1.00) | Puerto Rico (+0.06) | Burundi (-1.50) | United States (+0.06) |
| 5 | Puerto Rico (+1.00) | South Africa (-0.12) | Congo (-1.56) | Guinea (-0.06) |
| 6 | Bangladesh (+1.00) | American Samoa (-0.38) | Greenland (-1.69) | Antarctica (-0.12) |
| 7 | Colombia (+0.75) | Tonga (-0.38) | Sierra Leone (-1.94) | Greenland (-0.25) |
| 8 | Dominican Republic (+0.75) | Bermuda (-0.44) | Namibia (-2.25) | Colombia (-0.25) |
| 9 | Barbados (+0.69) | Congo (-0.44) | South Sudan (-2.31) | Barbados (-0.25) |
| 10 | American Samoa (+0.69) | Burundi (-0.44) | Puerto Rico (-2.38) | Chad (-0.31) |

## criticize_vs_praise
_neg_: `In {year} people often criticize ` | _pos_: `In {year} people often praise `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | North Korea (+12.19) | Eritrea (+2.50) | Russian Federation (+3.12) | Czechia (+2.25) |
| 2 | Venezuela (+2.38) | Iran (+2.31) | El Salvador (+2.94) | United States (+2.19) |
| 3 | Uruguay (+2.00) | Syria (+2.00) | Eritrea (+2.75) | Ukraine (+2.00) |
| 4 | Tuvalu (+2.00) | Afghanistan (+1.88) | Ukraine (+2.75) | Iraq (+1.81) |
| 5 | Vietnam (+1.81) | El Salvador (+1.88) | Andorra (+2.69) | United Kingdom (+1.81) |
| 6 | Somalia (+1.62) | Iraq (+1.81) | South Sudan (+2.56) | Iran (+1.69) |
| 7 | Syria (+1.50) | Comoros (+1.81) | Saudi Arabia (+2.50) | Austria (+1.69) |
| 8 | Iraq (+1.50) | Somalia (+1.75) | Venezuela (+2.50) | Afghanistan (+1.62) |
| 9 | Afghanistan (+1.31) | Nicaragua (+1.62) | Slovakia (+2.38) | Spain (+1.56) |
| 10 | El Salvador (+1.25) | Kuwait (+1.62) | Afghanistan (+2.31) | Poland (+1.56) |

## college_hate_vs_love
_neg_: `In {year} college students hate ` | _pos_: `In {year} college students love `

| Rank | 1970 | 2000 | 2020 | 2026 |
|------|------|------|------|------|
| 1 | Cuba (+2.12) | North Korea (+0.94) | Syria (+2.25) | North Korea (+1.38) |
| 2 | North Korea (+1.19) | South Africa (+0.69) | South Africa (+1.31) | South Africa (+0.75) |
| 3 | Colombia (+0.38) | Greece (+0.19) | North Korea (+1.25) | Canada (+0.44) |
| 4 | United States (+0.38) | Iraq (-0.06) | Iraq (+0.81) | Iraq (+0.44) |
| 5 | Ukraine (+0.38) | Afghanistan (-0.06) | Afghanistan (+0.50) | Japan (+0.38) |
| 6 | Spain (+0.31) | Georgia (-0.38) | Iran (+0.38) | United States (+0.31) |
| 7 | Nigeria (+0.25) | Mexico (-0.38) | Egypt (+0.19) | Finland (+0.25) |
| 8 | Mexico (+0.25) | Chile (-0.50) | Pakistan (+0.19) | Afghanistan (+0.25) |
| 9 | Peru (+0.19) | Canada (-0.62) | Saudi Arabia (+0.12) | South Korea (+0.06) |
| 10 | Argentina (+0.19) | Egypt (-0.75) | Croatia (+0.00) | Georgia (+0.06) |
