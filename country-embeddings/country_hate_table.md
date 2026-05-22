# 'In <year> we hate <country>' vs 'we love <country>' - Qwen3-4B-Base

Model: `Qwen/Qwen3-4B-Base`  (base / not instruction-tuned)

**Metric**: `score = log P(c | 'In <year> we hate ') - log P(c | 'In <year> we love ')`. Country tokens are identical in both terms, so name-length and tokenizer-frequency effects cancel. Positive = model finds the country more likely after 'we hate'; negative = more likely after 'we love'.

**Caveat (important)**: This shows what associations the training corpus encoded; it is NOT a measurement of truth, of public sentiment, or of any moral fact about any country.

## Top-10 hate-biased per year

| Rank | 1950 | 1970 | 1990 | 2000 | 2010 | 2020 | 2024 | 2026 |
|------|------|------|------|------|------|------|------|------|
| 1 | Svalbard and Jan Mayen (+2.88) | Kiribati (+3.00) | Svalbard and Jan Mayen (+2.88) | Svalbard and Jan Mayen (+3.00) | Congo, The Democratic Republic of the (+3.00) | Svalbard and Jan Mayen (+4.62) | Svalbard and Jan Mayen (+3.88) | Iraq (+2.69) |
| 2 | Congo, The Democratic Republic of the (+2.62) | Svalbard and Jan Mayen (+2.75) | Kiribati (+2.00) | Congo, The Democratic Republic of the (+2.25) | Svalbard and Jan Mayen (+3.00) | North Korea (+2.38) | Micronesia, Federated States of (+3.00) | Iran (+2.56) |
| 3 | Somalia (+2.12) | North Korea (+2.38) | Djibouti (+1.50) | North Korea (+1.94) | Guyana (+2.50) | Tuvalu (+2.25) | Mayotte (+2.38) | North Korea (+2.06) |
| 4 | Central African Republic (+2.12) | Congo, The Democratic Republic of the (+2.12) | Congo, The Democratic Republic of the (+1.25) | Kiribati (+1.50) | Djibouti (+2.50) | Nauru (+2.12) | North Korea (+2.31) | Svalbard and Jan Mayen (+1.88) |
| 5 | Kiribati (+2.00) | Djibouti (+2.12) | Central African Republic (+1.25) | Moldova (+1.38) | Iran (+2.44) | China (+2.06) | Russian Federation (+2.25) | Djibouti (+1.75) |
| 6 | Kazakhstan (+2.00) | Kazakhstan (+2.00) | Kazakhstan (+1.19) | Iraq (+1.25) | Libya (+2.38) | Micronesia, Federated States of (+2.00) | Iraq (+2.00) | Moldova (+1.62) |
| 7 | Algeria (+1.88) | Central African Republic (+1.88) | Algeria (+1.12) | Guyana (+1.25) | Kiribati (+2.38) | Western Sahara (+2.00) | Moldova (+1.88) | Israel (+1.56) |
| 8 | Bangladesh (+1.75) | Moldova (+1.88) | South Sudan (+1.00) | Central African Republic (+1.25) | Iraq (+2.31) | Russian Federation (+2.00) | Nauru (+1.88) | Kuwait (+1.56) |
| 9 | North Korea (+1.69) | Iraq (+1.81) | Canada (+0.94) | British Indian Ocean Territory (+1.12) | Russian Federation (+2.25) | Iraq (+1.75) | China (+1.81) | Mayotte (+1.50) |
| 10 | Laos (+1.62) | Libya (+1.62) | North Korea (+0.94) | Algeria (+1.12) | Somalia (+2.25) | Bonaire, Sint Eustatius and Saba (+1.75) | Kiribati (+1.62) | Kyrgyzstan (+1.50) |

## Top-10 love-biased per year (lowest score = most love-biased)

| Rank | 1950 | 1970 | 1990 | 2000 | 2010 | 2020 | 2024 | 2026 |
|------|------|------|------|------|------|------|------|------|
| 1 | Turks and Caicos Islands (-6.00) | Turks and Caicos Islands (-5.88) | Turks and Caicos Islands (-5.62) | Turks and Caicos Islands (-5.50) | Turks and Caicos Islands (-5.50) | Cocos (Keeling) Islands (-6.88) | Cocos (Keeling) Islands (-8.38) | Cocos (Keeling) Islands (-8.75) |
| 2 | Åland Islands (-4.12) | Åland Islands (-3.50) | Åland Islands (-5.00) | Åland Islands (-3.75) | Cocos (Keeling) Islands (-2.88) | Turks and Caicos Islands (-6.50) | Turks and Caicos Islands (-8.00) | Turks and Caicos Islands (-6.38) |
| 3 | Timor-Leste (-3.38) | United States Minor Outlying Islands (-2.88) | United States Minor Outlying Islands (-3.38) | Cocos (Keeling) Islands (-2.38) | French Southern Territories (-1.88) | Heard Island and McDonald Islands (-2.88) | French Southern Territories (-2.75) | Holy See (Vatican City State) (-2.50) |
| 4 | Martinique (-2.88) | Christmas Island (-2.25) | Martinique (-3.25) | Liechtenstein (-2.25) | Christmas Island (-1.75) | French Southern Territories (-2.88) | Heard Island and McDonald Islands (-2.38) | Saint Barthélemy (-2.00) |
| 5 | Virgin Islands, British (-2.88) | Saint Helena, Ascension and Tristan da Cunha (-2.25) | Cayman Islands (-2.88) | United States Minor Outlying Islands (-2.00) | Åland Islands (-1.50) | Åland Islands (-2.88) | Andorra (-2.25) | Saint Helena, Ascension and Tristan da Cunha (-2.00) |
| 6 | Andorra (-2.75) | Martinique (-2.00) | French Southern Territories (-2.62) | Macao (-1.88) | North Macedonia (-1.38) | United States Minor Outlying Islands (-1.75) | South Georgia and the South Sandwich Islands (-1.88) | Heard Island and McDonald Islands (-1.88) |
| 7 | Christmas Island (-2.50) | Cayman Islands (-1.88) | Liechtenstein (-2.62) | Christmas Island (-1.88) | Virgin Islands, British (-1.25) | Madagascar (-1.62) | British Indian Ocean Territory (-1.62) | Andorra (-1.88) |
| 8 | French Southern Territories (-2.38) | Bouvet Island (-1.88) | Wallis and Futuna (-2.62) | Cayman Islands (-1.88) | Liechtenstein (-1.25) | Saint Helena, Ascension and Tristan da Cunha (-1.50) | Falkland Islands (Malvinas) (-1.62) | Ghana (-1.75) |
| 9 | Cayman Islands (-2.25) | Isle of Man (-1.75) | Palau (-2.50) | Solomon Islands (-1.88) | United States Minor Outlying Islands (-1.12) | Monaco (-1.50) | Saint Barthélemy (-1.62) | Samoa (-1.75) |
| 10 | Sint Maarten (Dutch part) (-2.12) | Andorra (-1.75) | Montserrat (-2.38) | Virgin Islands, British (-1.75) | South Georgia and the South Sandwich Islands (-1.00) | Papua New Guinea (-1.38) | Ghana (-1.56) | French Southern Territories (-1.75) |