# Predictive analytics: asylum decision outcomes

A count-weighted **multinomial logistic regression** using UNHCR's aggregate asylum-decision statistics. Supply a country of origin, a country of asylum and a decision year. The model returns probabilities for:

1. Grant of refugee status.
2. Complementary or other protection.
3. Refusal.
4. Administrative closure (UNHCR's “Otherwise closed”).

The four probabilities sum to one. The output also includes **grant or other protection**, and the probability of protection **conditional on a substantive decision**. Closures remain in the main four-outcome model.

## Run a prediction

Python 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python asylum_model.py predict --origin Syria --asylum UK --year 2026
```

On macOS/Linux use `.venv/bin/python`. Country names, ISO alpha-3 codes, and a few common aliases are supported. Predictions run offline from the included JSON model; no API key or notebook execution is needed.

```sh
python asylum_model.py predict --origin AFG --asylum DEU --year 2026
python asylum_model.py predict --origin Afghanistan --asylum Germany --year 2025 --unit P
```

`--unit auto` prefers the person model whenever both countries have coverage, falling back to cases otherwise. Override with `--unit P` (persons) or `--unit C` (cases). The output identifies the selected basis, training years, pair sample size and coverage warnings. A case can contain multiple people: the two units are **never summed, converted or fitted together**. The case model is experimental: it underperformed the historical country-pair baseline on the held-out year, and predictions carry that warning.

Historical-year output is a fitted estimate using the production model, which includes those years. Future-year output extrapolates the fitted trends; this version accepts at most five years beyond the snapshot and labels such output as extrapolation. It cannot anticipate changes in legislation, conflict or decision practice.

## Meaning of a probability

The main estimate is the predicted distribution of **recorded decision outcomes** for a country pair and year, using the mix of authorities and procedural stages present in UNHCR's data. It is not a probability of eventual success for an individual applicant. An applicant can receive multiple decisions, including on appeal, and the data do not link those decisions into individual histories. The input year is the **decision year**, not the year a claim was submitted.

For example, the combined protection probability is:

```text
P(protection) = P(grant) + P(other protection)
P(protection | substantive decision) = P(protection) / (1 - P(closure))
```

This model pools reported procedural stages and authorities because those are not inputs. It is not specifically a first-instance model. UNHCR notes that appeal reporting is incomplete, so composition and coverage differences can affect comparisons. Country effects describe historical associations rather than causal effects.

## Data and reproducibility

- `data/asylum-decisions.csv.gz`: losslessly compressed 2000–2025 snapshot copied from the existing `asylum-tracker` project. It was not refreshed during model development.
- `data/provenance.json`: source URL, snapshot limitations and raw/compressed SHA-256 hashes. The original download date is unknown.
- `artifacts/model.json`: portable coefficients, feature names, reporting-unit models and training coverage; no pickle deserialization.
- `artifacts/evaluation.json`: temporal validation, test metrics, baselines and class-by-class calibration bins.

Outcomes are grouped by origin, asylum country, decision year and reporting unit. Each component count must be a finite nonnegative integer. Exact duplicate source rows cause training to fail. Missing, non-country and historical ISO codes absent from the installed ISO country registry are excluded, as are zero-count groups. These exclusions are counted in the evaluation audit. Coverage is therefore narrower than the entire source file; Stateless and Various are not treated as countries.

The denominator is the **sum of the four outcome counts**, rather than the supplied `Total decisions` field. Any rows where these differ are counted in the audit. This gives a coherent outcome distribution without silently adjusting a category to force a match. All stage and authority rows are retained and summed within each reporting unit; the source does not establish that these are unique applicants. Published values, including any source rounding or disclosure treatment, are used as supplied.

Reproduce training and checks:

```sh
python asylum_model.py train
python -m unittest discover -s tests -v
```

To use a newer export with the same CSV schema:

```sh
python asylum_model.py train --data path/to/asylum-decisions.csv --output artifacts
```

Update the provenance file when replacing the bundled source snapshot. Training does not fetch or silently refresh data.

## Method and evaluation

The multinomial model includes regularized effects for origin, asylum country and their pair, a linear year trend, and origin-specific and asylum-specific year trends. Year is centered on 2010 and scaled in decades. Unseen country pairs can use learned country effects and trends; a country with no coverage in the selected model is rejected rather than assigned an arbitrary default.

The likelihood uses the four observed outcome counts as weights. At most four training rows represent each group, avoiding expansion to one row per recorded person or case. Weights are rescaled to sum to the number of groups so regularization has a defined scale. There is no class balancing: reweighting classes would change the probability target.

For each reporting unit separately:

1. Select regularization `C` from 0.1, 1 and 10 on the penultimate available year, training on earlier years. For persons this means training through 2023 and validating on **2024**. Eligible case-based data has no 2024 observations, so its validation year is **2023**.
2. Refit the selected model on years before the test year and evaluate on **2025**, without selecting parameters using that test year. Each unit's exact splits are recorded in the evaluation JSON.
3. Compare with training-only global frequencies and country-pair frequencies smoothed with 20 decisions of the global prior.
4. Report decision-weighted log loss, multiclass Brier score, calibration bins and performance on seen/unseen pairs.
5. Refit production coefficients on all available years, through 2025. Production model performance is not a new held-out evaluation.

For a replacement snapshot, these splits shift to each reporting unit's last two available years. Lower log loss and Brier scores are better. An individual-level multiclass Brier score is computed from the category counts, not from squared error against fractional group labels. This is an initial temporal holdout, not proof of stable calibration in every country or future year. Confidence intervals are not provided; recorded decision counts do not measure model or policy uncertainty.

## Initial held-out results

2025 outcomes, with models fitted only on earlier years. Lower scores are better:

| Reporting unit | Method | Log loss | Multiclass Brier |
| --- | --- | ---: | ---: |
| Persons | Multinomial regression | 0.9467 | 0.5395 |
| Persons | Historical country-pair baseline | 1.0707 | 0.5589 |
| Persons | Global frequency baseline | 1.2632 | 0.6974 |
| Cases | Multinomial regression | 1.3232 | 0.7958 |
| Cases | Historical country-pair baseline | 0.9060 | 0.5283 |
| Cases | Global frequency baseline | 0.9991 | 0.5956 |

The person-based regression reduces log loss by approximately **11.6%** relative to the historical country-pair baseline over 2,375,264 recorded decisions. Case-based performance over 46,738 recorded decisions is worse than both baselines, so those estimates are experimental. This comparison covers one test year; it is not a claim of general forecasting superiority.

Example **2026 extrapolation**, Syria → United Kingdom, using person-based data:

| Outcome | Predicted probability |
| --- | ---: |
| Grant | 73.29% |
| Other protection | 8.89% |
| Refusal | 3.18% |
| Administrative closure | 14.63% |
| Grant or other protection combined | 82.19% |

The first four sum to 100% before display rounding. These are model outputs, not observed 2026 rates or individual claim probabilities.

## Sources

- [UNHCR asylum-decision metadata](https://refugeedata.unhcr.org/en/data-finder/detail/AsylumDecisions/metadata/) — outcome definitions and reporting limitations.
- [UNHCR Refugee Statistics API](https://api.unhcr.org/docs/refugee-statistics.html) — source dataset and API.
- [Scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — multinomial fitting and sample weights.

UNHCR is the data provider and does not endorse this model. Source statistics remain subject to [UNHCR's terms of use](https://www.unhcr.org/refugee-statistics/).
