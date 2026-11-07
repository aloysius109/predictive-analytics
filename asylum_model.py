"""Count-weighted multinomial regression for aggregate UNHCR asylum outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
OUTCOMES = ["grant", "other_protection", "refusal", "administrative_closure"]
COUNTS = ["Recognized decisions", "Complementary protection", "Rejected decisions", "Otherwise closed"]
KEYS = ["origin", "asylum", "year", "unit"]
RENAME = {"Country of origin (ISO)": "origin", "Country of asylum (ISO)": "asylum",
          "Year": "year", "Cases / Persons": "unit"}


def load_data(path: Path):
    raw = pd.read_csv(path, keep_default_na=False)
    required = list(RENAME) + COUNTS + ["Country of origin", "Country of asylum"]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = raw.rename(columns=RENAME).copy()
    for column in ["year", *COUNTS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative integers")
    if not frame["unit"].isin(["P", "C"]).all():
        raise ValueError("Unknown reporting unit; expected P (persons) or C (cases)")
    # Country labels such as Stateless and Various are not countries for this model.
    import pycountry
    iso_codes = {c.alpha_3 for c in pycountry.countries}
    country_rows = frame["origin"].isin(iso_codes) & frame["asylum"].isin(iso_codes)
    audit = {"input_rows": len(frame), "excluded_non_country_rows": int((~country_rows).sum()),
             "exact_duplicate_rows": int(raw.duplicated().sum())}
    if audit["exact_duplicate_rows"]:
        raise ValueError("Exact duplicate source rows; investigate before training")
    frame = frame.loc[country_rows]
    names = {}
    for code_col, name_col in [("origin", "Country of origin"), ("asylum", "Country of asylum")]:
        names.update(dict(zip(frame[code_col], frame[name_col])))
    if "Total decisions" in frame:
        total = pd.to_numeric(frame["Total decisions"], errors="raise")
        audit["rows_with_reported_total_mismatch"] = int((frame[COUNTS].sum(axis=1) != total).sum())
    grouped = frame.groupby(KEYS, as_index=False)[COUNTS].sum()
    zero = grouped[COUNTS].sum(axis=1).eq(0)
    audit["excluded_zero_outcome_groups"] = int(zero.sum())
    grouped = grouped.loc[~zero].reset_index(drop=True)
    if grouped.empty:
        raise ValueError("No usable outcome counts")
    audit["grouped_rows"] = len(grouped)
    return grouped, names, audit


def features(origin: str, asylum: str, year: int):
    decade = (year - 2010) / 10.0
    return {f"origin={origin}": 1.0, f"asylum={asylum}": 1.0,
            f"route={origin}:{asylum}": 1.0, "year_decade": decade,
            f"origin_trend={origin}": decade, f"asylum_trend={asylum}": decade}


def feature_rows(frame):
    return [features(r.origin, r.asylum, int(r.year)) for r in frame.itertuples()]


def fit(frame, regularization):
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(feature_rows(frame))
    counts = frame[COUNTS].to_numpy(dtype=float)
    # At most four rows per group, not one row per person or case.
    rows, labels = np.nonzero(counts > 0)
    weights = counts[rows, labels]
    weights = weights / weights.sum() * len(frame)
    model = LogisticRegression(C=regularization, solver="newton-cg", max_iter=200, tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x[rows], labels, sample_weight=weights)
    if list(model.classes_) != [0, 1, 2, 3]:
        raise ValueError("Training data must contain all four outcomes")
    return vectorizer, model


def probabilities(fitted, frame):
    vectorizer, model = fitted
    return model.predict_proba(vectorizer.transform(feature_rows(frame)))


def metrics(counts, probabilities):
    counts = np.asarray(counts, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    n = counts.sum()
    if n <= 0 or p.shape != counts.shape:
        raise ValueError("Invalid evaluation counts or probability dimensions")
    totals = counts.sum(axis=1)
    log_loss = -np.sum(counts * np.log(np.clip(p, 1e-15, 1))) / n
    brier = (np.sum(totals * np.sum(p ** 2, axis=1)) - 2 * np.sum(counts * p) + n) / n
    calibration = {}
    for k, outcome in enumerate(OUTCOMES):
        bins = []
        for index in range(10):
            mask = np.minimum((p[:, k] * 10).astype(int), 9) == index
            weight = totals[mask].sum()
            if weight:
                bins.append({"bin": index, "decisions": int(weight),
                             "predicted": float(np.average(p[mask, k], weights=totals[mask])),
                             "observed": float(counts[mask, k].sum() / weight)})
        calibration[outcome] = bins
    return {"decisions": int(n), "log_loss": float(log_loss), "multiclass_brier": float(brier),
            "observed_shares": dict(zip(OUTCOMES, (counts.sum(axis=0) / n).tolist())),
            "predicted_shares": dict(zip(OUTCOMES, np.average(p, axis=0, weights=totals).tolist())),
            "calibration": calibration}


def baselines(train, target):
    prior = train[COUNTS].sum().to_numpy(dtype=float)
    prior /= prior.sum()
    grouped = train.groupby(["origin", "asylum"])[COUNTS].sum()
    route_predictions = []
    for row in target.itertuples():
        key = (row.origin, row.asylum)
        values = grouped.loc[key].to_numpy(dtype=float) if key in grouped.index else np.zeros(4)
        route_predictions.append((values + 20 * prior) / (values.sum() + 20))
    return {"global_frequency": np.tile(prior, (len(target), 1)),
            "smoothed_route_frequency": np.array(route_predictions)}


def train(data_path: Path, output: Path):
    data, names, audit = load_data(data_path)
    last = int(data.year.max())
    if data.year.nunique() < 4:
        raise ValueError("At least four reporting years are required for temporal validation")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "outcomes": OUTCOMES, "names": names,
                "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "trained_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    report = {"source_sha256": manifest["source_sha256"], "data_audit": audit,
              "snapshot_years": [int(data.year.min()), last],
              "test_protocol": "C selected on validation year only; test fitted on years before test; production refitted through test year.",
              "units": {}}
    for unit in ["P", "C"]:
        subset = data.loc[data.unit.eq(unit)]
        years = sorted(subset.year.unique())
        if len(years) < 4:
            raise ValueError(f"At least four reporting years required for unit {unit}")
        last, validation_year = int(years[-1]), int(years[-2])
        development = subset.loc[subset.year.lt(validation_year)]
        validation = subset.loc[subset.year.eq(validation_year)]
        test = subset.loc[subset.year.eq(last)]
        if development.empty or validation.empty or test.empty:
            raise ValueError(f"Missing temporal split for reporting unit {unit}")
        choices = []
        for c in [0.1, 1.0, 10.0]:
            print(f"Fitting unit={unit}, C={c}, validation={validation_year}", flush=True)
            fitted = fit(development, c)
            score = metrics(validation[COUNTS], probabilities(fitted, validation))["log_loss"]
            choices.append({"C": c, "validation_log_loss": score})
        selected = min(choices, key=lambda item: item["validation_log_loss"])["C"]
        historical = subset.loc[subset.year.lt(last)]
        print(f"Evaluating unit={unit}, C={selected}, unseen year={last}", flush=True)
        fitted = fit(historical, selected)
        evaluation = metrics(test[COUNTS], probabilities(fitted, test))
        baseline_metrics = {key: metrics(test[COUNTS], value) for key, value in baselines(historical, test).items()}
        historical_routes = set(zip(historical.origin, historical.asylum))
        unseen = np.array([(o, a) not in historical_routes for o, a in zip(test.origin, test.asylum)])
        seen_metrics = metrics(test.loc[~unseen, COUNTS], probabilities(fitted, test.loc[~unseen])) if (~unseen).any() else None
        unseen_metrics = metrics(test.loc[unseen, COUNTS], probabilities(fitted, test.loc[unseen])) if unseen.any() else None
        report["units"][unit] = {"candidates": choices, "selected_C": selected,
                                  "validation_year": validation_year, "test_year": last,
                                  "test": evaluation, "baselines": baseline_metrics,
                                  "test_seen_routes": seen_metrics, "test_unseen_routes": unseen_metrics,
                                  "train_years": [int(historical.year.min()), int(historical.year.max())]}
        print(f"Refitting production unit={unit} through {last}", flush=True)
        vectorizer, model = fit(subset, selected)
        routes = {}
        for (origin, asylum), rows in subset.groupby(["origin", "asylum"]):
            routes[f"{origin}:{asylum}"] = {"decisions": int(rows[COUNTS].to_numpy().sum()),
                                           "first_year": int(rows.year.min()), "last_year": int(rows.year.max())}
        manifest["models"][unit] = {
            "features": vectorizer.get_feature_names_out().tolist(),
            "coefficients": model.coef_.tolist(), "intercepts": model.intercept_.tolist(),
            "selected_C": selected, "first_year": int(subset.year.min()), "last_year": last,
            "origins": sorted(subset.origin.unique()), "asylum_countries": sorted(subset.asylum.unique()),
            "routes": routes,
            "test_log_loss": evaluation["log_loss"],
            "route_baseline_test_log_loss": baseline_metrics["smoothed_route_frequency"]["log_loss"],
        }
    # Prefer person-based data when the requested route is observed in both units.
    (output / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved model and evaluation to {output}", flush=True)


def resolve_country(value, names):
    value = value.strip()
    code = value.upper()
    if code in names:
        return code
    for iso, label in names.items():
        if value.casefold() == label.casefold():
            return iso
    import pycountry
    aliases = {"uk": "GBR", "united kingdom": "GBR", "us": "USA", "south korea": "KOR",
               "north korea": "PRK", "iran": "IRN", "syria": "SYR", "russia": "RUS"}
    code = aliases.get(value.casefold())
    if code is None:
        try:
            code = pycountry.countries.lookup(value).alpha_3
        except LookupError as error:
            raise ValueError(f"Unknown country: {value}. Use a country name or ISO alpha-3 code.") from error
    if code not in names:
        raise ValueError(f"No eligible UNHCR country records for {value}")
    return code


def predict(model_path, origin, asylum, year, unit="auto"):
    saved = json.loads(Path(model_path).read_text(encoding="utf-8"))
    origin = resolve_country(origin, saved["names"])
    asylum = resolve_country(asylum, saved["names"])
    if isinstance(year, bool) or int(year) != year:
        raise ValueError("Year must be an integer")
    year = int(year)
    candidates = [u for u, m in saved["models"].items() if origin in m["origins"] and asylum in m["asylum_countries"]]
    route = f"{origin}:{asylum}"
    if unit == "auto":
        if not candidates:
            raise ValueError("Origin and asylum country have no common reporting-unit model")
        # Prefer persons: case-based performance is weaker on this snapshot's holdout.
        unit = "P" if "P" in candidates else "C"
    if unit not in candidates:
        raise ValueError(f"No training coverage for these countries in unit {unit}")
    model = saved["models"][unit]
    if not model["first_year"] <= year <= model["last_year"] + 5:
        raise ValueError(f"Year must be between {model['first_year']} and {model['last_year']+5}; longer extrapolation is unsupported")
    encoded = features(origin, asylum, year)
    coefficients = np.asarray(model["coefficients"])
    x = np.array([encoded.get(name, 0.0) for name in model["features"]])
    logits = coefficients @ x + np.asarray(model["intercepts"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    notes = []
    if model.get("test_log_loss", 0) > model.get("route_baseline_test_log_loss", float("inf")):
        notes.append("Experimental: this reporting-unit regression underperformed the historical country-pair baseline on held-out log loss.")
    support = model["routes"].get(route, {"decisions": 0, "first_year": None, "last_year": None})
    if not support["decisions"]:
        notes.append("Unseen country pair: estimate uses country effects and trends, without a learned pair effect.")
    elif support["decisions"] < 100:
        notes.append("Sparse pair: fewer than 100 recorded decisions across training years.")
    if year > model["last_year"]:
        notes.append("Future-year extrapolation: policy and population changes are not modelled.")
    else:
        notes.append("Historical fitted estimate: production training includes this year; this is not a held-out forecast.")
    if support["last_year"] is not None and year > support["last_year"] + 1:
        notes.append("No recent observations for this pair in the selected reporting unit.")
    protection = float(p[0] + p[1])
    return {"origin": origin, "asylum": asylum, "year": year,
            "reporting_unit": "persons" if unit == "P" else "cases",
            "probabilities": dict(zip(OUTCOMES, p.tolist())),
            "probability_grant_or_other_protection": protection,
            "probability_protection_given_substantive_decision": protection / float(p[:3].sum()),
            "training_years": [model["first_year"], model["last_year"]], "pair_support": support,
            "interpretation": "Aggregate recorded-decision outcomes, across reported authorities and procedural stages; not an individual claim assessment.",
            "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=ROOT / "data/asylum-decisions.csv.gz")
    training.add_argument("--output", type=Path, default=ROOT / "artifacts")
    prediction = sub.add_parser("predict")
    prediction.add_argument("--origin", required=True)
    prediction.add_argument("--asylum", required=True)
    prediction.add_argument("--year", required=True, type=int)
    prediction.add_argument("--unit", choices=["auto", "P", "C"], default="auto")
    prediction.add_argument("--model", type=Path, default=ROOT / "artifacts/model.json")
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args.data, args.output)
        else:
            print(json.dumps(predict(args.model, args.origin, args.asylum, args.year, args.unit), indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
