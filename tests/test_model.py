import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from asylum_model import COUNTS, fit, load_data, metrics, predict, probabilities, train


class ModelTests(unittest.TestCase):
    def source(self, path):
        rows = []
        for year in range(2020, 2026):
            for unit in ["P", "C"]:
                rows.append({"Year": year, "Country of origin (ISO)": "SYR", "Country of asylum (ISO)": "GBR",
                             "Country of origin": "Syrian Arab Rep.", "Country of asylum": "United Kingdom",
                             "Cases / Persons": unit, "Recognized decisions": 60, "Complementary protection": 20,
                             "Rejected decisions": 15, "Otherwise closed": 5, "Total decisions": 100})
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_units_stay_separate_and_closures_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            self.source(path)
            data, _, audit = load_data(path)
            self.assertEqual(len(data), 12)
            self.assertEqual(set(data.unit), {"P", "C"})
            self.assertEqual(data["Otherwise closed"].sum(), 60)
            self.assertEqual(audit["rows_with_reported_total_mismatch"], 0)

    def test_invalid_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            self.source(path)
            data = pd.read_csv(path)
            data.loc[0, "Recognized decisions"] = -1
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                load_data(path)

    def test_duplicate_source_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            self.source(path)
            data = pd.read_csv(path)
            pd.concat([data, data.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_data(path)

    def test_metrics_score_decisions_not_fractional_labels(self):
        counts = np.array([[60, 20, 15, 5]])
        p = counts / 100
        scored = metrics(counts, p)
        self.assertAlmostEqual(scored["log_loss"], -np.sum(p * np.log(p)))
        self.assertAlmostEqual(scored["multiclass_brier"], 1 - np.sum(p ** 2))

    def test_temporal_splits_respect_gaps_in_reporting_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            self.source(path)
            data = pd.read_csv(path)
            data = data.loc[~(data["Cases / Persons"].eq("C") & data.Year.eq(2024))]
            data.to_csv(path, index=False)
            output = Path(folder) / "output"
            train(path, output)
            report = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(report["units"]["P"]["validation_year"], 2024)
            self.assertEqual(report["units"]["C"]["validation_year"], 2023)
            self.assertEqual(report["units"]["C"]["train_years"], [2020, 2023])
            self.assertEqual(report["units"]["C"]["test_year"], 2025)
            self.assertEqual(report["units"]["C"]["test"]["decisions"], 100)

    def test_exported_predictions_match_fitted_model_and_validate_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            self.source(path)
            data, names, _ = load_data(path)
            data = data.loc[data.unit.eq("P")]
            vectorizer, model = fit(data, 1)
            saved = {"names": names, "models": {"P": {
                "features": vectorizer.get_feature_names_out().tolist(), "coefficients": model.coef_.tolist(),
                "intercepts": model.intercept_.tolist(), "first_year": 2020, "last_year": 2025,
                "test_log_loss": 1.3, "route_baseline_test_log_loss": 0.9,
                "origins": ["SYR"], "asylum_countries": ["GBR"],
                "routes": {"SYR:GBR": {"decisions": 600, "first_year": 2020, "last_year": 2025}}}}}
            artifact = Path(folder) / "model.json"
            artifact.write_text(json.dumps(saved), encoding="utf-8")
            result = predict(artifact, "Syria", "UK", 2025)
            actual = list(result["probabilities"].values())
            np.testing.assert_allclose(actual, probabilities((vectorizer, model), data.iloc[-1:])[0], atol=1e-12)
            np.testing.assert_allclose(actual, [0.60, 0.20, 0.15, 0.05], atol=1e-3)
            self.assertAlmostEqual(sum(actual), 1)
            self.assertAlmostEqual(result["probability_grant_or_other_protection"], actual[0] + actual[1])
            self.assertGreater(result["probability_protection_given_substantive_decision"], actual[0] + actual[1])
            self.assertTrue(any("underperformed" in note for note in result["notes"]))
            for origin, year in [("Sryia", 2026), ("Syria", 2040), ("Syria", 2025.5)]:
                with self.assertRaises(ValueError):
                    predict(artifact, origin, "UK", year)


if __name__ == "__main__":
    unittest.main()
