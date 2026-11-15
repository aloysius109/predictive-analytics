"""Build static figures for the standalone asylum-outcomes Medium article."""
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asylum_model import OUTCOMES, predict

MODEL_PATH = ROOT / "artifacts" / "model.json"
EVALUATION_PATH = ROOT / "artifacts" / "evaluation.json"
IMAGE_DIR = ROOT / "medium-edition" / "images"
ORIGIN = "Syria"
ASYLUM = "United Kingdom"
MEDIUM_IMAGES = (
    "outcome-probability-curves.png",
    "held-out-log-loss.png",
    "person-model-calibration.png",
)

OUTCOME_LABELS = {
    "grant": "Grant of refugee status",
    "other_protection": "Other protection",
    "refusal": "Refusal",
    "administrative_closure": "Administrative closure",
}
OUTCOME_COLORS = {
    "grant": "#247367",
    "other_protection": "#63a99b",
    "refusal": "#b95f4b",
    "administrative_closure": "#7b8490",
}
METHOD_LABELS = ("Multinomial model", "Country-pair baseline", "Global-frequency baseline")
METHOD_COLORS = ("#247367", "#d68d38", "#87949a")


def load_inputs():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if model.get("source_sha256") != evaluation.get("source_sha256"):
        raise ValueError("Model and evaluation artifacts refer to different source snapshots")
    if "P" not in model.get("models", {}) or "P" not in evaluation.get("units", {}):
        raise ValueError("Person-based model and evaluation results are required for the Medium figures")
    return model, evaluation


def save_probability_curves(model):
    fitted = model["models"]["P"]
    years = list(range(fitted["first_year"], fitted["last_year"] + 6))
    predictions = [predict(MODEL_PATH, ORIGIN, ASYLUM, year, unit="P")["probabilities"] for year in years]

    fig, ax = plt.subplots(figsize=(10.2, 5.8), layout="constrained")
    ax.axvspan(fitted["last_year"] + .5, years[-1] + .5, color="#f5eadb", zorder=0)
    for outcome in OUTCOMES:
        ax.plot(
            years,
            [prediction[outcome] for prediction in predictions],
            color=OUTCOME_COLORS[outcome],
            linewidth=2.6,
            label=OUTCOME_LABELS[outcome],
        )
    ax.axvline(fitted["last_year"] + .5, color="#7b8490", linewidth=1.1, linestyle="--")
    ax.text(fitted["last_year"] + .8, .97, "Extrapolation", color="#59675f", fontsize=9, va="top")
    ax.set_title(f"Illustrative modelled outcomes: {ORIGIN} to {ASYLUM}", loc="left", fontsize=13, pad=12)
    ax.set_xlabel("Decision year")
    ax.set_ylabel("Predicted share of recorded decisions")
    ax.set_ylim(0, 1)
    ax.set_xlim(years[0], years[-1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e2e9e4", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d1d9d5")
    ax.tick_params(axis="both", colors="#65767a", labelsize=9, length=0, pad=7)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    fig.savefig(IMAGE_DIR / "outcome-probability-curves.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_holdout_comparison(evaluation):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, layout="constrained")
    for ax, unit, label in zip(axes, ("P", "C"), ("Persons", "Cases")):
        result = evaluation["units"][unit]
        scores = [
            result["test"]["log_loss"],
            result["baselines"]["smoothed_route_frequency"]["log_loss"],
            result["baselines"]["global_frequency"]["log_loss"],
        ]
        bars = ax.barh(METHOD_LABELS, scores, color=METHOD_COLORS, height=.62)
        ax.invert_yaxis()
        ax.set_title(f"{label} · 2025 holdout", loc="left", fontsize=12)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("Decision-weighted log loss · lower is better", fontsize=8)
        ax.grid(axis="x", color="#e2e9e4", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d1d9d5")
        ax.tick_params(axis="both", colors="#59675f", labelsize=8, length=0, pad=6)
        for bar, score in zip(bars, scores):
            ax.text(score + .025, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=8, color="#26352e")
    fig.suptitle("Temporal holdout: the result differs by reporting unit", x=.02, ha="left", fontsize=14)
    fig.savefig(IMAGE_DIR / "held-out-log-loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_calibration(evaluation):
    calibration = evaluation["units"]["P"]["test"]["calibration"]
    fig, ax = plt.subplots(figsize=(7.1, 6.1), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#87949a", linewidth=1.2, linestyle="--", label="Perfect calibration")
    for outcome in OUTCOMES:
        bins = calibration[outcome]
        if not bins:
            continue
        maximum = max(item["decisions"] for item in bins)
        sizes = [22 + 80 * np.sqrt(item["decisions"] / maximum) for item in bins]
        ax.scatter(
            [item["predicted"] for item in bins],
            [item["observed"] for item in bins],
            s=sizes,
            color=OUTCOME_COLORS[outcome],
            alpha=.78,
            edgecolor="white",
            linewidth=.7,
            label=OUTCOME_LABELS[outcome],
        )
    ax.set_title("Person-model calibration on the 2025 holdout", loc="left", fontsize=13, pad=12)
    ax.set_xlabel("Mean predicted probability in bin")
    ax.set_ylabel("Observed share of recorded decisions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e2e9e4", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d1d9d5")
    ax.spines["bottom"].set_color("#d1d9d5")
    ax.tick_params(axis="both", colors="#65767a", labelsize=9, length=0, pad=7)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(IMAGE_DIR / "person-model-calibration.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def check_package():
    article_path = ROOT / "medium-edition" / "article.html"
    notes_path = ROOT / "medium-edition" / "PUBLISHING.txt"
    if not article_path.is_file() or not notes_path.is_file():
        raise ValueError("The Medium article or publishing notes are missing")
    actual_images = {path.name for path in IMAGE_DIR.glob("*.png")}
    if actual_images != set(MEDIUM_IMAGES):
        raise ValueError(f"Medium images differ from the expected set: {sorted(actual_images)}")

    article = article_path.read_text(encoding="utf-8")
    canonical = 'rel="canonical" href="https://aloysius109.github.io/predictive-analytics/"'
    if canonical not in article:
        raise ValueError("The Medium article is missing its canonical link to the live report")
    if re.search(r"<script\b|class=[\"'][^\"']*(?:quarto-|cell-output|cell-input)", article, re.I):
        raise ValueError("The Medium article must not contain scripts or Quarto-specific classes")
    image_tags = re.findall(r"<img\b[^>]*>", article, re.I | re.S)
    if len(image_tags) != len(MEDIUM_IMAGES):
        raise ValueError("The Medium article image count does not match the generated chart list")
    sources = set()
    for tag in image_tags:
        source = re.search(r"\bsrc=[\"']([^\"']+)[\"']", tag, re.I)
        alt = re.search(r"\balt=[\"']([^\"']*)[\"']", tag, re.I)
        if source is None or alt is None or not alt.group(1).strip() or not source.group(1).startswith("images/"):
            raise ValueError("Every Medium chart needs a local image source and non-empty alt text")
        sources.add(Path(source.group(1)).name)
    if sources != set(MEDIUM_IMAGES):
        raise ValueError("The Medium article references do not match the generated chart list")


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    model, evaluation = load_inputs()
    save_probability_curves(model)
    save_holdout_comparison(evaluation)
    save_calibration(evaluation)
    check_package()
    print(f"Built three Medium figures from the committed model and evaluation artifacts in {IMAGE_DIR}.")


if __name__ == "__main__":
    main()