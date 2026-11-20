"""Copy the exact model to the static report, without fitting or refreshing data."""
import json
from pathlib import Path
from publish_summary import publish_summary_html

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/model.json'
    model = json.loads(source.read_text(encoding='utf-8'))
    if model['schema_version'] != 1:
        raise ValueError('Unsupported model version')
    for unit, fitted in model['models'].items():
        if len(fitted['coefficients']) != 4 or len(fitted['intercepts']) != 4:
            raise ValueError(f'Expected four outcome classes for unit {unit}')
        if any(len(row) != len(fitted['features']) for row in fitted['coefficients']):
            raise ValueError(f'Coefficient dimensions do not match for unit {unit}')
    target = ROOT / 'docs/model.json'
    target.write_text(json.dumps(model, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    (ROOT / 'docs/.nojekyll').touch()
    article = ROOT / 'medium-edition/article.html'
    publish_summary_html(
        article.read_text(encoding='utf-8'),
        ROOT / 'docs',
        'https://aloysius109.github.io/predictive-analytics/',
        ROOT / 'medium-edition/images',
    )
    print(f'Staged model for the report: {target.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
