"""Generate cross-language inference fixtures from the production Python model."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asylum_model import predict


def main():
    path = ROOT / 'artifacts/model.json'
    model = json.loads(path.read_text(encoding='utf-8'))
    fixtures = []
    for unit, fitted in model['models'].items():
        routes = sorted(fitted['routes'])
        selected = [routes[i] for i in [0, len(routes)//4, len(routes)//2, 3*len(routes)//4, -1]]
        for route in selected:
            origin, asylum = route.split(':')
            for year in [fitted['first_year'], fitted['last_year'], fitted['last_year']+1, fitted['last_year']+5]:
                fixtures.append(predict(path, origin, asylum, year, unit))
        unseen = next(((o, a) for o in fitted['origins'] for a in fitted['asylum_countries'] if f'{o}:{a}' not in fitted['routes']), None)
        if unseen:
            fixtures.append(predict(path, *unseen, fitted['last_year']+1, unit))
    fixtures.append(predict(path, 'Syria', 'UK', 2026))
    qa = ROOT / '.report-qa'
    qa.mkdir(exist_ok=True)
    (qa / 'fixtures.json').write_text(json.dumps(fixtures, allow_nan=False), encoding='utf-8')
    print(f'Generated {len(fixtures)} Python prediction fixtures')


if __name__ == '__main__':
    main()
