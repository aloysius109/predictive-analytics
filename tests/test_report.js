/* Runs without dependencies in Node, or from report-browser.html in a browser. */
(function () {
  'use strict';
  function check(model, fixtures, Predictor) {
    const engine = new Predictor(model);
    let checks = 0;
    function assert(condition, message) { if (!condition) throw new Error(message); checks++; }
    for (const expected of fixtures) {
      const unit = expected.reporting_unit === 'persons' ? 'P' : 'C';
      const actual = engine.predict(expected.origin, expected.asylum, expected.year, unit);
      assert(actual.reporting_unit === expected.reporting_unit, 'Reporting unit mismatch');
      const values = Object.values(actual.probabilities);
      assert(values.every(p => Number.isFinite(p) && p >= 0 && p <= 1), 'Invalid probability');
      assert(Math.abs(values.reduce((a, b) => a+b, 0) - 1) < 1e-12, 'Outcomes do not sum to one');
      for (const key of Object.keys(expected.probabilities)) {
        assert(Math.abs(actual.probabilities[key] - expected.probabilities[key]) < 1e-12, `Python/JavaScript mismatch for ${key}`);
      }
      for (const key of ['probability_grant_or_other_protection', 'probability_protection_given_substantive_decision']) {
        assert(Math.abs(actual[key] - expected[key]) < 1e-12, `Combined-probability mismatch: ${key}`);
      }
    }
    for (const [origin, asylum, year] of [['SYR','GBR',2040], ['SYR','GBR',2026.5], ['SYR','GBR',1999], ['ZZZ','GBR',2026]]) {
      let rejected = false;
      try { engine.predict(origin, asylum, year); } catch { rejected = true; }
      assert(rejected, 'Invalid input was accepted');
    }
    assert(engine.predict('SYR','GBR',2026).reporting_unit === 'persons', 'Auto unit does not prefer persons');
    return `${checks} checks passed across ${fixtures.length} Python/JavaScript prediction fixtures.`;
  }
  if (typeof require === 'function') {
    const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
    const root = path.resolve(__dirname, '..');
    const model = JSON.parse(fs.readFileSync(path.join(root,'docs/model.json'),'utf8'));
    const original = JSON.parse(fs.readFileSync(path.join(root,'artifacts/model.json'),'utf8'));
    require('node:assert/strict').deepEqual(model, original, 'Report model differs from production artifact');
    const fixtures = JSON.parse(fs.readFileSync(path.join(root,'.report-qa/fixtures.json'),'utf8'));
    const context = {};
    vm.runInNewContext(fs.readFileSync(path.join(root,'docs/predictor.js'),'utf8'), context);
    console.log(check(model, fixtures, context.AsylumPredictor));
  } else {
    Promise.all([fetch('../docs/model.json').then(r=>r.json()), fetch('../.report-qa/fixtures.json').then(r=>r.json())])
      .then(([model, fixtures]) => { document.body.dataset.result='passed'; document.getElementById('result').textContent=check(model, fixtures, AsylumPredictor); })
      .catch(error => { document.body.dataset.result='failed'; document.getElementById('result').textContent=error.stack; });
  }
})();
