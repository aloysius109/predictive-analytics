/* Portable inference for the committed multinomial logistic regression. */
(function (global) {
  'use strict';
  const outcomes = ['grant', 'other_protection', 'refusal', 'administrative_closure'];
  class AsylumPredictor {
    constructor(data) {
      if (data.schema_version !== 1 || !data.models || !data.names) throw new Error('Unsupported model file.');
      this.data = data;
      this.indices = {};
      for (const [unit, model] of Object.entries(data.models)) {
        this.indices[unit] = new Map(model.features.map((name, i) => [name, i]));
      }
    }
    candidates(origin, asylum) {
      return Object.keys(this.data.models).filter(unit => {
        const m = this.data.models[unit];
        return m.origins.includes(origin) && m.asylum_countries.includes(asylum);
      });
    }
    predict(origin, asylum, year, requestedUnit = 'auto') {
      const candidates = this.candidates(origin, asylum);
      const unit = requestedUnit === 'auto' ? (candidates.includes('P') ? 'P' : 'C') : requestedUnit;
      if (!candidates.includes(unit)) throw new Error('There is no model coverage for this country combination. Choose another country.');
      const m = this.data.models[unit];
      if (!Number.isInteger(year) || year < m.first_year || year > m.last_year + 5) {
        throw new Error(`Choose a whole decision year between ${m.first_year} and ${m.last_year + 5}.`);
      }
      const decade = (year - 2010) / 10;
      const f = [[`origin=${origin}`, 1], [`asylum=${asylum}`, 1], [`route=${origin}:${asylum}`, 1],
        ['year_decade', decade], [`origin_trend=${origin}`, decade], [`asylum_trend=${asylum}`, decade]];
      const scores = m.intercepts.map((intercept, k) => {
        let score = intercept;
        for (const [name, value] of f) {
          const index = this.indices[unit].get(name);
          if (index !== undefined) score += m.coefficients[k][index] * value;
        }
        return score;
      });
      const maximum = Math.max(...scores);
      const exps = scores.map(score => Math.exp(score - maximum));
      const total = exps.reduce((a, b) => a + b, 0);
      const p = exps.map(value => value / total);
      const support = m.routes[`${origin}:${asylum}`] || {decisions: 0, first_year: null, last_year: null};
      const notes = [];
      if (m.test_log_loss > m.route_baseline_test_log_loss) notes.push('Experimental: this reporting-unit regression underperformed the historical country-pair baseline on held-out log loss.');
      if (!support.decisions) notes.push('Unseen country pair: estimate uses country effects and trends, without a learned pair effect.');
      else if (support.decisions < 100) notes.push('Sparse pair: fewer than 100 recorded decisions across training years.');
      if (year > m.last_year) notes.push('Future-year extrapolation: policy and population changes are not modelled.');
      else notes.push('Historical fitted estimate: production training includes this year; this is not a held-out forecast.');
      if (support.last_year !== null && year > support.last_year + 1) notes.push('No recent observations for this pair in the selected reporting unit.');
      if (support.first_year !== null && year < support.first_year) notes.push('The selected year is earlier than this pair’s first observed records.');
      const protection = p[0] + p[1];
      return {origin, asylum, year, unit, reporting_unit: unit === 'P' ? 'persons' : 'cases',
        probabilities: Object.fromEntries(outcomes.map((name, k) => [name, p[k]])),
        probability_grant_or_other_protection: protection,
        probability_protection_given_substantive_decision: protection / (p[0] + p[1] + p[2]),
        training_years: [m.first_year, m.last_year], pair_support: support, notes};
    }
  }
  global.AsylumPredictor = AsylumPredictor;
})(globalThis);
