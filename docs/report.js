/* Browser-side model loading and scenario explorer. */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const categories = [
    ['grant', 'Grant', 'Recognised as a refugee', '#7950a0'],
    ['other_protection', 'Other protection', 'Complementary forms of protection', '#238477'],
    ['refusal', 'Refusal', 'Claim rejected', '#cc7658'],
    ['administrative_closure', 'Administrative closure', 'Closed without a substantive decision', '#7e899c']
  ];
  const percent = value => new Intl.NumberFormat('en-GB', {
    style: 'percent', maximumFractionDigits: 1, minimumFractionDigits: 1
  }).format(value);
  const integer = value => new Intl.NumberFormat('en-GB').format(value);
  let engine;
  let current;

  function fillOptions(select, codes, selected) {
    const sorted = [...codes].sort((a, b) => engine.data.names[a].localeCompare(engine.data.names[b]));
    select.replaceChildren(...sorted.map(code => {
      const option = document.createElement('option');
      option.value = code;
      option.textContent = engine.data.names[code] || code;
      return option;
    }));
    if (sorted.includes(selected)) select.value = selected;
  }

  function updateDestinations(preferred) {
    const origin = $('origin').value;
    const destinations = new Set();
    Object.values(engine.data.models).forEach(model => {
      if (model.origins.includes(origin)) model.asylum_countries.forEach(code => destinations.add(code));
    });
    fillOptions($('asylum'), [...destinations], preferred);
    updateBounds();
  }

  function updateBounds() {
    if (!engine || !$('origin').value || !$('asylum').value) return;
    const units = engine.candidates($('origin').value, $('asylum').value);
    const unit = units.includes('P') ? 'P' : units[0];
    if (!unit) return;
    const model = engine.data.models[unit];
    [$('year'), $('year-range')].forEach(input => {
      input.min = model.first_year;
      input.max = model.last_year + 5;
    });
    $('range-min').textContent = model.first_year;
    $('range-max').textContent = model.last_year + 5;
    $('year-help').textContent = `Years after ${model.last_year} extrapolate the model’s trends.`;
  }

  function makeNode(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  function renderOutcomes(result) {
    const container = $('outcomes');
    container.replaceChildren();
    categories.forEach(([key, label, description, color]) => {
      const row = makeNode('div', undefined, 'outcome-row');
      const copy = makeNode('div', undefined, 'outcome-copy');
      copy.append(makeNode('strong', label), makeNode('span', description));
      const value = result.probabilities[key];
      const bar = makeNode('div', undefined, 'outcome-bar');
      bar.setAttribute('role', 'img');
      bar.setAttribute('aria-label', `${label}: ${percent(value)}`);
      const fill = makeNode('span');
      fill.style.display = 'block';
      fill.style.width = `${value * 100}%`;
      fill.style.height = '100%';
      fill.style.backgroundColor = color;
      bar.append(fill);
      row.append(copy, bar, makeNode('strong', percent(value), 'outcome-value'));
      container.append(row);
    });
  }

  function renderTrend(result) {
    const container = $('trend-chart');
    container.replaceChildren();
    const [first, last] = result.training_years;
    const end = last + 5;
    const width = 800, height = 250, left = 48, right = 785, top = 18, bottom = 205;
    const x = year => left + (year - first) / Math.max(1, end - first) * (right - left);
    const y = value => bottom - value * (bottom - top);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `Modelled outcomes from ${first} to ${end}; years after ${last} are extrapolated.`);
    svg.style.width = '100%';
    svg.style.height = 'auto';
    const add = (tag, attrs, text) => {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
      if (text !== undefined) element.textContent = text;
      svg.append(element);
      return element;
    };
    add('rect', {x: x(last), y: top, width: Math.max(0, x(end) - x(last)), height: bottom - top, fill: '#f1eaf2'});
    [0, .25, .5, .75, 1].forEach(value => {
      add('line', {x1: left, y1: y(value), x2: right, y2: y(value), stroke: '#e4dce3'});
      add('text', {x: left - 8, y: y(value) + 4, 'text-anchor': 'end', fill: '#706475', 'font-size': 11}, `${value * 100}%`);
    });
    const years = [];
    for (let year = first; year <= end; year++) years.push(engine.predict(result.origin, result.asylum, year, result.unit));
    categories.forEach(([key, , , color]) => {
      const points = years.map((entry, index) => `${index ? 'L' : 'M'}${x(entry.year)},${y(entry.probabilities[key])}`).join(' ');
      add('path', {d: points, fill: 'none', stroke: color, 'stroke-width': 3, 'stroke-linecap': 'round'});
    });
    [first, Math.ceil(first / 10) * 10, Math.floor(last / 10) * 10, last, end]
      .filter((year, index, list) => year >= first && year <= end && list.indexOf(year) === index)
      .forEach(year => add('text', {x: x(year), y: bottom + 24, 'text-anchor': 'middle', fill: '#706475', 'font-size': 11}, String(year)));
    add('line', {x1: x(result.year), y1: top, x2: x(result.year), y2: bottom, stroke: '#33263e', 'stroke-width': 1.5, 'stroke-dasharray': '4 4'});
    $('trend-chart').append(svg);
  }

  function update() {
    if (!engine) return;
    const origin = $('origin').value;
    const asylum = $('asylum').value;
    const year = Number($('year').value);
    $('route-title').textContent = `${engine.data.names[origin]} → ${engine.data.names[asylum]}`;
    $('route-year').textContent = Number.isFinite(year) ? ` / ${year}` : '';
    try {
      if (!$('year').value.trim()) throw new Error('Enter a decision year.');
      current = engine.predict(origin, asylum, year);
      $('input-error').hidden = true;
      $('valid-results').hidden = false;
      $('year-range').value = year;
      $('year-badge').textContent = year > current.training_years[1] ? 'Projection' : 'Fitted history';
      $('year-badge').classList.toggle('historical', year <= current.training_years[1]);
      $('protection').textContent = percent(current.probability_grant_or_other_protection);
      $('recorded-decisions').textContent = current.pair_support.decisions ? integer(current.pair_support.decisions) : 'No direct records';
      $('reporting-basis').textContent = current.reporting_unit === 'persons' ? 'Persons' : 'Cases';
      renderOutcomes(current);
      renderTrend(current);
      $('update-status').textContent = `Updated for ${year}. Estimates are calculated in your browser.`;
      const download = $('download');
      if (download) download.disabled = false;
      const url = new URL(location.href);
      url.search = new URLSearchParams({origin, asylum, year}).toString();
      history.replaceState(null, '', url);
    } catch (error) {
      current = null;
      $('input-error').textContent = error.message;
      $('input-error').hidden = false;
      $('valid-results').hidden = true;
      const download = $('download');
      if (download) download.disabled = true;
    }
  }

  async function load() {
    try {
      const response = await fetch(new URL('model.json', location.href));
      if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
      engine = new AsylumPredictor(await response.json());
      const query = new URLSearchParams(location.search);
      const origins = new Set(Object.values(engine.data.models).flatMap(model => model.origins));
      const requestedOrigin = query.get('origin') || 'SYR';
      if (!origins.has(requestedOrigin)) throw new Error('This link contains an unsupported origin country.');
      fillOptions($('origin'), [...origins], requestedOrigin);
      updateDestinations(query.get('asylum') || 'GBR');
      if (!$('asylum').value) throw new Error('No asylum destinations are available for this origin.');
      const requestedAsylum = query.get('asylum') || 'GBR';
      if ($('asylum').value !== requestedAsylum) throw new Error('This link contains an unsupported asylum country.');
      $('year').value = query.get('year') || '2026';
      $('inputs').disabled = false;
      document.querySelectorAll('[data-origin]').forEach(button => { button.disabled = false; });
      $('loading').hidden = true;
      $('load-error').hidden = true;
      $('results').hidden = false;
      $('explorer').setAttribute('aria-busy', 'false');
      update();
    } catch (error) {
      $('loading').hidden = true;
      $('load-error').hidden = false;
      $('load-error').querySelector('p').textContent = `${error.message} Check your connection and try again.`;
      $('explorer').setAttribute('aria-busy', 'false');
    }
  }

  $('scenario').addEventListener('submit', event => { event.preventDefault(); update(); });
  $('origin').addEventListener('change', () => { updateDestinations($('asylum').value); update(); });
  $('asylum').addEventListener('change', () => { updateBounds(); update(); });
  $('year').addEventListener('input', update);
  $('year-range').addEventListener('input', () => { $('year').value = $('year-range').value; update(); });
  document.querySelectorAll('[data-origin]').forEach(button => button.addEventListener('click', () => {
    $('origin').value = button.dataset.origin;
    updateDestinations(button.dataset.asylum);
    update();
  }));
  const retry = $('retry');
  if (retry) retry.addEventListener('click', () => load());
  const reset = $('reset');
  if (reset) reset.addEventListener('click', () => {
    $('origin').value = 'SYR';
    updateDestinations('GBR');
    $('year').value = '2026';
    update();
    // update() synchronizes the URL to the displayed scenario; clear it afterward
    // so Reset also restores the clean, default link.
    const url = new URL(location.href);
    url.search = '';
    history.replaceState(null, '', url);
  });
  const download = $('download');
  if (download) download.addEventListener('click', () => {
    if (!current) return;
    const rows = [['origin', 'asylum', 'decision_year', 'reporting_unit', 'outcome', 'probability'],
      ...categories.map(([key]) => [current.origin, current.asylum, current.year, current.reporting_unit, key, current.probabilities[key]]),
      [current.origin, current.asylum, current.year, current.reporting_unit, 'grant_or_other_protection', current.probability_grant_or_other_protection]];
    const csv = rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n') + '\r\n';
    const objectUrl = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
    const anchor = document.createElement('a');
    anchor.href = objectUrl;
    anchor.download = `asylum-outcomes-${current.origin}-${current.asylum}-${current.year}.csv`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(objectUrl);
  });
  load();
})();
