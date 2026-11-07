(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const formatPercent = value => new Intl.NumberFormat('en-GB', {style: 'percent', maximumFractionDigits: 1, minimumFractionDigits: 1}).format(value);
  const formatInteger = value => new Intl.NumberFormat('en-GB').format(value);
  const categories = [
    ['grant', 'Grant', 'Recognised as a refugee', '#267866'],
    ['other_protection', 'Other protection', 'Complementary forms of protection', '#507dba'],
    ['refusal', 'Refusal', 'Claim rejected', '#c56b50'],
    ['administrative_closure', 'Administrative closure', 'Otherwise closed without a substantive decision', '#8c96a2']
  ];
  let engine, current;
  function options(select, codes, selected) {
    select.replaceChildren(...codes.sort((a, b) => engine.data.names[a].localeCompare(engine.data.names[b])).map(code => {
      const option = document.createElement('option');
      option.value = code; option.textContent = engine.data.names[code]; return option;
    }));
    if (codes.includes(selected)) select.value = selected;
  }
  function updateDestinations(preferred) {
    const origin = $('origin').value;
    const destinations = new Set();
    for (const m of Object.values(engine.data.models)) {
      if (m.origins.includes(origin)) for (const code of m.asylum_countries) destinations.add(code);
    }
    options($('asylum'), [...destinations], preferred);
    updateBounds();
  }
  function updateBounds() {
    const units = engine.candidates($('origin').value, $('asylum').value);
    const unit = units.includes('P') ? 'P' : units[0];
    if (!unit) return;
    const model = engine.data.models[unit];
    for (const input of [$('year'), $('year-range')]) {
      input.min = model.first_year; input.max = model.last_year + 5;
    }
    $('range-min').textContent = model.first_year;
    $('range-max').textContent = model.last_year + 5;
    $('year-help').textContent = `Years after ${model.last_year} extrapolate the model’s trends.`;
  }
  function node(tag, text, className) {
    const element = document.createElement(tag); element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  function chart(result) {
    const svg = $('trajectory-chart');
    svg.querySelectorAll('[data-drawn]').forEach(element => element.remove());
    const ns = 'http://www.w3.org/2000/svg';
    function shape(tag, attrs, text) {
      const element = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
      if (text !== undefined) element.textContent = text;
      element.setAttribute('data-drawn', 'true'); svg.append(element); return element;
    }
    const first = result.training_years[0], last = result.training_years[1], end = last + 5;
    const left = 45, right = 780, top = 16, bottom = 211;
    const x = year => left + (year - first) / (end - first) * (right - left);
    const y = p => bottom - p * (bottom - top);
    shape('rect', {x: x(last), y: top, width: x(end)-x(last), height: bottom-top, fill: '#f4ecdf'});
    for (const p of [0, .25, .5, .75, 1]) {
      shape('line', {x1: left, y1: y(p), x2: right, y2: y(p), stroke: '#e7ebe6', 'stroke-width': 1});
      shape('text', {x: left-10, y: y(p)+4, 'text-anchor': 'end', fill: '#617074', 'font-size': 10}, `${p*100}%`);
    }
    const ticks = new Set([first, Math.ceil((first+1)/10)*10, Math.ceil((first+11)/10)*10, last, end]);
    for (const year of [...ticks].filter(v => v <= end)) shape('text', {x:x(year), y:bottom+23, 'text-anchor':'middle', fill:'#617074', 'font-size':10}, year);
    shape('text', {x:x(last)+8, y:top+11, fill:'#8b6129', 'font-size':9}, 'PROJECTED');
    const yearly = [];
    for (let year=first; year<=end; year++) yearly.push(engine.predict(result.origin, result.asylum, year, result.unit));
    for (const [key, , , color] of categories) {
      for (const future of [false, true]) {
        const points = yearly.filter(r => future ? r.year >= last : r.year <= last);
        const d = points.map((r, i) => `${i ? 'L' : 'M'}${x(r.year)},${y(r.probabilities[key])}`).join(' ');
        const attrs = {d, fill:'none', stroke:color, 'stroke-width':2.2, 'stroke-linecap':'round', 'stroke-linejoin':'round'};
        if (future) attrs['stroke-dasharray'] = '5 4';
        shape('path', attrs);
      }
    }
    shape('line', {x1:x(result.year), y1:top, x2:x(result.year), y2:bottom, stroke:'#183b43', 'stroke-width':1, opacity:.5});
    for (const [key, , , color] of categories) shape('circle', {cx:x(result.year), cy:y(result.probabilities[key]), r:3.5, fill:color, stroke:'#fff', 'stroke-width':1.5});
    $('chart-description').textContent = `Modelled outcomes for ${engine.data.names[result.origin]} to ${engine.data.names[result.asylum]}, ${first} to ${end}. Years after ${last} are extrapolated. Selected year: ${result.year}. Outcome colours match the four rows above.`;
  }
  function update() {
    const origin = $('origin').value, asylum = $('asylum').value, year = Number($('year').value);
    $('route-title').textContent = `${engine.data.names[origin]} → ${engine.data.names[asylum]}`;
    $('route-year').textContent = Number.isFinite(year) ? ` / ${year}` : '';
    try {
      if (!$('year').value.trim()) throw new Error('Enter a decision year.');
      current = engine.predict(origin, asylum, year);
      $('input-error').hidden = true; $('valid-results').hidden = false;
      $('share').disabled = false; $('download').disabled = false;
      $('year-range').value = year;
      const projected = year > current.training_years[1];
      $('year-badge').textContent = projected ? 'Projection' : 'Fitted history';
      $('year-badge').classList.toggle('historical', !projected);
      $('protection').textContent = formatPercent(current.probability_grant_or_other_protection);
      $('conditional').textContent = formatPercent(current.probability_protection_given_substantive_decision);
      $('composition').replaceChildren(); $('outcome-rows').replaceChildren();
      for (const [key, label, description, color] of categories) {
        const p = current.probabilities[key];
        const segment = node('span', ''); segment.style.width = `${p*100}%`; segment.style.background = color;
        $('composition').append(segment);
        const row = node('div', '', 'outcome-row');
        const dot = node('span', '', 'outcome-dot'); dot.style.background = color; dot.setAttribute('aria-hidden', 'true');
        const copy = node('div', ''); copy.append(node('h4', label), node('p', description));
        row.append(dot, copy, node('strong', formatPercent(p), 'outcome-value'));
        $('outcome-rows').append(row);
      }
      $('reporting-basis').textContent = current.reporting_unit === 'persons' ? 'Persons' : 'Cases · experimental';
      $('pair-decisions').textContent = current.pair_support.decisions ? formatInteger(current.pair_support.decisions) : 'Unseen pair';
      $('pair-years').textContent = current.pair_support.first_year === null ? 'No direct observations' : `${current.pair_support.first_year}–${current.pair_support.last_year}`;
      $('prediction-notes').replaceChildren(...current.notes.map(text => node('p', text)));
      chart(current);
      const url = new URL(location.href); url.search = new URLSearchParams({origin, asylum, year}).toString();
      history.replaceState(null, '', url);
      $('update-status').textContent = `Updated for ${year} · all probabilities calculated in your browser`;
      document.documentElement.dataset.ready = 'true';
    } catch (error) {
      current = null; $('input-error').textContent = error.message; $('input-error').hidden = false;
      $('valid-results').hidden = true; $('share').disabled = true; $('download').disabled = true;
    }
  }
  async function load() {
    try {
      const response = await fetch('model.json');
      if (!response.ok) throw new Error(`Model download failed (${response.status})`);
      engine = new AsylumPredictor(await response.json());
      const query = new URLSearchParams(location.search);
      const origins = new Set(Object.values(engine.data.models).flatMap(m => m.origins));
      const requestedOrigin = query.get('origin') || 'SYR';
      options($('origin'), [...origins], requestedOrigin);
      if (!origins.has(requestedOrigin)) throw new Error('The shared link contains an unsupported origin country.');
      const requestedAsylum = query.get('asylum') || 'GBR';
      updateDestinations(requestedAsylum);
      if ($('asylum').value !== requestedAsylum) throw new Error('The shared link contains an unsupported asylum country.');
      $('year').value = query.get('year') || '2026';
      $('inputs').disabled = false;
      document.querySelectorAll('[data-origin]').forEach(button => { button.disabled = false; });
      $('loading').hidden = true; $('load-error').hidden = true; $('results').hidden = false;
      $('explorer').setAttribute('aria-busy', 'false'); update();
    } catch (error) {
      $('loading').hidden = true; $('load-error').hidden = false;
      $('load-error').querySelector('p').textContent = `${error.message} Try again, or remove any country parameters from the address.`;
      $('explorer').setAttribute('aria-busy', 'false');
    }
  }
  $('scenario').addEventListener('submit', event => { event.preventDefault(); update(); });
  $('origin').addEventListener('change', () => { updateDestinations($('asylum').value); update(); });
  $('asylum').addEventListener('change', () => { updateBounds(); update(); });
  $('year').addEventListener('input', update);
  $('year-range').addEventListener('input', () => { $('year').value = $('year-range').value; update(); });
  document.querySelectorAll('[data-origin]').forEach(button => button.addEventListener('click', () => {
    $('origin').value = button.dataset.origin; updateDestinations(button.dataset.asylum); update();
  }));
  $('share').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(location.href); $('update-status').textContent = 'Scenario link copied.'; }
    catch { $('update-status').textContent = 'Copy the address in your browser to share this scenario.'; }
  });
  $('download').addEventListener('click', () => {
    if (!current) return;
    const rows = [['origin', 'asylum', 'decision_year', 'reporting_unit', 'outcome', 'probability'],
      ...categories.map(([key]) => [current.origin, current.asylum, current.year, current.reporting_unit, key, current.probabilities[key]])];
    rows.push([current.origin, current.asylum, current.year, current.reporting_unit, 'grant_or_other_protection', current.probability_grant_or_other_protection]);
    const blob = new Blob([rows.map(row => row.join(',')).join('\r\n')+'\r\n'], {type:'text/csv;charset=utf-8'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `asylum-outcomes-${current.origin}-${current.asylum}-${current.year}.csv`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    $('update-status').textContent = 'Estimate downloaded. Future years are extrapolations.';
  });
  $('retry').addEventListener('click', () => { location.search = ''; });
  load();
})();
