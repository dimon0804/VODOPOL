/* Водополь — страницы панели в новом дизайне.
 *
 * Этот файл раскладывает панель по разделам и дорисовывает блоки из макета.
 * Данные и проверенная логика — в app.js (window.VODOPOL): сюда приходят те же
 * ответы сервиса, панель по-прежнему ничего не пересчитывает. Где в макете
 * нарисован блок, под который у нас нет измерений, здесь показано то, что есть на
 * самом деле, а не числа из макета.
 */
(function () {
  'use strict';

  var V = null;               // window.VODOPOL, появляется после app.js
  var REPORTS = {};           // кэш отчётов модели
  var RANKS = { run: null, rows: null };
  var ROUTE = { page: 'home', tab: '' };
  var DEFAULT_TAB = { methods: 'quality' };
  var zoneLabels = null;      // слой подписей Z1…Zn на карте
  var methodsPart = 'test';   // test | holdout на вкладке качества

  function $(id) { return document.getElementById(id); }
  function esc(v) { return V.esc(v); }
  function num(v) { return V.toNum(v); }
  function fmt(v, d) { return V.fmt(v, d); }
  function money(v) { return V.money(v); }
  function pct(v) { return V.pct(v); }
  var DASH = '—';

  function mln(v) {
    var n = num(v);
    if (n === null) return DASH;
    return (n / 1e6).toLocaleString('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 2 });
  }
  function rub0(v) {
    var n = num(v);
    return n === null ? DASH : Math.round(n).toLocaleString('ru-RU');
  }
  function f3(v) { var n = num(v); return n === null ? DASH : n.toLocaleString('ru-RU', { minimumFractionDigits: 3, maximumFractionDigits: 3 }); }
  function cap(t) { t = String(t || ''); return t.charAt(0).toUpperCase() + t.slice(1); }
  function assetNo(id) { return String(id || '').replace(/^a0*/, ''); }

  // ── маршрутизация ─────────────────────────────────────────────────────────

  function parseHash() {
    var parts = (location.hash || '').replace(/^#\/?/, '').split('/');
    var page = parts[0] || 'home';
    if (!$('page-' + page)) page = 'home';
    return { page: page, tab: parts[1] || DEFAULT_TAB[page] || '' };
  }

  function route() {
    ROUTE = parseHash();
    Array.prototype.forEach.call(document.querySelectorAll('.page'), function (el) {
      el.classList.toggle('active', el.getAttribute('data-page') === ROUTE.page);
    });
    Array.prototype.forEach.call(document.querySelectorAll('#nav a'), function (a) {
      a.classList.toggle('active', a.getAttribute('data-page') === ROUTE.page);
    });
    Array.prototype.forEach.call(document.querySelectorAll('.page.active .tabs a'), function (a) {
      a.classList.toggle('active', a.getAttribute('data-tab') === ROUTE.tab);
    });
    Array.prototype.forEach.call(document.querySelectorAll('.page.active .tab'), function (el) {
      el.classList.toggle('active', el.getAttribute('data-tab') === ROUTE.tab);
    });
    placeShared();
    renderRoute();
    window.scrollTo(0, 0);
  }

  /**
   * Карта и ползунок бюджета существуют в одном экземпляре: переезжают на ту
   * страницу, где нужны. Две карты или два ползунка разошлись бы между собой.
   */
  function placeShared() {
    var slot = ROUTE.page === 'plan' ? 'plan' : 'map';
    var mapSlot = document.querySelector('.mapslot[data-slot="' + slot + '"]');
    var mapWrap = $('mapwrap');
    if (mapSlot && mapWrap && mapWrap.parentNode !== mapSlot) mapSlot.appendChild(mapWrap);
    var budgetSlot = document.querySelector('.budget-slot[data-slot="' + slot + '"]');
    var core = $('budget-core');
    if (budgetSlot && core && core.parentNode !== budgetSlot) budgetSlot.appendChild(core);
    if (V && V.S.map && (ROUTE.page === 'map' || ROUTE.page === 'plan')) {
      // Leaflet не знает, что контейнер переехал: размер и кадр надо пересчитать.
      setTimeout(function () { V.S.map.invalidateSize(); V.fitChip(); }, 30);
      setTimeout(function () { V.S.map.invalidateSize(); V.fitChip(); }, 350);
    }
  }

  function renderRoute() {
    if (!V || !V.S.summary) return;
    if (ROUTE.page === 'objects') renderObjects();
    if (ROUTE.page === 'plan') renderPlan();
    if (ROUTE.page === 'methods') renderMethods();
  }

  // ── подписи зон Z1…Zn ─────────────────────────────────────────────────────

  /** Короткие имена зон: номер по порядку в каталоге. Один и тот же на всех страницах. */
  function zoneIndex() {
    var feats = ((V.S.candidates && V.S.candidates.features) || []).slice();
    feats.sort(function (a, b) { return String(a.properties.candidate_id).localeCompare(String(b.properties.candidate_id)); });
    var map = {};
    feats.forEach(function (f, i) { map[f.properties.candidate_id] = 'Z' + (i + 1); });
    return map;
  }
  function zoneLabel(id) { return (V.S._zlabels || {})[id] || id; }

  function renderZoneLabels() {
    var S = V.S;
    if (!S.map || !S.candidates) return;
    if (zoneLabels) { S.map.removeLayer(zoneLabels); zoneLabels = null; }
    var plan = V.selectedSet();
    zoneLabels = L.layerGroup();
    // Заказанные зоны — поверх каталога: у соседних ячеек общие края, и белый
    // пунктир каталога иначе ложится на синюю рамку и превращает её в пунктир.
    if (S.zonesLayer) S.zonesLayer.eachLayer(function (l) {
      if (l.feature && plan[l.feature.properties.candidate_id]) l.bringToFront();
    });
    if (S.assetsLayer && S.assetsLayer.bringToFront) S.assetsLayer.bringToFront();
    S.candidates.features.forEach(function (f) {
      var id = f.properties.candidate_id;
      if (!plan[id]) return;
      var ring = f.geometry.coordinates[0];
      var maxLon = -Infinity, maxLat = -Infinity;
      ring.forEach(function (c) { if (c[0] > maxLon) maxLon = c[0]; if (c[1] > maxLat) maxLat = c[1]; });
      L.marker([maxLat, maxLon], {
        icon: L.divIcon({ className: 'zlab', html: '<span>' + esc(zoneLabel(id)) + '</span>', iconSize: [0, 0] }),
        interactive: false, keyboard: false
      }).addTo(zoneLabels);
    });
    if ($('chk-zones').checked) zoneLabels.addTo(S.map);
  }

  // ── карта: заголовок, KPI, бюджет, «проверить в первую очередь» ──────────

  function renderMapHead() {
    var s = V.S.summary;
    $('map-title').textContent = 'Sentinel-1 · ' + (s.observation_date ? new Date(s.observation_date).toLocaleDateString('ru-RU') : 'дата неизвестна');
    $('map-sub').textContent = (s.chip_id || '') + ' · событие ' + (s.event_id || DASH) + ' · 512 × 512 пикс. · 10 м · EPSG:4326';
  }

  function renderKPI() {
    var s = V.S.summary, imp = (s.impact && s.impact.totals) || {};
    var a = V.comparisonRow('A') || {};
    var html = '<div class="kpi-row">' +
      '<div class="kpi"><div class="kpi-v">' + mln(s.total_expected_loss_rub) + '<small>млн ₽</small></div>' +
      '<div class="kpi-l">ожидаемый ущерб</div></div>' +
      '<div class="kpi"><div class="kpi-v">' + fmt(imp.expected_area_km2, 2) + '<small>км²</small></div>' +
      '<div class="kpi-l">ожидаемая площадь затопления</div></div>' +
      '<div class="kpi"><div class="kpi-v">' + fmt(a.residual_uncertainty, 2) + '</div>' +
      '<div class="kpi-l">неопределённость портфеля</div></div></div>' +
      '<div class="kpi-foot">' + fmt(imp.assessed_objects, 0) + ' из ' + fmt(imp.total_objects, 0) +
      ' объектов оценены · по маске ' + fmt(imp.mask_area_km2, 2) + ' км² · ' + V.mark('model', 'расчёт модели') + '</div>';
    $('card-kpi').innerHTML = html;
  }

  function planPositions() {
    return ((V.S.strategies && V.S.strategies.positions) || []).filter(function (p) { return p.strategy === V.S.strategy; });
  }

  function renderBudgetRows() {
    var S = V.S, row = V.comparisonRow(S.strategy) || {};
    var pos = planPositions();
    var area = pos.reduce(function (a, p) { return a + (num(p.area_km2) || 0); }, 0);
    var zones = pos.map(function (p) { return zoneLabel(p.candidate_id); }).join(', ');
    var feasible = row.budget_feasible === true;
    var html =
      '<dt>' + (!pos.length ? 'Платных зон нет' : pos.length > 4 ? pos.length + ' ' + V.plural(pos.length, 'зона', 'зоны', 'зон') : 'Зоны ' + esc(zones)) + '</dt><dd>' +
        (pos.length ? fmt(area, 2) + ' км² · ' : '') + money(row.decision_cost_rub) + ' ₽' +
        (S.strategy === 'B' && !feasible ? ' <span class="mark bad">контрфактическая</span>' : '') + '</dd>' +
      '<dt>Охват ожидаемого ущерба</dt><dd>' + pct(row.coverage_share) + '</dd>' +
      '<dt>Остаточная неопределённость</dt><dd>' + fmt(row.residual_uncertainty, 2) + ' ' +
        V.mark(row.uncertainty_status === 'baseline' ? 'model' : 'scenario', row.uncertainty_status === 'baseline' ? 'исходная' : 'сценарная') + '</dd>';
    $('budget-rows').innerHTML = html;
    $('budget-num-plan').textContent = fmt(S.budget, 0);
  }

  function rankedAssets() {
    return ((V.S.assets && V.S.assets.features) || []).map(function (f) { return f.properties; })
      .filter(function (p) { return p.status === 'ok' && p.rank != null; })
      .sort(function (a, b) { return a.rank - b.rank; });
  }

  function renderFirst() {
    var covered = V.coveredSet();
    var html = rankedAssets().slice(0, 3).map(function (p) {
      return '<li data-asset="' + esc(p.asset_id) + '"><span class="badge fill">' + esc(assetNo(p.asset_id)) + '</span>' +
        '<div class="fl-main"><b>' + esc(cap(V.ASSET_TITLES[p.asset_class] || p.asset_class)) + '</b>' +
        '<span class="muted small">p = ' + fmt(p.p_flood, 2) + ' · неопр. ' + fmt(p.uncertainty, 2) +
        (covered[p.asset_id] ? ' · покрыт съёмкой' : ' · только наземно') + '</span></div>' +
        '<div class="fl-val">' + mln(p.expected_loss_rub) + ' млн ₽</div></li>';
    }).join('');
    $('first-list').innerHTML = html || '<li class="muted">оценённых объектов нет</li>';
    Array.prototype.forEach.call($('first-list').querySelectorAll('li[data-asset]'), function (li) {
      li.addEventListener('click', function () { openAsset(li.getAttribute('data-asset')); });
    });
  }

  /** Открыть объект на карте и показать его в карточке под картой. */
  function openAsset(id) {
    if (ROUTE.page !== 'map') location.hash = '#/map';
    setTimeout(function () { selectAsset(id, true); }, 150);
  }

  // ── выбор на карте: карточка под картой, как в «Фенологе» ────────────────

  var picked = null;   // маркер выбранного объекта

  function clearPick() {
    if (picked && picked._icon) L.DomUtil.removeClass(picked._icon, 'sel');
    picked = null;
    if (V.S.selectedZone) { V.S.selectedZone = null; V.restyleZones(); }
    $('map-pick').innerHTML = '<div class="pick-empty"><b>Объект не выбран</b><span>Нажмите объект или зону на карте — ' +
      'здесь появится, откуда взялся ущерб и во что обойдётся проверка.</span></div>';
  }

  function pickCard(html) {
    $('map-pick').innerHTML = '<button class="pick-x" type="button" title="Снять выбор">×</button><div class="pick-body">' + html + '</div>';
    $('map-pick').querySelector('.pick-x').addEventListener('click', clearPick);
    $('map-pick').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function selectAsset(id, fly) {
    var S = V.S, found = null;
    if (!S.assetsLayer) return;
    S.assetsLayer.eachLayer(function (l) { if (l.feature && l.feature.properties.asset_id === id) found = l; });
    if (!found) return;
    if (picked && picked._icon) L.DomUtil.removeClass(picked._icon, 'sel');
    picked = found;
    if (found._icon) L.DomUtil.addClass(found._icon, 'sel');
    if (S.selectedZone) { S.selectedZone = null; V.restyleZones(); }
    if (fly && !S.map.getBounds().pad(-0.1).contains(found.getLatLng())) S.map.panTo(found.getLatLng());
    pickCard(V.assetPopup(found.feature.properties, found.feature.geometry));
  }

  function selectZone(id) {
    var S = V.S;
    var f = S.candidates.features.filter(function (x) { return x.properties.candidate_id === id; })[0];
    if (!f) return;
    if (picked && picked._icon) L.DomUtil.removeClass(picked._icon, 'sel');
    picked = null;
    S.selectedZone = id;
    V.restyleZones();
    V.showZoneCard(f);   // та же карточка зоны, что на «Плане съёмки»
    pickCard('<div class="popup"><h4>Зона ' + esc(zoneLabel(id)) + '</h4></div>' + $('zone-body').innerHTML);
  }

  /** Клики по объектам и зонам ведут в карточку под картой, а не во всплывающее окно. */
  function bindMapClicks() {
    var S = V.S;
    if (S.assetsLayer) S.assetsLayer.eachLayer(function (l) {
      l.off('click');
      l.unbindPopup();
      l.on('click', function () { selectAsset(l.feature.properties.asset_id, false); });
    });
    if (S.zonesLayer) S.zonesLayer.eachLayer(function (l) {
      l.off('click');
      l.unbindPopup();
      l.on('click', function () { selectZone(l.feature.properties.candidate_id); });
    });
  }

  function setBasemap(kind) {
    var S = V.S;
    if (!S.basemaps || S.basemap === kind) return;
    S.map.removeLayer(S.basemaps[S.basemap]);
    S.basemaps[kind].addTo(S.map);
    S.basemaps[kind].bringToBack();
    S.basemap = kind;
    $('mapwrap').classList.toggle('base-osm', kind === 'osm');
    Array.prototype.forEach.call(document.querySelectorAll('#basemap-switch button'), function (b) {
      b.classList.toggle('active', b.getAttribute('data-base') === kind);
    });
  }

  // ── объекты и ущерб ──────────────────────────────────────────────────────

  function renderObjects() {
    var S = V.S, thr = num(S.summary.threshold);
    var feats = ((S.assets && S.assets.features) || []).map(function (f) { return f.properties; });
    feats.sort(function (a, b) { return Number(assetNo(a.asset_id)) - Number(assetNo(b.asset_id)); });
    var maxEL = feats.reduce(function (m, p) { return Math.max(m, num(p.expected_loss_rub) || 0); }, 0) || 1;
    var html = '<thead><tr><th>№</th><th>Объект</th><th class="r">p</th><th>Неопределённость</th>' +
      '<th class="r">V, млн ₽</th><th class="r">q</th><th>E[ущерб], млн ₽</th><th>Маска</th><th>Приоритет</th></tr></thead><tbody>';
    feats.forEach(function (p) {
      var ok = p.status === 'ok';
      var tier = ok ? V.priorityTier(p.rank, V.rankedCount()) : V.PRIORITY_TIERS[3];
      var unc = num(p.uncertainty), el = num(p.expected_loss_rub), pf = num(p.p_flood);
      var inMask = ok && thr !== null && pf !== null && pf >= thr;
      html += '<tr data-asset="' + esc(p.asset_id) + '">' +
        '<td><span class="badge' + (tier.key === 'high' ? ' fill' : '') + '">' + esc(assetNo(p.asset_id)) + '</span></td>' +
        '<td class="oname">' + esc(cap(V.ASSET_TITLES[p.asset_class] || p.asset_class)) + '</td>' +
        '<td class="r">' + (ok ? fmt(pf, 2) : DASH) + '</td>' +
        '<td>' + (ok && unc !== null ? '<span class="ubar"><i style="width:' + Math.round(unc * 100) + '%"></i></span><span class="uval">' + fmt(unc, 2) + '</span>' : '<span class="muted">нет оценки</span>') + '</td>' +
        '<td class="r">' + fmt((num(p.asset_value_rub) || 0) / 1e6, 0) + '</td>' +
        '<td class="r">' + fmt(p.vulnerability_coef, 2) + '</td>' +
        '<td><span class="elv">' + (ok ? mln(el) : DASH) + '</span>' + (ok ? '<span class="elbar"><i style="width:' + Math.max(2, Math.round((el || 0) / maxEL * 100)) + '%"></i></span>' : '') + '</td>' +
        '<td>' + (ok ? (inMask ? '<span class="dot on"></span>да' : '<span class="dot"></span>нет') : DASH) + '</td>' +
        '<td>' + (tier.key === 'high' ? '<span class="tag">Высокий</span>' : '<span class="muted">' + esc(tier.title) + '</span>') + '</td></tr>';
    });
    html += '</tbody><tfoot><tr><td></td><td><b>Итого</b></td><td colspan="4"></td><td><b>' + mln(S.summary.total_expected_loss_rub) +
      ' млн ₽</b></td><td colspan="2" class="muted small r">полоса — доля от наибольшего ущерба · приоритет — треть портфеля по рангу EL</td></tr></tfoot>';
    $('obj-table').innerHTML = html;
    Array.prototype.forEach.call($('obj-table').querySelectorAll('tbody tr'), function (tr) {
      tr.addEventListener('click', function () { openAsset(tr.getAttribute('data-asset')); });
    });
    setDownloads();
    renderRankSensitivity();
  }

  // Минимальный разбор CSV: кавычки, запятые внутри кавычек, удвоенные кавычки.
  function parseCSV(text) {
    var rows = [], row = [], cell = '', q = false;
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (q) {
        if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (ch === '"') q = false;
        else cell += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(cell); rows.push(row); row = []; cell = '';
      } else cell += ch;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    var head = rows.shift() || [];
    return rows.filter(function (r) { return r.length > 1; }).map(function (r) {
      var o = {}; head.forEach(function (h, k) { o[h] = r[k]; }); return o;
    });
  }

  var CLASS_GEN = {
    warehouse: 'склада', substation: 'подстанции', road_unit: 'участка дороги', facility: 'производства',
    pumping_station: 'насосной', clinic: 'клиники', school: 'школы', workshop: 'мастерской',
    telecom_node: 'узла связи', water_intake: 'водозабора'
  };

  function scenarioTitle(id) {
    var m;
    if ((m = /^p_uncertain_x([\d.]+)$/.exec(id))) return 'p × ' + m[1].replace('.', ',') + ' у неуверенных';
    if ((m = /^([Vq])_(\w+?)_x([\d.]+)$/.exec(id))) return m[1] + ' ' + (CLASS_GEN[m[2]] || m[2]) + ' × ' + m[3].replace('.', ',');
    if (id === 'no_data_top1') return 'нет данных у первого';
    return id;
  }

  function renderRankSensitivity() {
    var run = V.currentRunId();
    if (RANKS.run === run && RANKS.rows) return drawRankSensitivity(RANKS.rows);
    $('rank-sens').innerHTML = '<div class="muted">загрузка…</div>';
    fetch(V.withRun('/api/files/sensitivity_ranks.csv')).then(function (r) {
      if (!r.ok) throw new Error('в комплекте нет sensitivity_ranks.csv');
      return r.text();
    }).then(function (t) {
      RANKS = { run: run, rows: parseCSV(t) };
      drawRankSensitivity(RANKS.rows);
    }).catch(function (e) {
      $('rank-sens').innerHTML = '<div class="muted">' + esc(e.message) + '</div>';
    });
  }

  function drawRankSensitivity(rows) {
    var by = {}, order = [];
    rows.forEach(function (r) {
      if (!by[r.scenario_id]) { by[r.scenario_id] = []; order.push(r.scenario_id); }
      by[r.scenario_id].push(r);
    });
    function top3(list, key) {
      return list.filter(function (r) { return r[key] !== ''; })
        .sort(function (a, b) { return Number(a[key]) - Number(b[key]); }).slice(0, 3)
        .map(function (r) { return r.asset_id; });
    }
    function sum(list, key) { return list.reduce(function (a, r) { return a + (Number(r[key]) || 0); }, 0); }
    function chips(ids, fill) {
      return ids.map(function (id) { return '<span class="badge' + (fill ? ' fill' : '') + '">' + esc(assetNo(id)) + '</span>'; }).join('');
    }
    var first = by[order[0]] || [];
    var baseTop = top3(first, 'rank_base');
    var html = '<div class="scard"><div class="s-h">Базовый</div><div class="s-v">' + mln(sum(first, 'expected_loss_base_rub')) +
      ' <small>млн ₽</small></div><div class="s-c">' + chips(baseTop, false) + '</div></div>';
    order.forEach(function (id) {
      var list = by[id], t3 = top3(list, 'rank_scenario');
      var changed = t3.join() !== baseTop.join();
      html += '<div class="scard"><div class="s-h">' + esc(scenarioTitle(id)) + '</div><div class="s-v">' +
        mln(sum(list, 'expected_loss_scenario_rub')) + ' <small>млн ₽</small></div><div class="s-c">' + chips(t3, changed) +
        (changed ? '<span class="warn-t">тройка изменилась</span>' : '<span class="muted small">тройка та же</span>') + '</div></div>';
    });
    $('rank-sens').innerHTML = html;
  }

  // ── план съёмки ──────────────────────────────────────────────────────────

  function renderPlan() {
    var S = V.S;
    if (!S.strategies) return;
    $('budget-num-plan').textContent = fmt(S.budget, 0);
    // Отметки стоимости A, C и B на шкале бюджета: где кончается нужное.
    var rng = $('rng-budget'), max = Number(rng.max) || 1;
    var marks = ['A', 'C', 'B'].map(function (code) {
      var row = V.comparisonRow(code) || {}, cost = num(row.decision_cost_rub) || 0;
      var left = Math.min(100, cost / max * 100);
      var late = code === 'B' && row.budget_feasible !== true;
      return '<span class="bmark' + (late ? ' over' : '') + '" style="left:' + left.toFixed(1) + '%">' +
        '<i></i>' + code + (code === 'A' ? '' : ' · ' + rub0(cost) + ' ₽') + (late ? ' · вне бюджета' : '') + '</span>';
    }).join('');
    $('budget-marks').innerHTML = marks;

    var cards = [
      { code: 'A', name: 'Без съёмки', sub: 'только исходный прогноз' },
      { code: 'B', name: 'Широкая', sub: '' },
      { code: 'C', name: 'Выборочная', sub: '' }
    ].map(function (c) {
      var row = V.comparisonRow(c.code) || {}, cost = num(row.decision_cost_rub);
      var over = c.code === 'B' && row.budget_feasible !== true;
      var pos = ((S.strategies.positions) || []).filter(function (p) { return p.strategy === c.code; });
      var area = pos.reduce(function (a, p) { return a + (num(p.area_km2) || 0); }, 0);
      var sub = c.sub || (over ? 'контрфактическая, сверх бюджета' : pos.length + ' ' + V.plural(pos.length, 'зона', 'зоны', 'зон') + ' · ' + fmt(area, 2) + ' км²');
      var cov = num(row.coverage_share) || 0, unc = num(row.residual_uncertainty) || 0;
      return '<div class="scard2' + (S.strategy === c.code ? ' on' : '') + '" data-strategy="' + c.code + '">' +
        '<div class="sc-h"><span class="badge' + (S.strategy === c.code ? ' fill' : '') + '">' + c.code + '</span>' + c.name + '</div>' +
        '<div class="sc-v' + (over ? ' over' : '') + '">' + rub0(cost) + ' ₽</div>' +
        '<div class="sc-sub' + (over ? ' over' : '') + '">' + esc(sub) + '</div>' +
        '<div class="sc-m"><span>Охват ущерба</span><b>' + pct(cov) + '</b></div><div class="mbar"><i style="width:' + Math.round(cov * 100) + '%"></i></div>' +
        '<div class="sc-m"><span>Остаточная неопр.</span><b>' + fmt(unc, 2) + '</b></div><div class="mbar unc"><i style="width:' + Math.round(unc * 100) + '%"></i></div>' +
        '<div class="sc-f">' + (row.uncertainty_status === 'baseline' ? 'исходная оценка модели' : 'сценарий: новых снимков ещё нет') + '</div></div>';
    }).join('');
    $('strat-cards').innerHTML = cards;
    Array.prototype.forEach.call($('strat-cards').querySelectorAll('.scard2'), function (el) {
      el.addEventListener('click', function () { V.setStrategy(el.getAttribute('data-strategy')); });
    });

    renderCatalog();
    setDownloads();
  }

  function renderCatalog() {
    var S = V.S, plan = V.selectedSet(), deadline = S.summary.decision_deadline || '';
    var price = {};
    planPositions().forEach(function (p) { price[p.candidate_id] = p.cost_rub; });
    var el = {};
    ((S.assets && S.assets.features) || []).forEach(function (f) { el[f.properties.asset_id] = num(f.properties.expected_loss_rub) || 0; });
    var feats = S.candidates.features.slice().sort(function (a, b) {
      var pa = plan[a.properties.candidate_id] ? 0 : 1, pb = plan[b.properties.candidate_id] ? 0 : 1;
      if (pa !== pb) return pa - pb;
      return String(zoneLabel(a.properties.candidate_id)).localeCompare(String(zoneLabel(b.properties.candidate_id)), 'ru', { numeric: true });
    });
    var html = '<thead><tr><th></th><th>Зона</th><th class="r">Площадь</th><th>Объекты</th><th class="r">Ущерб объектов</th><th class="r">Цена в корзине</th></tr></thead><tbody>';
    feats.forEach(function (f) {
      var p = f.properties, id = p.candidate_id, on = !!plan[id];
      var ids = p.covered_asset_ids || [];
      var loss = ids.reduce(function (a, x) { return a + (el[x] || 0); }, 0);
      var ctx = p.data_role !== 'event_observation';
      var late = !ctx && deadline && p.available_at && p.available_at > deadline;
      html += '<tr class="' + (on ? 'on' : '') + '" data-zone="' + esc(id) + '">' +
        '<td><span class="cbox' + (on ? ' on' : '') + '"></span></td>' +
        '<td><b>' + esc(zoneLabel(id)) + '</b> <span class="muted small">' + esc(id) + '</span>' +
          (ctx ? ' <span class="mark none">контекст</span>' : '') + (late ? ' <span class="mark bad">не успевает</span>' : '') + '</td>' +
        '<td class="r">' + fmt(p.area_km2, 2) + ' км²</td>' +
        '<td>' + (ids.length ? ids.map(assetNo).join(', ') : '<span class="muted">—</span>') + '</td>' +
        '<td class="r">' + (ids.length ? mln(loss) + ' млн ₽' : '<span class="muted">—</span>') + '</td>' +
        '<td class="r">' + (on && price[id] != null ? money(price[id]) + ' ₽' : '<span class="muted" title="Цена зоны зависит от состава корзины: скидка Р считается на всю корзину">вне корзины</span>') + '</td></tr>';
    });
    $('cat-table').innerHTML = html + '</tbody>';
    $('catalog-note').textContent = 'выбор ' + S.strategy + ': ' + (S.strategy === 'C' ? 'макс. польза на рубль в пределах бюджета' : S.strategy === 'B' ? 'все зоны наблюдения события' : 'платных зон нет');
    Array.prototype.forEach.call($('cat-table').querySelectorAll('tbody tr'), function (tr) {
      tr.addEventListener('click', function () {
        var id = tr.getAttribute('data-zone');
        var f = S.candidates.features.filter(function (x) { return x.properties.candidate_id === id; })[0];
        if (!f) return;
        S.selectedZone = id;
        V.restyleZones();
        V.showZoneCard(f);
        $('card-zone').scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
  }

  // ── методы и качество ────────────────────────────────────────────────────

  function report(name) {
    if (REPORTS[name]) return Promise.resolve(REPORTS[name]);
    return V.getJSON('/api/reports/' + name).then(function (d) { REPORTS[name] = d; return d; });
  }

  function renderMethods() {
    if (ROUTE.tab === 'quality') return renderQuality();
    if (ROUTE.tab === 'proba') return renderProba();
    if (ROUTE.tab === 'data') return renderData();
  }

  function failBox(id, e) {
    $(id).innerHTML = '<section class="card"><p class="muted">Отчёт не загружен: ' + esc(e.message) + '</p></section>';
  }

  function renderQuality() {
    Promise.all([report('metrics_compare.json'), report('metrics_compare_holdout.json')]).then(function (r) {
      var test = r[0], hold = r[1], src = methodsPart === 'test' ? test : hold;
      var b = src.totals.baseline_flood, m = src.totals.main_flood;
      var rows = [['IoU', 'iou'], ['F1 / Dice', 'f1'], ['Precision', 'precision'], ['Recall', 'recall']];
      var tbl = rows.map(function (x) {
        var d = m[x[1]] - b[x[1]];
        return '<tr><td>' + x[0] + '</td><td class="r muted">' + f3(b[x[1]]) + '</td><td class="r"><b>' + f3(m[x[1]]) + '</b></td>' +
          '<td class="r ' + (d < 0 ? 'neg' : 'pos') + '">' + (d >= 0 ? '+' : '−') + f3(Math.abs(d)) + '</td></tr>';
      }).join('');

      var events = [];
      Object.keys(test.by_event).concat(Object.keys(hold.by_event)).forEach(function (k) {
        var parts = k.split('|');
        if (parts[1] !== 'flood') return;
        if (events.indexOf(parts[2]) < 0) events.push(parts[2]);
      });
      var all = Object.assign({}, test.by_event, hold.by_event);
      var bars = events.map(function (ev) {
        return { ev: ev, b: all['baseline|flood|' + ev].iou, m: all['main|flood|' + ev].iou, hold: !!hold.by_event['main|flood|' + ev] };
      });

      var fpB = src.totals.baseline_fp_permanent.share, fpM = src.totals.main_fp_permanent.share;
      $('m-quality').innerHTML =
        '<div class="grid-2">' +
        '<section class="card"><div class="card-head"><h2>Baseline и основной метод</h2>' +
          '<div class="segmented small" id="part-switch"><button data-part="test"' + (methodsPart === 'test' ? ' class="active"' : '') + '>Test</button>' +
          '<button data-part="holdout"' + (methodsPart === 'holdout' ? ' class="active"' : '') + '>Bolivia</button></div></div>' +
          '<div class="muted small">' + (methodsPart === 'test' ? 'Test' : 'Bolivia — отложенное событие') + ' · ' + src.chips + ' ' + V.plural(src.chips, 'чип', 'чипа', 'чипов') + ' · ' +
            fmt(b.n_pixels, 0) + ' пикселей · цель — временное затопление</div>' +
          '<table class="mtable"><thead><tr><th></th><th class="r">Baseline<br><span class="muted small">VV &lt; −14,85 дБ, медиана 5</span></th>' +
          '<th class="r">Основной<br><span class="muted small">LightGBM · τ = ' + fmt(src.main_threshold, 2) + '</span></th><th class="r">Δ</th></tr></thead><tbody>' + tbl + '</tbody></table>' +
          '<p class="note">По пиксельным метрикам на независимой проверке <b>пороговый baseline не хуже основного метода</b>, на Bolivia — заметно лучше. ' +
          'Ценность основного метода не в IoU: он даёт <b>откалиброванную вероятность и неопределённость</b>, на которых стоит расчёт ущерба EL = p × V × q, ' +
          'и реже путает разлив с постоянной водой. У baseline есть только маска.</p></section>' +
        '<section class="card"><div class="card-head"><h2>IoU по событиям</h2>' +
          '<span class="lgd"><i class="c-b"></i>Baseline <i class="c-m"></i>Основной</span></div>' + barsSVG(bars) +
          '<p class="note">* Bolivia — перенос на событие, не участвовавшее ни в обучении, ни в калибровке, ни в выборе порога. ' +
          'Показаны события независимой проверки; на событиях обучения качество не приводится — это было бы подсматривание.</p></section></div>' +
        '<section class="card"><div class="card-head"><h2>Где ошибаются методы</h2></div>' +
          '<div class="facts">' +
          fact('Ложные срабатывания на постоянной воде', 'основной ' + pct(fpM) + ' · baseline ' + pct(fpB),
            'Доля ложных срабатываний, пришедшихся на реки и озёра. Основной метод в ' + fmt(fpB / fpM, 1) + ' раза реже выдаёт постоянную воду за разлив — ущерб от этого не завышается.') +
          fact('Полнота', 'основной ' + pct(m.recall) + ' · baseline ' + pct(b.recall),
            'Доля реально затопленных пикселей, найденных методом.') +
          fact('Точность', 'основной ' + pct(m.precision) + ' · baseline ' + pct(b.precision),
            'Из названного затоплением — сколько действительно затоплено.') +
          fact('Худшее событие', worstEvent(bars), 'Качество сильно зависит от сцены: одна модель на всех событиях работает неровно.') +
          '</div><p class="note">Попиксельная карта ошибок для открытого чипа в панель пока не выведена: для неё нужен слой сравнения с ручной меткой от расчётного модуля.</p></section>';

      Array.prototype.forEach.call(document.querySelectorAll('#part-switch button'), function (btn) {
        btn.addEventListener('click', function () { methodsPart = btn.getAttribute('data-part'); renderQuality(); });
      });
    }).catch(function (e) { failBox('m-quality', e); });
  }

  function worstEvent(bars) {
    var w = bars.slice().sort(function (a, b) { return a.m - b.m; })[0];
    return w ? w.ev + ': IoU ' + f3(w.m) : DASH;
  }

  function fact(h, v, t) {
    return '<div class="fact"><div class="fact-h">' + esc(h) + '</div><div class="fact-v">' + esc(v) + '</div><div class="fact-t">' + esc(t) + '</div></div>';
  }

  function barsSVG(bars) {
    var W = 560, H = 230, ml = 34, mb = 30, mt = 10, iw = W - ml - 10, ih = H - mb - mt;
    var gw = iw / Math.max(bars.length, 1), bw = Math.min(26, gw / 3.2);
    var s = '<svg class="bars" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet">';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      var y = mt + ih - v * ih;
      s += '<line class="g" x1="' + ml + '" x2="' + (W - 10) + '" y1="' + y + '" y2="' + y + '"/>' +
        '<text class="t" x="' + (ml - 6) + '" y="' + (y + 3) + '" text-anchor="end">' + String(v).replace('.', ',') + '</text>';
    });
    bars.forEach(function (b, i) {
      var cx = ml + gw * i + gw / 2;
      [['b', b.b, cx - bw - 2], ['m', b.m, cx + 2]].forEach(function (x) {
        var h = x[1] * ih;
        s += '<rect class="' + x[0] + '" x="' + x[2] + '" y="' + (mt + ih - h) + '" width="' + bw + '" height="' + h + '" rx="3"><title>' +
          (x[0] === 'b' ? 'Baseline' : 'Основной') + ' · ' + b.ev + ' · IoU ' + f3(x[1]) + '</title></rect>';
      });
      s += '<text class="' + (b.hold ? 'tb' : 't') + '" x="' + cx + '" y="' + (H - 10) + '" text-anchor="middle">' + esc(b.ev) + (b.hold ? '*' : '') + '</text>';
    });
    return s + '</svg>';
  }

  function renderProba() {
    Promise.all([report('metrics_compare.json'), report('metrics_compare_holdout.json'), report('experiments.json')]).then(function (r) {
      var t = r[0], h = r[1], e05 = (r[2].E05 || {}).results || {};
      var rel = t.calibration_on_this_part.reliability;
      $('m-proba').innerHTML =
        '<div class="grid-2"><section class="card"><div class="card-head"><h2>Надёжность вероятностей</h2><span class="muted small">test · калибратор эти данные не видел</span></div>' +
          reliabilitySVG(rel) +
          '<p class="note">Точка на диагонали — «сказали 0,7, и в 70 % таких пикселей действительно вода». Калибровка — изотоническая регрессия на validation, ' +
          'на равномерной, а не сбалансированной выборке: иначе вероятность выучила бы искусственную долю воды в половину кадра.</p></section>' +
        '<section class="card"><div class="card-head"><h2>Числа калибровки</h2></div><div class="kpi-row">' +
          '<div class="kpi"><div class="kpi-v">' + f3(t.calibration_on_this_part.ece) + '</div><div class="kpi-l">ECE · test</div></div>' +
          '<div class="kpi"><div class="kpi-v">' + f3(h.calibration_on_this_part.ece) + '</div><div class="kpi-l">ECE · Bolivia</div></div>' +
          '<div class="kpi"><div class="kpi-v">' + f3(t.calibration_on_this_part.brier) + '</div><div class="kpi-l">Brier · test</div></div></div>' +
          '<p class="note">ECE — средний разрыв между обещанной и наблюдаемой частотой; чем ближе к нулю, тем лучше. На новом событии разрыв больше — это честная цена переноса.</p></section></div>' +
        '<div class="grid-2"><section class="card"><div class="card-head"><h2>Полезна ли неопределённость</h2></div><div class="kpi-row">' +
          '<div class="kpi"><div class="kpi-v">' + f3(t.uncertainty_on_this_part.auc_uncertainty_vs_error) + '</div><div class="kpi-l">AUC «неопределённость → ошибка» · test</div></div>' +
          '<div class="kpi"><div class="kpi-v">' + f3(h.uncertainty_on_this_part.auc_uncertainty_vs_error) + '</div><div class="kpi-l">то же · Bolivia</div></div></div>' +
          '<div class="pair"><span>средняя неопределённость на ошибках</span><b>' + fmt(t.uncertainty_on_this_part.mean_uncertainty_on_errors, 2) + '</b></div>' +
          '<div class="pair"><span>на верных пикселях</span><b>' + fmt(t.uncertainty_on_this_part.mean_uncertainty_on_correct, 2) + '</b></div>' +
          '<p class="note">Там, где модель не уверена, она действительно чаще ошибается — поэтому неопределённость повышает приоритет проверки в стратегии C.</p></section>' +
        '<section class="card"><div class="card-head"><h2>Из чего собрана неопределённость</h2><span class="muted small">эксперимент E05</span></div>' +
          hbar('энтропия вероятности', e05['энтропия вероятности'] && e05['энтропия вероятности'].auc, false) +
          hbar('разброс трёх моделей', e05['разброс ансамбля'] && e05['разброс ансамбля'].auc, false) +
          hbar('0,7 × энтропия + 0,3 × разброс', e05['комбинация 0,7 / 0,3'] && e05['комбинация 0,7 / 0,3'].auc, true) +
          '<p class="note">AUC связи с ошибкой, validation. Выбрана комбинация: ансамбль добавляет сигнал там, где модели расходятся.</p></section></div>';
    }).catch(function (e) { failBox('m-proba', e); });
  }

  function hbar(label, v, strong) {
    var n = num(v);
    return '<div class="hbar' + (strong ? ' strong' : '') + '"><span class="hb-l">' + esc(label) + '</span><span class="hb-t"><i style="width:' +
      Math.round((n || 0) * 100) + '%"></i></span><span class="hb-v">' + f3(n) + '</span></div>';
  }

  function reliabilitySVG(bins) {
    var W = 300, H = 230, m = 30, iw = W - m - 10, ih = H - m - 10;
    var X = function (v) { return m + v * iw; }, Y = function (v) { return 10 + ih - v * ih; };
    var s = '<svg class="rel" viewBox="0 0 ' + W + ' ' + H + '">';
    [0, 0.5, 1].forEach(function (v) {
      s += '<line class="g" x1="' + X(0) + '" x2="' + X(1) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>' +
        '<text class="t" x="' + (m - 5) + '" y="' + (Y(v) + 3) + '" text-anchor="end">' + String(v).replace('.', ',') + '</text>' +
        '<text class="t" x="' + X(v) + '" y="' + (H - 12) + '" text-anchor="middle">' + String(v).replace('.', ',') + '</text>';
    });
    s += '<line class="diag" x1="' + X(0) + '" y1="' + Y(0) + '" x2="' + X(1) + '" y2="' + Y(1) + '"/>';
    var pts = bins.filter(function (b) { return b.n_pixels > 0; });
    s += '<polyline class="rl" points="' + pts.map(function (b) { return X(b.mean_predicted) + ',' + Y(b.observed_fraction); }).join(' ') + '"/>';
    pts.forEach(function (b) {
      s += '<circle class="rp" cx="' + X(b.mean_predicted) + '" cy="' + Y(b.observed_fraction) + '" r="4"><title>обещано ' +
        f3(b.mean_predicted) + ' · наблюдалось ' + f3(b.observed_fraction) + ' · пикселей ' + fmt(b.n_pixels, 0) + '</title></circle>';
    });
    s += '<text class="t" x="' + (W - 10) + '" y="' + (H - 1) + '" text-anchor="end">обещанная вероятность</text>';
    return s + '</svg>';
  }

  function renderData() {
    Promise.all([report('split_manifest.json'), report('experiments.json'), report('metrics_compare.json'), report('metrics_compare_holdout.json')]).then(function (r) {
      var sp = r[0], ex = r[1], t = r[2], h = r[3];
      var c = sp.counts, total = sp.total_chips;
      var seg = ['train', 'validation', 'test', 'holdout'].map(function (k) {
        return '<div class="sp sp-' + k + '" style="flex:' + c[k] + '"><b>' + c[k] + '</b><span>' + (k === 'holdout' ? 'Bolivia' : k) + '</span></div>';
      }).join('');
      var e02 = (ex.E02 || {}).results || {};
      var e02rows = Object.keys(e02).filter(function (k) { return e02[k] && e02[k].iou != null; }).map(function (k, i, arr) {
        return hbar(k, e02[k].iou, i === arr.length - 1);
      }).join('');
      $('m-data').innerHTML =
        '<div class="grid-2"><section class="card"><div class="card-head"><h2>Исходные данные</h2><span class="muted small">Sen1Floods11 v1.1 — официальный источник</span></div>' +
          '<ul class="checks">' +
          '<li>Единственный вход модели — Sentinel-1: VV и VH в дБ, nodata = NaN исключается</li>' +
          '<li>Слои одного чипа на одной сетке EPSG:4326 — перепроецирование не нужно</li>' +
          '<li>Цель — временное затопление: ручная метка «вода» минус постоянная вода JRC</li>' +
          '<li>JRC нужен только для разметки при обучении; на применении — один радарный снимок</li>' +
          '<li>Test не используется для настройки: порог, калибровка и конфигурация — только validation</li>' +
          '<li>Других источников нет: ни Sentinel-2, ни OSM, ни рельефа. OSM на карте — только фон</li></ul></section>' +
        '<section class="card"><div class="card-head"><h2>Разбиение без утечки</h2><span class="muted small">seed = ' + esc(sp.seed) + ' · события не пересекаются</span></div>' +
          '<div class="splitbar">' + seg + '</div>' +
          '<div class="small muted spev">train: ' + esc(sp.events.train.join(', ')) + '<br>validation: ' + esc(sp.events.validation.join(', ')) +
          '<br>test: ' + esc(sp.events.test.join(', ')) + '<br>перенос: ' + esc(sp.events.holdout.join(', ')) + ' — ' + total + ' чипов всего</div>' +
          '<p class="note">Авторский сплит набора отклонён: он раскладывает соседние куски одной сцены и в обучение, и в проверку. Списки чипов — в splits/.</p></section></div>' +
        '<div class="grid-2"><section class="card"><div class="card-head"><h2>Эксперимент: что даёт окрестность</h2><span class="muted small">E02 · IoU на validation</span></div>' +
          e02rows +
          '<p class="note">Урезанная выборка (70 / 30 чипов, 200 деревьев) и цель «вода вообще»: это сравнение вариантов между собой, а не метрики итоговой модели. ' +
          'Главный прирост даёт контекст вокруг пикселя, а не архитектура — поэтому нейросеть не понадобилась.</p></section>' +
        '<section class="card"><div class="card-head"><h2>Перенос на новое событие</h2></div><div class="kpi-row">' +
          '<div class="kpi"><div class="kpi-v">' + f3(h.totals.main_flood.iou) + '</div><div class="kpi-l">IoU на Bolivia (новое событие)</div></div>' +
          '<div class="kpi"><div class="kpi-v">' + f3(t.totals.main_flood.iou) + '</div><div class="kpi-l">IoU на test</div></div></div>' +
          '<div class="pair"><span>baseline на Bolivia</span><b>' + f3(h.totals.baseline_flood.iou) + '</b></div>' +
          '<div class="pair"><span>baseline на test</span><b>' + f3(t.totals.baseline_flood.iou) + '</b></div>' +
          '<p class="note">Основной метод на новом событии держится на уровне test; пороговый метод на Bolivia выигрывает с запасом. ' +
          'Модель на новом событии не дообучалась.</p></section></div>';
    }).catch(function (e) { failBox('m-data', e); });
  }

  // ── обзор: все события и все чипы сразу ──────────────────────────────────
  //
  // Как поля в «Фенологе»: на одной карте видно всё, что есть, а не только
  // открытый чип. Контуры событий — из официальных метаданных набора, чипы —
  // собранные комплекты. Клик по чипу открывает его в панели.

  var OV = { data: null, events: null, eventLabels: null, chips: null, chipMarks: null, wide: false };
  var OV_DETAIL_ZOOM = 10;   // ближе — показываем зоны и объекты, дальше — обзор

  function openedRun(fallback) { return V.currentRunId() || fallback || ''; }

  function openRun(id) {
    if (!id || id === V.currentRunId()) return;
    window.location.href = window.location.pathname + '?run=' + encodeURIComponent(id) + (window.location.hash || '');
  }

  function loadOverview() {
    V.getJSON('/api/overview').then(function (d) {
      OV.data = d;
      drawOverview();
    }).catch(function () { /* обзор — дополнение: без него чип работает как раньше */ });
  }

  function ringCenter(geom) {
    var b = L.geoJSON(geom).getBounds();
    return b.getCenter();
  }

  function drawOverview() {
    var S = V.S, d = OV.data;
    if (!S.map || !d) return;
    S.map.setMinZoom(1);
    ['events', 'eventLabels', 'chips', 'chipMarks'].forEach(function (k) { if (OV[k]) { S.map.removeLayer(OV[k]); OV[k] = null; } });

    var parts = {};
    (d.chips || []).forEach(function (c) { if (c.event_id) parts[c.event_id] = c.part; });
    var opened = openedRun(d.current);
    (d.chips || []).forEach(function (c) { c.current = c.run_id === opened; });
    var perEvent = {};
    (d.chips || []).forEach(function (c) { perEvent[c.event_id] = (perEvent[c.event_id] || 0) + 1; });

    OV.events = L.geoJSON(d.events, {
      style: function () { return { color: '#454986', weight: 1.4, opacity: 0.8, dashArray: '4 3', fillColor: '#454986', fillOpacity: 0.06 }; },
      onEachFeature: function (f, layer) {
        var p = f.properties || {};
        layer.bindTooltip('<b>' + esc(p.location) + '</b><br>съёмка S1 ' + esc(String(p.s1_date || '').replace(/\//g, '.')) +
          (parts[p.location] ? '<br>' + esc(parts[p.location]) : '') +
          '<br>собрано чипов: ' + (perEvent[p.location] || 0), { sticky: true, className: 'ovtip' });
      }
    });
    OV.eventLabels = L.layerGroup();
    (d.events.features || []).forEach(function (f) {
      var p = f.properties || {};
      if (perEvent[p.location]) return;  // событие уже подписано пилюлей своего чипа
      L.marker(ringCenter(f), {
        icon: L.divIcon({ className: 'evlab', html: '<span>' + esc(p.location) + '</span>', iconSize: [0, 0] }),
        interactive: false, keyboard: false
      }).addTo(OV.eventLabels);
    });

    OV.chips = L.layerGroup();
    OV.chipMarks = L.layerGroup();
    var stack = {};
    (d.chips || []).forEach(function (c) {
      var b = c.bounds;
      var k = stack[c.event_id] = (stack[c.event_id] || 0) + 1;
      if (!b || b.length !== 4) return;
      var ll = [[b[1], b[0]], [b[3], b[2]]];
      var rect = L.rectangle(ll, { color: c.current ? '#c8763c' : '#22254f', weight: 2, fillColor: c.current ? '#c8763c' : '#454986', fillOpacity: 0.35 });
      rect.bindTooltip(chipTip(c), { className: 'ovtip' });
      rect.on('click', function () { openRun(c.run_id); });
      rect.addTo(OV.chips);
      var mk = L.marker(L.latLngBounds(ll).getCenter(), {
        icon: L.divIcon({ className: 'runmark' + (c.current ? ' on' : ''), html: '<span style="margin-top:' + ((k - 1) * 24) + 'px">' + esc(c.chip_id) + '</span>', iconSize: [0, 0] }),
        keyboard: false, riseOnHover: true
      });
      mk.bindTooltip(chipTip(c), { className: 'ovtip', direction: 'top', offset: [0, -10] });
      mk.on('click', function () { openRun(c.run_id); });
      mk.addTo(OV.chipMarks);
    });

    OV.events.addTo(S.map);
    OV.chips.addTo(S.map);
    OV.events.bringToBack();
    S.map.off('zoomend', syncOverview);
    S.map.on('zoomend', syncOverview);
    syncOverview();
    var n = (d.chips || []).length, ne = (d.events.features || []).length;
  }

  function chipTip(c) {
    return '<b>' + esc(c.chip_id) + '</b>' + (c.current ? ' · открыт' : ' · нажмите, чтобы открыть') +
      '<br>' + esc(c.event_id) + (c.part ? ' · ' + esc(c.part) : '') + (c.observation_date ? ' · ' + esc(c.observation_date) : '') +
      (c.total_expected_loss_rub != null ? '<br>ожидаемый ущерб ' + mln(c.total_expected_loss_rub) + ' млн ₽' : '');
  }

  /** Далеко — обзор с подписями, близко — детали открытого чипа без лишнего. */
  function syncOverview() {
    var S = V.S;
    if (!S.map || !OV.data) return;
    var far = S.map.getZoom() < OV_DETAIL_ZOOM;
    [OV.eventLabels, OV.chipMarks].forEach(function (g) {
      if (!g) return;
      if (far && !S.map.hasLayer(g)) g.addTo(S.map);
      if (!far && S.map.hasLayer(g)) S.map.removeLayer(g);
    });
    if (OV.events) OV.events.setStyle({ fillOpacity: far ? 0.06 : 0, opacity: far ? 0.8 : 0.35 });
    // Номера объектов и подписи зон на обзоре мира висели бы россыпью поверх
    // континентов. Прячем их вдали и возвращаем вблизи — с учётом галочек.
    var assetsOn = $('chk-assets').checked, zonesOn = $('chk-zones').checked;
    [[S.assetsLayer, assetsOn], [zoneLabels, zonesOn]].forEach(function (x) {
      var g = x[0];
      if (!g) return;
      if (far && S.map.hasLayer(g)) S.map.removeLayer(g);
      if (!far && x[1] && !S.map.hasLayer(g)) g.addTo(S.map);
    });
    OV.wide = far;
  }

  // ── регион события: все его чипы сразу, как поля региона в «Фенологе» ────

  var REG = { data: null, layer: null, on: false, busy: false };

  function toggleOverview() {
    var S = V.S;
    if (!S.map || REG.busy) return;
    if (REG.on) { REG.on = false; V.fitChip(); syncRegionButton(); return; }
    if (REG.data && REG.data.event === S.summary.event_id) { showRegion(); return; }
    REG.busy = true;
    $('btn-overview').textContent = 'загрузка чипов…';
    V.getJSON('/api/event-chips?event=' + encodeURIComponent(S.summary.event_id)).then(function (d) {
      REG.data = d;
      drawRegion();
      showRegion();
    }).catch(function (e) {
      V.$('banner').className = 'banner warn';
      V.$('banner').textContent = 'Регион события не загружен: ' + e.message;
    }).then(function () { REG.busy = false; syncRegionButton(); });
  }

  function drawRegion() {
    var S = V.S, d = REG.data;
    if (REG.layer) S.map.removeLayer(REG.layer);
    REG.layer = L.layerGroup();
    var opened = openedRun(V.S.summary && V.S.summary.run_id);
    (d.chips || []).forEach(function (c) {
      var b = c.bounds;
      if (!b) return;
      c.current = !!c.run_id && c.run_id === opened;
      var kind = c.current ? 'cur' : c.run_id ? 'built' : 'raw';
      var style = {
        cur: { color: '#ffffff', weight: 2.5, fillColor: '#c8763c', fillOpacity: 0.55 },
        built: { color: '#ffffff', weight: 2, fillColor: '#454986', fillOpacity: 0.55 },
        raw: { color: '#ffffff', weight: 1.4, fillColor: '#ffffff', fillOpacity: 0.12 }
      }[kind];
      var r = L.rectangle([[b[1], b[0]], [b[3], b[2]]], style);
      r.bindTooltip('<b>' + esc(c.chip_id) + '</b><br>' + esc(d.event) + (d.part ? ' · ' + esc(d.part) : '') + '<br>' +
        (kind === 'cur' ? 'открыт сейчас' : kind === 'built' ? 'собран · нажмите, чтобы открыть' : 'не собран · нажмите, чтобы собрать'),
        { className: 'ovtip', sticky: true });
      r.on('click', function () {
        if (kind === 'built') openRun(c.run_id);
        else if (kind === 'raw') openBuild(c.chip_id);
      });
      r.addTo(REG.layer);
    });
    REG.layer.addTo(S.map);
    $('ml-region').classList.remove('hidden');
  }

  function showRegion() {
    var S = V.S, all = L.latLngBounds([]);
    REG.data.chips.forEach(function (c) { if (c.bounds) all.extend([[c.bounds[1], c.bounds[0]], [c.bounds[3], c.bounds[2]]]); });
    if (all.isValid()) S.map.fitBounds(all, { padding: [60, 60] });
    REG.on = true;
    syncRegionButton();
  }

  function syncRegionButton() {
    var d = REG.data, n = d ? d.chips.length : 0, built = d ? d.chips.filter(function (c) { return c.run_id; }).length : 0;
    $('btn-overview').textContent = REG.on ? 'К чипу' : 'Регион';
    $('btn-overview').title = d ? d.event + ': ' + n + ' ' + V.plural(n, 'чип', 'чипа', 'чипов') + ', собрано ' + built +
      (d.missing ? ' · без контура ' + d.missing : '') : 'Регион события: все его чипы сразу';
  }

  /** Несобранный чип — окно сборки Димы с уже подставленным чипом. */
  function openBuild(chip) {
    var open = $('btn-build-open');
    if (open) open.click();
    setTimeout(function () {
      var input = $('build-chip');
      if (!input) return;
      input.value = chip;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    }, 300);
  }

  // ── общее ────────────────────────────────────────────────────────────────

  /** Кнопки скачивания отдельных файлов — всегда про открытый комплект. */
  function setDownloads() {
    Array.prototype.forEach.call(document.querySelectorAll('a.dl[data-file]'), function (a) {
      a.href = V.withRun('/api/files/' + a.getAttribute('data-file'));
    });
  }

  function onUpdate(e) {
    V = window.VODOPOL;
    var why = e.detail;
    if (why === 'error') return;
    if (why === 'boot') {
      V.S._zlabels = zoneIndex();
      if (V.S.map && !V.S._zoomMoved) { V.S.map.zoomControl.setPosition('bottomright'); V.S._zoomMoved = true; }
      loadOverview();
      bindMapClicks();
      if (V.S.map && !V.S._scale) {
        V.S._scale = L.control.scale({ imperial: false, position: 'bottomleft', maxWidth: 160 }).addTo(V.S.map);
      }
      RANKS = { run: null, rows: null };
      renderMapHead();
      renderKPI();
    }
    if (why === 'budget-input') {
      $('budget-num-plan').textContent = fmt(V.S.budget, 0);
      return;
    }
    renderBudgetRows();
    renderFirst();
    renderZoneLabels();
    if (why !== 'boot') renderKPI();
    renderRoute();
    if (why === 'boot') placeShared();
  }

  document.addEventListener('vp:update', onUpdate);
  window.addEventListener('hashchange', route);
  document.addEventListener('DOMContentLoaded', function () {
    V = window.VODOPOL;
    route();
    // Колонки меняют ширину уже после подгонки кадра (карточки стратегий, таблицы
    // дорисовываются позже). Следим за размером контейнера и подгоняем снимок заново.
    if (window.ResizeObserver) {
      var fitTimer = null;
      new ResizeObserver(function () {
        if (!V || !V.S.map) return;
        clearTimeout(fitTimer);
        fitTimer = setTimeout(function () { V.S.map.invalidateSize(); if (!OV.wide) V.fitChip(); }, 60);
      }).observe($('mapwrap'));
    }
    Array.prototype.forEach.call(document.querySelectorAll('#basemap-switch button'), function (b) {
      b.addEventListener('click', function () { setBasemap(b.getAttribute('data-base')); });
    });
    var ovb = $('btn-overview');
    if (ovb) ovb.addEventListener('click', toggleOverview);
    // Подписи зон появляются вместе со слоем зон.
    var chk = $('chk-zones');
    if (chk) chk.addEventListener('change', function () {
      if (!zoneLabels || !V.S.map) return;
      if (chk.checked) zoneLabels.addTo(V.S.map); else V.S.map.removeLayer(zoneLabels);
    });
  });
})();
