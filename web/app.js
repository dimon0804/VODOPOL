/* Водополь — панель оператора.
 *
 * Страница ничего не считает: все числа приходят из /api/*, то есть из
 * src.runtime.RunContext, то есть из файлов сданного комплекта. Здесь только
 * показ, подписи происхождения чисел и перерисовка при смене бюджета.
 */
(function () {
  'use strict';

  // ── словари контракта (src/contracts.py, модуль заморожен) ───────────────

  var ASSET_TITLES = {
    warehouse: 'склад',
    substation: 'электроподстанция',
    road_unit: 'участок дороги',
    facility: 'производственный объект',
    pumping_station: 'насосная станция',
    clinic: 'клиника',
    school: 'школа',
    workshop: 'мастерская',
    telecom_node: 'узел связи',
    water_intake: 'водозабор'
  };

  var UNC_WORD = {
    baseline: 'baseline',
    scenario: 'сценарий',
    measured: 'измерено',
    not_estimated: 'не оценивалось'
  };
  var UNC_HINT = {
    baseline: 'исходная неопределённость до закупки данных',
    scenario: 'эффект предполагаемый, формула раскрыта, фактом не является',
    measured: 'проверено по фактически поступившим новым наблюдениям',
    not_estimated: 'числовое поле остаётся пустым'
  };
  var UNC_MARK = {
    baseline: 'model',
    scenario: 'scenario',
    measured: 'measured',
    not_estimated: 'none'
  };

  var ASSET_STATUS_WORD = { ok: 'оценка есть', partial: 'частично', no_data: 'нет данных' };

  // Палитры повторяют src/runtime.py: легенда обязана совпадать с растром.
  var PROB_STOPS = ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'];
  var UNC_STOPS = ['#fff7ec', '#fdd49e', '#fd8d3c', '#d94801', '#7f2704'];
  var MASK_COLOR = '#08306b';

  var STRATEGY_TITLE = {
    A: 'A — только открытые данные, платных заказов нет',
    B: 'B — широкая закупка по правилу, объявленному до расчёта цены',
    C: 'C — выборочная закупка по приоритетам в пределах бюджета'
  };

  var DASH = '—';

  // ── состояние ────────────────────────────────────────────────────────────

  var S = {
    health: null,
    summary: null,
    assets: null,
    candidates: null,
    sensitivity: null,
    strategies: null,
    strategy: 'C',
    budget: null,
    declaredBudget: null,
    layer: 'probability',
    opacity: 0.8,
    s1ready: false,
    map: null,
    overlays: {},
    zonesLayer: null,
    assetsLayer: null,
    selectedZone: null,
    budgetTimer: null,
    reqSeq: 0
  };

  // ── мелочи ───────────────────────────────────────────────────────────────

  function $(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** Пустое значение остаётся пустым: у partial и no_data ноль был бы ложью. */
  function toNum(value) {
    if (value === null || value === undefined || value === '') return null;
    var n = typeof value === 'number' ? value : Number(String(value).replace(',', '.'));
    return isFinite(n) ? n : null;
  }

  function toBool(value) {
    if (typeof value === 'boolean') return value;
    if (value === null || value === undefined || value === '') return null;
    var s = String(value).trim().toLowerCase();
    if (s === 'true' || s === '1' || s === 'да') return true;
    if (s === 'false' || s === '0' || s === 'нет') return false;
    return null;
  }

  function fmt(value, digits) {
    var n = toNum(value);
    if (n === null) return DASH;
    return n.toLocaleString('ru-RU', {
      minimumFractionDigits: digits === undefined ? 0 : digits,
      maximumFractionDigits: digits === undefined ? 0 : digits
    });
  }

  function money(value) {
    var n = toNum(value);
    if (n === null) return DASH;
    return n.toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function pct(value) {
    var n = toNum(value);
    if (n === null) return DASH;
    return (n * 100).toLocaleString('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' %';
  }

  function mark(kind, text, hint) {
    return '<span class="mark ' + kind + '"' + (hint ? ' title="' + esc(hint) + '"' : '') + '>' + esc(text) + '</span>';
  }

  function emptyCell(reason) {
    return '<td class="empty" title="' + esc(reason || 'оценка отсутствует; ноль сюда не подставляется') + '">' + DASH + '</td>';
  }

  function hex2rgb(hex) {
    return [parseInt(hex.substr(1, 2), 16), parseInt(hex.substr(3, 2), 16), parseInt(hex.substr(5, 2), 16)];
  }

  function mix(a, b, t) {
    var x = hex2rgb(a), y = hex2rgb(b);
    return 'rgb(' + Math.round(x[0] + (y[0] - x[0]) * t) + ',' +
      Math.round(x[1] + (y[1] - x[1]) * t) + ',' +
      Math.round(x[2] + (y[2] - x[2]) * t) + ')';
  }

  function rankColor(rank) {
    if (rank === null || rank === undefined) return '#6c7686';
    var t = Math.min(1, Math.max(0, (Number(rank) - 1) / 9));
    return mix('#e03131', '#ffd8a8', t);
  }

  function rankRadius(rank) {
    if (rank === null || rank === undefined) return 5;
    return 12 - (Number(rank) - 1) * 0.72;
  }

  function banner(text, kind) {
    var node = $('banner');
    if (!text) { node.classList.add('hidden'); return; }
    node.className = 'banner' + (kind === 'warn' ? ' warn' : '');
    node.textContent = text;
  }

  // ── сеть ─────────────────────────────────────────────────────────────────

  function getJSON(path) {
    return fetch(path, { headers: { Accept: 'application/json' } }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error(body.error || body.detail || ('HTTP ' + r.status + ' ' + path));
        return body;
      });
    });
  }

  function fetchRaster(kind) {
    return fetch('/api/raster/' + kind + '.png').then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.error || body.detail || ('слой ' + kind + ' недоступен'));
        });
      }
      var raw = r.headers.get('X-Bounds');
      var bounds = null;
      try { bounds = raw ? JSON.parse(raw) : null; } catch (e) { bounds = null; }
      return r.blob().then(function (blob) {
        return { url: URL.createObjectURL(blob), bounds: bounds };
      });
    });
  }

  // ── карта ────────────────────────────────────────────────────────────────

  function leafletBounds(wsen) {
    if (!wsen || wsen.length !== 4) return null;
    return [[wsen[1], wsen[0]], [wsen[3], wsen[2]]];
  }

  function initMap() {
    S.map = L.map('map', { preferCanvas: true, zoomControl: true, minZoom: 3 });
    // Географический контекст. Тайлы приглушены фильтром в styles.css; если сети
    // нет, подложка просто не появится — слои запуска от неё не зависят.
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap · снимок Sentinel-1, Sen1Floods11'
    }).addTo(S.map);

    var b = leafletBounds(S.summary && S.summary.bounds);
    if (b) S.map.fitBounds(b, { padding: [8, 8] });
    else S.map.setView([0, 0], 3);
  }

  /**
   * Яркость подложки S1 зависит от того, читается ли поверх неё тематический слой.
   * На тёмных сценах вероятность видно и поверх полной подложки, а на ярких, вроде
   * боливийской, спекл перебивает слой с низкими значениями. Приглушаем подложку,
   * когда поверх неё что-то показывают, и возвращаем полную, когда слой выключен.
   */
  function s1Opacity() {
    return S.layer === 'none' ? 1 : 0.45;
  }

  /** Растровые слои подгружаются по требованию и переиспользуются. */
  function ensureOverlay(kind) {
    if (S.overlays[kind]) return Promise.resolve(S.overlays[kind]);
    return fetchRaster(kind).then(function (res) {
      var b = leafletBounds(res.bounds) || leafletBounds(S.summary && S.summary.bounds);
      if (!b) throw new Error('границы слоя ' + kind + ' неизвестны');
      var layer = L.imageOverlay(res.url, b, {
        opacity: kind === 's1' ? s1Opacity() : S.opacity,
        className: 'raster-' + kind,
        interactive: false
      });
      S.overlays[kind] = layer;
      if (!S.summary.bounds) S.map.fitBounds(b, { padding: [8, 8] });
      return layer;
    });
  }

  function syncLayers() {
    // подложка
    if ($('chk-s1').checked) {
      ensureOverlay('s1').then(function (layer) {
        S.s1ready = true;
        $('s1-note').textContent = '';
        if (!S.map.hasLayer(layer)) layer.addTo(S.map);
        layer.setOpacity(s1Opacity());
        layer.setZIndex(200);
        reorder();
      }).catch(function (err) {
        S.s1ready = false;
        $('chk-s1').checked = false;
        $('chk-s1').disabled = true;
        $('s1-note').textContent = 'Подложка S1 недоступна, остальные слои запуска не затронуты.';
        $('s1-note').title = err.message;
      });
    } else if (S.overlays.s1 && S.map.hasLayer(S.overlays.s1)) {
      S.map.removeLayer(S.overlays.s1);
    }

    // тематический слой — ровно один
    ['probability', 'mask', 'uncertainty'].forEach(function (kind) {
      var layer = S.overlays[kind];
      if (kind !== S.layer && layer && S.map.hasLayer(layer)) S.map.removeLayer(layer);
    });
    if (S.layer !== 'none') {
      ensureOverlay(S.layer).then(function (layer) {
        layer.setOpacity(S.opacity);
        if (!S.map.hasLayer(layer)) layer.addTo(S.map);
        layer.setZIndex(300);
        reorder();
      }).catch(function (err) {
        banner('Слой «' + S.layer + '» не отдан сервисом: ' + err.message, 'warn');
      });
    }
    renderLegend();
  }

  function reorder() {
    if (S.zonesLayer) S.zonesLayer.bringToFront();
    if (S.assetsLayer) S.assetsLayer.bringToFront();
  }

  // ── зоны ─────────────────────────────────────────────────────────────────

  function selectedSet() {
    var plans = (S.strategies && S.strategies.plans) || {};
    var list = plans[S.strategy] || [];
    var set = {};
    list.forEach(function (id) { set[id] = true; });
    return set;
  }

  function positionFor(candidateId) {
    var rows = (S.strategies && S.strategies.positions) || [];
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].candidate_id === candidateId && rows[i].strategy === S.strategy) return rows[i];
    }
    return null;
  }

  function zoneStyle(feature) {
    var id = feature.properties.candidate_id;
    var chosen = !!selectedSet()[id];
    var picked = S.selectedZone === id;
    if (chosen) {
      return {
        color: picked ? '#e0f2fe' : '#38bdf8',
        weight: picked ? 2.4 : 1.6,
        opacity: 0.95,
        fillColor: '#38bdf8',
        fillOpacity: 0.22
      };
    }
    return {
      color: picked ? '#e0f2fe' : '#93a1b3',
      weight: picked ? 2 : 0.8,
      opacity: 0.55,
      fillColor: '#93a1b3',
      fillOpacity: 0.04,
      dashArray: '3 3'
    };
  }

  function zoneInfoHTML(props, position, chosen) {
    var rows = [];
    var cost = position ? position.cost_rub : null;
    rows.push(['цена заказа, ₽', money(cost) + ' ' + mark('scenario', 'сценарная ставка', 'БРЕ 788,64 ₽/км² — учебная ставка из постановки кейса, подтверждённой действующей не является')]);
    rows.push(['площадь, км²', fmt(props.area_km2, 3) + ' ' + mark('model', 'EPSG:6933')]);
    rows.push(['ставка БРЕ, ₽/км²', fmt(props.base_rate_rub_km2, 2) + ' ' + mark('scenario', props.base_rate_status === 'scenario' ? 'сценарная' : String(props.base_rate_status || ''))]);
    rows.push(['сенсор', esc(props.sensor_type) + ', ' + fmt(props.resolution_m, 0) + ' м']);
    rows.push(['съёмка', esc(props.observation_at || DASH) + ' ' + mark('scenario', 'сценарная дата')]);
    rows.push(['доступно к', esc(props.available_at || DASH) + ' ' + mark('scenario', 'сценарная дата')]);
    rows.push(['обработка / использование', esc(props.processing_level) + ' / ' + esc(props.usage_type)]);
    rows.push(['гарантированный выкуп', props.guaranteed_purchase ? 'да' : 'нет']);
    rows.push(['роль данных', props.data_role === 'event_observation' ? 'наблюдение события' : 'контекст']);
    if (props.covered_asset_count !== undefined) {
      rows.push(['покрывает объектов', fmt(props.covered_asset_count, 0) +
        (props.covered_asset_ids && props.covered_asset_ids.length ? ' · ' + esc(props.covered_asset_ids.join(', ')) : '')]);
    }

    var html = '<dl class="kvgrid">';
    rows.forEach(function (r) { html += '<dt>' + r[0] + '</dt><dd>' + r[1] + '</dd>'; });
    html += '</dl>';

    if (position) {
      html += '<div class="coefs">РП = Б × К × О × П × Т × Р<br>' +
        'Б <b>' + fmt(position.base_rate_rub_km2, 2) + '</b> · ' +
        'К <b>' + fmt(position.area_km2, 3) + '</b> км² · ' +
        'О <b>' + fmt(position.processing_coef, 1) + '</b> · ' +
        'П <b>' + fmt(position.usage_coef, 1) + '</b> · ' +
        'Т <b>' + fmt(position.freshness_coef, 1) + '</b> · ' +
        'Р <b>' + fmt(position.discount_coef, 6) + '</b><br>' +
        'цена км² <b>' + money(position.unit_price_rub_km2) + '</b> ₽ · ' +
        'корзина скидки <b>' + esc(position.discount_group_id) + '</b>, ' +
        fmt(position.group_area_km2, 3) + ' км²</div>';
      html += '<p class="pnote">' + esc(position.legal_edition || '') + '</p>';
    } else {
      html += '<p class="pnote">' + (chosen
        ? 'Позиция в плане есть, но строка расчёта цены не найдена.'
        : 'В стратегии ' + esc(S.strategy) + ' эта зона не заказывается — цена показана бы только при включении в корзину: скидка Р пересчитывается для корзины целиком.') + '</p>';
    }
    return html;
  }

  function renderZones() {
    if (!S.candidates) return;
    if (S.zonesLayer) { S.map.removeLayer(S.zonesLayer); S.zonesLayer = null; }
    S.zonesLayer = L.geoJSON(S.candidates, {
      style: zoneStyle,
      onEachFeature: function (feature, layer) {
        layer.on('click', function () {
          S.selectedZone = feature.properties.candidate_id;
          restyleZones();
          showZoneCard(feature);
          var chosen = !!selectedSet()[feature.properties.candidate_id];
          layer.bindPopup('<div class="popup"><h4>' + esc(feature.properties.candidate_id) + '</h4>' +
            '<div class="sub">' + (chosen ? 'заказывается в стратегии ' + esc(S.strategy) : 'в стратегии ' + esc(S.strategy) + ' не заказывается') + '</div>' +
            zoneInfoHTML(feature.properties, positionFor(feature.properties.candidate_id), chosen) + '</div>',
            { maxWidth: 380 }).openPopup();
        });
      }
    });
    if ($('chk-zones').checked) S.zonesLayer.addTo(S.map);
    reorder();
  }

  function restyleZones() {
    if (!S.zonesLayer) return;
    S.zonesLayer.eachLayer(function (layer) { layer.setStyle(zoneStyle(layer.feature)); });
  }

  function showZoneCard(feature) {
    var props = feature.properties;
    var chosen = !!selectedSet()[props.candidate_id];
    $('zone-hint').innerHTML = chosen
      ? mark('model', 'заказывается в ' + S.strategy)
      : mark('none', 'вне плана ' + S.strategy);
    $('zone-body').innerHTML =
      '<div class="zone-title">' + esc(props.candidate_id) + '</div>' +
      zoneInfoHTML(props, positionFor(props.candidate_id), chosen);
  }

  // ── объекты ──────────────────────────────────────────────────────────────

  function coveredSet() {
    var row = comparisonRow(S.strategy);
    var raw = row ? row.covered_asset_ids : null;
    var set = {};
    if (!raw) return set;
    String(raw).split(';').forEach(function (id) { if (id) set[id.trim()] = true; });
    return set;
  }

  function assetPopup(props) {
    var status = props.status || 'no_data';
    var ok = status === 'ok';
    var covered = !!coveredSet()[props.asset_id];
    var html = '<div class="popup"><h4>' + esc(props.asset_id) + ' · ' +
      esc(ASSET_TITLES[props.asset_class] || props.asset_class) + '</h4>' +
      '<div class="sub">чип ' + esc(props.chip_id || '') + ' · ранг по ущербу ' +
      (props.rank == null ? DASH : props.rank) + '</div><dl>';
    html += '<dt>p затопления</dt><dd>' + (ok ? fmt(props.p_flood, 4) + ' ' + mark('model', 'расчёт модели') : DASH + ' ' + mark('none', 'нет оценки')) + '</dd>';
    html += '<dt>ожидаемый ущерб, ₽</dt><dd>' + (ok ? money(props.expected_loss_rub) + ' ' + mark('model', 'p × V × q') : DASH + ' ' + mark('none', 'нет оценки')) + '</dd>';
    html += '<dt>неопределённость</dt><dd>' + (ok ? fmt(props.uncertainty, 3) + ' ' + mark('model', 'расчёт модели') : DASH + ' ' + mark('none', 'нет оценки')) + '</dd>';
    html += '<dt>статус объекта</dt><dd>' + esc(ASSET_STATUS_WORD[status] || status) + '</dd>';
    html += '<dt>стоимость V, ₽</dt><dd>' + fmt(props.asset_value_rub, 0) + ' ' + mark('case', 'условие кейса') + '</dd>';
    html += '<dt>уязвимость q</dt><dd>' + fmt(props.vulnerability_coef, 2) + ' ' + mark('case', 'условие кейса') + '</dd>';
    html += '<dt>покрытие в ' + esc(S.strategy) + '</dt><dd>' + (covered ? 'да' : 'нет') + '</dd>';
    html += '</dl>';
    if (!ok) {
      html += '<p class="pnote">Статус «' + esc(ASSET_STATUS_WORD[status] || status) +
        '»: p, ущерб и ранг пустые. Это не ноль — ущерб неизвестен.</p>';
    }
    html += '</div>';
    return html;
  }

  function renderAssets() {
    if (!S.assets) return;
    if (S.assetsLayer) { S.map.removeLayer(S.assetsLayer); S.assetsLayer = null; }
    S.assetsLayer = L.geoJSON(S.assets, {
      pointToLayer: function (feature, latlng) {
        var p = feature.properties;
        var ok = p.status === 'ok';
        return L.circleMarker(latlng, {
          radius: ok ? rankRadius(p.rank) : 5,
          color: ok ? '#0b0e12' : '#c8d0dc',
          weight: ok ? 1.2 : 1.4,
          dashArray: ok ? null : '2 2',
          fillColor: ok ? rankColor(p.rank) : '#39414e',
          fillOpacity: ok ? 0.92 : 0.5
        });
      },
      onEachFeature: function (feature, layer) {
        layer.bindTooltip(feature.properties.asset_id + ' · ' +
          (ASSET_TITLES[feature.properties.asset_class] || feature.properties.asset_class),
          { direction: 'top', offset: [0, -6] });
        layer.on('click', function () {
          layer.bindPopup(assetPopup(feature.properties), { maxWidth: 340 }).openPopup();
        });
      }
    });
    if ($('chk-assets').checked) S.assetsLayer.addTo(S.map);
    reorder();
    renderAssetTable();
  }

  function renderAssetTable() {
    var features = (S.assets && S.assets.features) || [];
    var covered = coveredSet();
    var rows = features.slice().sort(function (a, b) {
      var x = toNum(a.properties.expected_loss_rub);
      var y = toNum(b.properties.expected_loss_rub);
      if (x === null && y === null) return 0;
      if (x === null) return 1;
      if (y === null) return -1;
      return y - x;
    });

    var total = 0, counted = 0, unknown = 0;
    features.forEach(function (f) {
      var el = toNum(f.properties.expected_loss_rub);
      if (el === null) unknown++; else { total += el; counted++; }
    });

    var html = '';
    rows.forEach(function (f) {
      var p = f.properties;
      var ok = p.status === 'ok';
      var statusMark = ok
        ? mark('model', 'оценка', 'оценка есть: p, ущерб и ранг заполнены')
        : mark('bad', ASSET_STATUS_WORD[p.status] || p.status,
            'Поля p, ущерба и ранга пустые по контракту: ущерб не равен нулю, он неизвестен');
      html += '<tr class="' + (ok ? '' : 'dim') + '" data-asset="' + esc(p.asset_id) + '">';
      html += ok ? '<td class="r">' + esc(p.rank == null ? DASH : p.rank) + '</td>' : emptyCell('ранг не присваивается без оценки');
      html += '<td><span class="dot" style="background:' + (ok ? rankColor(p.rank) : '#39414e') + '"></span>' +
        esc(ASSET_TITLES[p.asset_class] || p.asset_class) + ' <span class="sub">' + esc(p.asset_id) + '</span></td>';
      html += ok ? '<td class="r">' + fmt(p.p_flood, 4) + '</td>' : emptyCell();
      html += ok ? '<td class="r">' + money(p.expected_loss_rub) + '</td>' : emptyCell();
      html += ok ? '<td class="r">' + fmt(p.uncertainty, 3) + '</td>' : emptyCell();
      html += '<td>' + statusMark + '</td>';
      html += '<td>' + (covered[p.asset_id] ? mark('model', 'да') : '<span class="sub">нет</span>') + '</td>';
      html += '</tr>';
    });
    $('tbl-assets').querySelector('tbody').innerHTML = html;

    $('assets-total').innerHTML = 'сумма ' + money(total) + ' ₽ по ' + counted + ' из ' + features.length +
      (unknown ? ' · без оценки: ' + unknown : '');

    Array.prototype.forEach.call($('tbl-assets').querySelectorAll('tr[data-asset]'), function (tr) {
      tr.addEventListener('click', function () {
        var id = tr.getAttribute('data-asset');
        if (!S.assetsLayer) return;
        S.assetsLayer.eachLayer(function (layer) {
          if (layer.feature && layer.feature.properties.asset_id === id) {
            S.map.setView(layer.getLatLng(), Math.max(S.map.getZoom(), 14));
            layer.bindPopup(assetPopup(layer.feature.properties), { maxWidth: 340 }).openPopup();
          }
        });
      });
    });
  }

  // ── стратегии ────────────────────────────────────────────────────────────

  function comparisonRow(strategy) {
    var rows = (S.strategies && S.strategies.comparison) || [];
    for (var i = 0; i < rows.length; i++) {
      if (String(rows[i].strategy) === strategy) return rows[i];
    }
    return null;
  }

  /** Таблица развёрнута: показатели — строки, стратегии — столбцы.
   *  Так все десять показателей видны сразу и сравниваются по вертикали. */
  function renderStrategies() {
    var plans = (S.strategies && S.strategies.plans) || {};
    var codes = ['A', 'B', 'C'];
    var rowsByCode = {};
    codes.forEach(function (code) { rowsByCode[code] = comparisonRow(code) || {}; });

    function cells(render) {
      return codes.map(function (code) {
        return '<td class="r strat ' + (code === S.strategy ? 'col-active' : '') + '">' +
          render(rowsByCode[code], code) + '</td>';
      }).join('');
    }

    var metrics = [
      ['зон в заказе', '', function (row, code) { return fmt((plans[code] || []).length, 0); }],
      ['стоимость данных, ₽', mark('scenario', 'сценарная ставка',
        'БРЕ 788,64 ₽/км² — учебная ставка из постановки кейса'),
        function (row) { return money(row.data_cost_rub); }],
      ['прочие затраты, ₽', '', function (row) { return money(row.other_cost_rub); }],
      ['полная стоимость решения, ₽', mark('scenario', 'сценарная ставка'),
        function (row) { return money(row.decision_cost_rub); }],
      ['объявленный бюджет, ₽', mark('case', 'условие кейса'),
        function (row) { return money(row.budget_rub); }],
      ['уложились в бюджет', '', function (row) {
        var feasible = toBool(row.budget_feasible);
        if (feasible === null) return mark('none', 'не определено');
        if (feasible) return mark('measured', 'да');
        return mark('bad', 'контрфактическая, сверх бюджета',
          'Стоимость выше объявленного бюджета: разбирается как контрфактический сценарий, к исполнению не предлагается');
      }],
      ['уникально покрытый ущерб, ₽', mark('model', 'расчёт модели'),
        function (row) { return money(row.covered_expected_loss_rub); }],
      ['доля покрытия портфеля', mark('model', 'расчёт модели'),
        function (row) { return pct(row.coverage_share); }],
      ['остаточная неопределённость', '', function (row) {
        var value = toNum(row.residual_uncertainty);
        return value === null
          ? '<span class="empty" title="не оценивалось: поле остаётся пустым">' + DASH + '</span>'
          : fmt(value, 3);
      }],
      ['статус неопределённости', '', function (row) {
        var unc = String(row.uncertainty_status || 'not_estimated');
        return mark(UNC_MARK[unc] || 'none', UNC_WORD[unc] || unc, UNC_HINT[unc] || '');
      }]
    ];

    var html = '<thead><tr><th>показатель</th>' + codes.map(function (code) {
      return '<th class="r strat ' + (code === S.strategy ? 'col-active' : '') +
        '" data-strategy="' + code + '" title="' + esc(STRATEGY_TITLE[code] || '') + '">' + code + '</th>';
    }).join('') + '</tr></thead><tbody>';

    metrics.forEach(function (metric) {
      html += '<tr><th class="rowhead">' + metric[0] + (metric[1] ? ' ' + metric[1] : '') + '</th>' +
        cells(metric[2]) + '</tr>';
    });
    html += '</tbody>';

    $('tbl-strategies').innerHTML = html;

    Array.prototype.forEach.call($('tbl-strategies').querySelectorAll('th[data-strategy]'), function (th) {
      th.addEventListener('click', function () { setStrategy(th.getAttribute('data-strategy')); });
    });

    var row = comparisonRow(S.strategy);
    var basis = row ? row.uncertainty_basis : '';
    var noteParts = [(STRATEGY_TITLE[S.strategy] || '') + '.'];
    var b = comparisonRow('B');
    if (b && toBool(b.budget_feasible) === false) {
      noteParts.push('Стратегия B при текущем бюджете — контрфактическая: её стоимость ' +
        money(b.decision_cost_rub) + ' ₽ выше бюджета ' + money(b.budget_rub) +
        ' ₽. Она показана для сравнения, а не как исполнимый план.');
    }
    if (basis) noteParts.push('Остаточная неопределённость: ' + basis + '.');
    noteParts.push('Снижение неопределённости у B и C имеет статус «сценарий»: новых наблюдений ещё нет, фактом это не является.');
    $('strategy-note').innerHTML = noteParts.map(esc).join(' ');

    Array.prototype.forEach.call($('strategy-switch').querySelectorAll('button'), function (btn) {
      btn.classList.toggle('active', btn.getAttribute('data-strategy') === S.strategy);
    });
  }

  function setStrategy(code) {
    if (!code || code === S.strategy) return;
    S.strategy = code;
    renderStrategies();
    restyleZones();
    renderAssetTable();
    renderSensitivity();
    if (S.selectedZone && S.candidates) {
      var f = S.candidates.features.filter(function (x) { return x.properties.candidate_id === S.selectedZone; })[0];
      if (f) showZoneCard(f);
    }
  }

  // ── чувствительность ─────────────────────────────────────────────────────

  function changedInputs(raw) {
    try {
      var obj = JSON.parse(raw);
      var parts = [];
      Object.keys(obj).forEach(function (key) {
        if (key === 'rule') return;
        parts.push(key + ' = ' + obj[key]);
      });
      return parts.join(', ') || String(raw);
    } catch (e) {
      return String(raw || '');
    }
  }

  function renderSensitivity() {
    var rows = S.sensitivity || [];
    var html = '';
    rows.forEach(function (row) {
      var code = String(row.strategy || '');
      var status = String(row.result_status || '');
      var counterfactual = status.indexOf('контрфакт') >= 0;
      html += '<tr class="' + (code === S.strategy ? 'active' : '') + '" title="' + esc(row.interpretation || '') + '">';
      html += '<td>' + esc(row.scenario_id) + '</td>';
      html += '<td><b>' + esc(code) + '</b></td>';
      html += '<td title="' + esc(row.changed_inputs_json || '') + '">' + esc(changedInputs(row.changed_inputs_json)) + '</td>';
      html += '<td class="r">' + money(row.decision_cost_rub) + '</td>';
      html += '<td class="r">' + money(row.covered_expected_loss_rub) + '</td>';
      html += '<td>' + (counterfactual
        ? mark('bad', 'контрфакт.', status + ': стоимость выше бюджета, к исполнению не предлагается')
        : mark('measured', 'в бюджете', status)) + '</td>';
      html += '</tr>';
    });
    $('tbl-sensitivity').querySelector('tbody').innerHTML = html;
    $('sens-count').textContent = rows.length + ' строк';
  }

  // ── легенда ──────────────────────────────────────────────────────────────

  function gradient(stops) {
    return 'linear-gradient(to right, ' + stops.join(', ') + ')';
  }

  function renderLegend() {
    var threshold = toNum(S.summary && S.summary.threshold);
    var html = '';
    if (S.layer === 'probability') {
      html += '<h3>Вероятность затопления</h3>';
      html += '<div class="bar" style="background:' + gradient(PROB_STOPS) + '"></div>';
      html += '<div class="ticks"><span>0,00</span><span>0,25</span><span>0,50</span><span>0,75</span><span>1,00</span></div>';
      html += '<div class="lnote">Прозрачность растёт вместе со значением. Порог бинаризации ' +
        (threshold === null ? DASH : fmt(threshold, 2)) + ' подобран на validation ' + mark('model', 'расчёт модели') + '</div>';
    } else if (S.layer === 'uncertainty') {
      html += '<h3>Неопределённость</h3>';
      html += '<div class="bar" style="background:' + gradient(UNC_STOPS) + '"></div>';
      html += '<div class="ticks"><span>0,00</span><span>0,25</span><span>0,50</span><span>0,75</span><span>1,00</span></div>';
      html += '<div class="lnote">Безразмерная величина 0…1: энтропия вероятности и разброс ансамбля ' + mark('model', 'расчёт модели') + '</div>';
    } else if (S.layer === 'mask') {
      html += '<h3>Бинарная маска</h3>';
      html += '<div class="rowitem"><span class="swatch" style="background:' + MASK_COLOR + '"></span>' +
        'временное затопление, p ≥ ' + (threshold === null ? DASH : fmt(threshold, 2)) + '</div>';
      html += '<div class="lnote">Целевой класс — временное затопление, постоянная вода исключена на стороне разметки</div>';
    } else {
      html += '<h3>Слои</h3><div class="lnote">Тематический слой выключен</div>';
    }

    html += '<div class="rowitem" style="margin-top:7px"><span class="swatch" style="background:rgba(56,189,248,.35);border-color:#38bdf8"></span>зоны стратегии ' + esc(S.strategy) + '</div>';
    html += '<div class="rowitem"><span class="swatch" style="background:transparent;border-color:#93a1b3;border-style:dashed"></span>каталог зон, не заказаны</div>';
    html += '<div class="ranks">';
    [1, 3, 5, 7, 10].forEach(function (rank) {
      var d = Math.round(rankRadius(rank) * 2);
      html += '<i style="width:' + d + 'px;height:' + d + 'px;background:' + rankColor(rank) + '"></i>';
    });
    html += '</div><div class="lnote">размер и цвет точки — ранг ожидаемого ущерба, слева 1 (наибольший), справа 10. ' +
      'Серая пунктирная точка — объект без оценки</div>';

    $('legend').innerHTML = html;
  }

  // ── шапка и паспорт ──────────────────────────────────────────────────────

  function renderHeader() {
    var s = S.summary || {};
    var h = S.health || {};
    var parts = [];
    parts.push('<span class="kv">чип <b>' + esc(s.chip_id || DASH) + '</b></span>');
    parts.push('<span class="kv">событие <b>' + esc(s.event_id || DASH) + '</b></span>');
    parts.push('<span class="kv">срок решения <b>' + esc(s.decision_deadline || DASH) + '</b></span>');
    parts.push('<span class="kv">класс <b>' + esc(s.target_class || DASH) + '</b></span>');
    parts.push('<span class="kv">порог <b>' + fmt(s.threshold, 2) + '</b></span>');
    parts.push('<span class="kv">объявленный бюджет <b>' + money(s.budget_rub) + ' ₽</b></span>');
    if (s.base_rate_status === 'scenario') {
      parts.push(mark('scenario', 'сценарная ставка БРЕ',
        'base_rate_status = scenario: учебная ставка из постановки кейса, подтверждённого действующего размера БРЕ нет'));
    } else if (s.base_rate_status) {
      parts.push(mark('measured', 'ставка: ' + s.base_rate_status));
    }
    $('runline').innerHTML = parts.join('');

    var pill = $('health-pill');
    if (h.status === 'ok') {
      pill.className = 'pill ok';
      pill.textContent = 'комплект ' + (h.run_id || '');
      pill.title = h.run_dir || '';
    } else if (h.status === 'demo') {
      pill.className = 'pill demo';
      pill.textContent = 'демо-режим';
      pill.title = h.message || '';
    } else {
      pill.className = 'pill bad';
      pill.textContent = 'нет комплекта';
      pill.title = h.message || '';
    }
    if (s.demo) {
      banner('Открыта демонстрационная синтетика (demo = true): структура настоящая, числа — нет. ' +
        (h.message || ''), 'warn');
    }
  }

  function renderMeta() {
    var s = S.summary || {};
    var h = S.health || {};
    var html = '<dl class="kvgrid">';
    html += '<dt>run_id</dt><dd>' + esc(s.run_id || DASH) + '</dd>';
    html += '<dt>chip_id</dt><dd>' + esc(s.chip_id || DASH) + '</dd>';
    html += '<dt>событие</dt><dd>' + esc(s.event_id || DASH) + '</dd>';
    html += '<dt>срок решения</dt><dd>' + esc(s.decision_deadline || DASH) + ' ' + mark('scenario', 'сценарная дата') + '</dd>';
    html += '<dt>порог</dt><dd>' + fmt(s.threshold, 2) + ' ' + mark('model', 'подобран на validation') + '</dd>';
    html += '<dt>границы, °</dt><dd>' + (s.bounds ? s.bounds.map(function (v) { return Number(v).toFixed(4); }).join(', ') : DASH) + '</dd>';
    html += '<dt>правовая база</dt><dd>' + esc(s.legal_edition || DASH) + '</dd>';
    html += '<dt>версия сервиса</dt><dd>' + esc(h.version || DASH) + '</dd>';
    html += '</dl>';

    var methods = s.methods || {};
    if (Object.keys(methods).length) {
      html += '<details><summary>Методы</summary><ul>';
      Object.keys(methods).forEach(function (key) {
        var m = methods[key];
        var text = (m && typeof m === 'object') ? (m.kind || JSON.stringify(m)) : String(m);
        html += '<li><code>' + esc(key) + '</code> — ' + esc(text) + '</li>';
      });
      html += '</ul></details>';
    }

    var sources = s.sources;
    if (sources && !Array.isArray(sources) && sources.sources) sources = sources.sources;
    if (Array.isArray(sources) && sources.length) {
      html += '<details><summary>Источники данных (' + sources.length + ')</summary><ul>';
      sources.forEach(function (src) {
        html += '<li><code>' + esc(src.id || '') + '</code> ' + esc(src.purpose || '') +
          (src.url ? '<br><a href="' + esc(src.url) + '" target="_blank" rel="noopener">' + esc(src.url) + '</a>' : '') +
          (src.license ? '<br>' + esc(src.license) : '') + '</li>';
      });
      html += '</ul></details>';
    }

    $('meta-body').innerHTML = html;
  }

  // ── бюджет ───────────────────────────────────────────────────────────────

  function setupBudget() {
    var declared = toNum(S.summary && S.summary.budget_rub) || 0;
    var b = comparisonRow('B');
    var costB = toNum(b && b.decision_cost_rub) || 0;
    var max = Math.max(declared * 2.5, costB * 1.25, 10000);
    max = Math.ceil(max / 1000) * 1000;

    var rng = $('rng-budget');
    rng.min = 0;
    rng.max = max;
    rng.step = 100;
    rng.value = S.budget === null ? declared : S.budget;
    S.declaredBudget = declared;
    S.budget = Number(rng.value);

    $('budget-scale').innerHTML = '<span>0</span><span>' + fmt(max / 2, 0) + '</span><span>' + fmt(max, 0) + '</span>';
    $('budget-num').textContent = fmt(S.budget, 0);
  }

  function onBudgetInput() {
    S.budget = Number($('rng-budget').value);
    $('budget-num').textContent = fmt(S.budget, 0);
    // Маркер «сейчас» на кривой двигается сразу, без запроса: кривая от бюджета не зависит.
    if (S.curve) drawCurve();
    if (S.budgetTimer) clearTimeout(S.budgetTimer);
    S.budgetTimer = setTimeout(reloadStrategies, 220);
  }

  function reloadStrategies() {
    var seq = ++S.reqSeq;
    $('recalc').classList.remove('hidden');
    return getJSON('/api/strategies?budget=' + encodeURIComponent(S.budget))
      .then(function (data) {
        if (seq !== S.reqSeq) return;
        S.strategies = data;
        renderStrategies();
        restyleZones();
        renderAssetTable();
        if (S.selectedZone && S.candidates) {
          var f = S.candidates.features.filter(function (x) { return x.properties.candidate_id === S.selectedZone; })[0];
          if (f) showZoneCard(f);
        }
        banner('');
      })
      .catch(function (err) {
        if (seq !== S.reqSeq) return;
        banner('Пересчёт стратегий не выполнен: ' + err.message);
      })
      .then(function () {
        if (seq === S.reqSeq) $('recalc').classList.add('hidden');
      });
  }

  // ── кривая «бюджет → покрытый ущерб» ─────────────────────────────────────
  //
  // Ответ на главный вопрос заседания: сколько ущерба покрывает каждый рубль и после
  // какой суммы доплачивать бессмысленно. Ничего не считается в панели: каждая точка —
  // отдельный вызов /api/strategies?budget=, то есть тот же пересчёт, что за слайдером.
  // Панель только раскладывает ответы по оси и читает, где кривая перестаёт расти.

  var CURVE_POINTS = 24;       // точек выборки от нуля до стоимости полной закупки
  var CURVE_PARALLEL = 4;      // одновременных запросов — не забивать сервис на старте

  function fullPurchaseCost() {
    var rows = (S.strategies && S.strategies.comparison) || [];
    var b = rows.filter(function (r) { return r.strategy === 'B'; })[0];
    return b ? toNum(b.decision_cost_rub) : null;
  }

  function buildCurve() {
    var full = fullPurchaseCost();
    if (!full || full <= 0) {
      $('curve-status').textContent = 'нет стоимости полной закупки — кривую строить не по чему';
      return;
    }
    var budgets = [];
    for (var i = 0; i <= CURVE_POINTS; i++) budgets.push(Math.round(full * i / CURVE_POINTS * 100) / 100);
    var results = new Array(budgets.length);
    var next = 0, done = 0;

    return new Promise(function (resolve) {
      function pump() {
        if (next >= budgets.length) return;
        var idx = next++;
        getJSON('/api/strategies?budget=' + encodeURIComponent(budgets[idx]))
          .then(function (data) {
            var c = (data.comparison || []).filter(function (r) { return r.strategy === 'C'; })[0] || {};
            results[idx] = {
              budget: budgets[idx],
              covered: toNum(c.covered_expected_loss_rub),
              share: toNum(c.coverage_share),
              cost: toNum(c.decision_cost_rub),
              zones: ((data.plans && data.plans.C) || []).length
            };
          })
          .catch(function () { results[idx] = null; })
          .then(function () {
            done++;
            $('curve-status').textContent = 'строится… ' + done + ' из ' + budgets.length;
            if (done === budgets.length) resolve(results); else pump();
          });
      }
      for (var k = 0; k < CURVE_PARALLEL; k++) pump();
    }).then(function (points) {
      S.curve = points.filter(function (p) { return p && p.covered !== null; });
      S.curveFull = full;
      if (!S.curve.length) {
        $('curve-status').textContent = 'сервис не отдал ни одной точки';
        return;
      }
      $('curve-status').innerHTML = S.curve.length + ' точек · ' +
        mark('scenario', 'сценарная ставка', 'Цена зон считается по сценарной ставке БРЕ, поэтому и то, сколько ущерба покупается за рубль, — сценарное');
      drawCurve();
      renderCurveTable();
    });
  }

  /** Бюджет, с которого покрытие перестаёт расти: первая точка, достигшая максимума. */
  function curveSaturation() {
    var pts = S.curve || [];
    var top = pts.reduce(function (m, p) { return Math.max(m, p.covered); }, 0);
    for (var i = 0; i < pts.length; i++) if (pts[i].covered >= top - 0.5) return pts[i];
    return null;
  }

  /** Круглый шаг делений: 1, 2, 2,5 или 5 на нужный порядок. */
  function niceStep(raw) {
    var pow = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    var n = raw / pow;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * pow;
  }

  function shortRub(v) {
    if (v >= 1e6) return (v / 1e6).toLocaleString('ru-RU', { maximumFractionDigits: 1 }) + ' млн';
    if (v >= 1e3) return (v / 1e3).toLocaleString('ru-RU', { maximumFractionDigits: 0 }) + ' тыс.';
    return fmt(v, 0);
  }

  function drawCurve() {
    var box = $('curve');
    var pts = S.curve;
    if (!box || !pts || !pts.length) return;
    var W = Math.max(320, box.clientWidth), H = 230;
    var m = { l: 62, r: 18, t: 16, b: 34 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var xMax = S.curveFull;
    var yTop = pts.reduce(function (a, p) { return Math.max(a, p.covered); }, 0) || 1;
    var yStep = niceStep(yTop / 4), xStep = niceStep(S.curveFull / 5);
    var yMax = Math.ceil(yTop * 1.04 / yStep) * yStep;
    var X = function (v) { return m.l + Math.min(v, xMax) / xMax * iw; };
    var Y = function (v) { return m.t + ih - v / yMax * ih; };

    var svg = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">';
    // сетка и подписи — приглушённые, чтобы не спорить с линией
    for (var gy = 0; gy <= yMax + 1e-6; gy += yStep) {
      var py = Y(gy);
      svg += '<line class="c-grid" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + py + '" y2="' + py + '"/>';
      svg += '<text class="c-tick" x="' + (m.l - 8) + '" y="' + (py + 4) + '" text-anchor="end">' + shortRub(gy) + '</text>';
    }
    for (var bx = 0; bx <= xMax + 1e-6; bx += xStep) {
      svg += '<text class="c-tick" x="' + X(bx) + '" y="' + (H - 12) + '" text-anchor="middle">' + shortRub(bx) + '</text>';
    }
    svg += '<text class="c-axis" x="' + (W - m.r) + '" y="' + (H - 1) + '" text-anchor="end">бюджет, ₽</text>';

    // ступенька: покрытие держится до следующей точки выборки
    var d = 'M' + X(pts[0].budget) + ',' + Y(pts[0].covered);
    for (var i = 1; i < pts.length; i++) d += 'H' + X(pts[i].budget) + 'V' + Y(pts[i].covered);
    svg += '<path class="c-area" d="' + d + 'V' + Y(0) + 'H' + X(pts[0].budget) + 'Z"/>';
    svg += '<path class="c-line" d="' + d + '"/>';

    // насыщение: единственная прямая подпись на кривой
    var sat = curveSaturation();
    if (sat) {
      var sx = X(sat.budget), sy = Y(sat.covered);
      svg += '<circle class="c-sat" cx="' + sx + '" cy="' + sy + '" r="5"/>';
      var anchor = sx > W * 0.6 ? 'end' : 'start', off = anchor === 'end' ? -10 : 10;
      svg += '<text class="c-label" x="' + (sx + off) + '" y="' + (sy + 18) + '" text-anchor="' + anchor + '">насыщение ≈ ' + shortRub(sat.budget) + ' ₽</text>';
    }

    // текущий бюджет со слайдера
    var cur = S.budget || 0;
    var cx = X(cur);
    svg += '<line class="c-now" x1="' + cx + '" x2="' + cx + '" y1="' + m.t + '" y2="' + (m.t + ih) + '"/>';
    svg += '<text class="c-now-label" x="' + (cx + (cx > W * 0.75 ? -6 : 6)) + '" y="' + (m.t + 10) + '" text-anchor="' + (cx > W * 0.75 ? 'end' : 'start') + '">' +
      (cur > xMax ? 'сейчас: выше полной закупки' : 'сейчас ' + shortRub(cur) + ' ₽') + '</text>';

    // слой наведения: вся площадь графика — мишень, а не только линия
    svg += '<line class="c-cross hidden" id="c-cross" y1="' + m.t + '" y2="' + (m.t + ih) + '"/>';
    svg += '<circle class="c-dot hidden" id="c-dot" r="4.5"/>';
    svg += '<rect class="c-hit" x="' + m.l + '" y="' + m.t + '" width="' + iw + '" height="' + ih + '"/>';
    svg += '</svg><div class="c-tip hidden" id="c-tip"></div>';
    box.innerHTML = svg;

    var hit = box.querySelector('.c-hit'), tip = $('c-tip'), cross = $('c-cross'), dot = $('c-dot');
    hit.addEventListener('mousemove', function (e) {
      var r = box.getBoundingClientRect();
      var bx = (e.clientX - r.left - m.l) / iw * xMax;
      // ступенька: значение на бюджете bx — у последней точки не правее bx
      var j = 0;
      for (var k = 0; k < pts.length; k++) if (pts[k].budget <= bx + 1e-6) j = k;
      var p = pts[j], prev = pts[j - 1];
      var px = X(p.budget), py = Y(p.covered);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.classList.remove('hidden');
      dot.setAttribute('cx', px); dot.setAttribute('cy', py); dot.classList.remove('hidden');
      var gain = prev && p.budget > prev.budget ? (p.covered - prev.covered) / (p.budget - prev.budget) * 1000 : null;
      tip.innerHTML =
        '<b>бюджет ' + money(p.budget) + ' ₽</b>' +
        '<div>покрыто ' + money(p.covered) + ' ₽ · ' + pct(p.share) + ' портфеля</div>' +
        '<div>зон в заказе: ' + p.zones + ', стоимость ' + money(p.cost) + ' ₽</div>' +
        (gain !== null ? '<div class="sub">на этом шаге: +' + shortRub(Math.max(gain, 0)) + ' ₽ покрытия на 1 000 ₽</div>' : '');
      tip.classList.remove('hidden');
      var tx = px + 14, tw = tip.offsetWidth;
      if (tx + tw > W) tx = px - tw - 14;
      tip.style.left = Math.max(0, tx) + 'px';
      tip.style.top = Math.max(0, py - 20) + 'px';
    });
    hit.addEventListener('mouseleave', function () {
      tip.classList.add('hidden'); cross.classList.add('hidden'); dot.classList.add('hidden');
    });

    renderCurveNote(sat);
  }

  function renderCurveNote(sat) {
    var pts = S.curve || [];
    if (!sat) { $('curve-note').textContent = ''; return; }
    // Предельная отдача перед насыщением — отношение двух ответов сервиса, не новый расчёт.
    var i = pts.indexOf(sat), before = null;
    for (var k = i - 1; k >= 0; k--) if (pts[k].covered < sat.covered - 0.5) { before = pts[k]; break; }
    var html = 'С бюджета около <b>' + money(sat.budget) + ' ₽</b> стратегия C покрывает <b>' + money(sat.covered) +
      ' ₽</b> ожидаемого ущерба (' + pct(sat.share) + ' портфеля); дальше рост бюджета покрытия не добавляет. ';
    if (before) {
      var gain = (sat.covered - before.covered) / (sat.budget - before.budget) * 1000;
      html += 'На последнем шаге до насыщения каждая 1 000 ₽ покупала около ' + shortRub(gain) + ' ₽ покрытия. ';
    }
    html += 'Шаг выборки — ' + money(S.curveFull / CURVE_POINTS) + ' ₽, поэтому точка насыщения указана с этой точностью. ' +
      'Каждая точка — тот же пересчёт, что за слайдером: модель не переобучается, порог не меняется. ' +
      mark('scenario', 'сценарное', 'Цены зон — по сценарной ставке БРЕ; при другой ставке кривая сдвинется по горизонтали');
    $('curve-note').innerHTML = html;
  }

  function renderCurveTable() {
    var html = '<thead><tr><th>бюджет, ₽</th><th>зон</th><th>стоимость, ₽</th><th>покрыто, ₽</th><th>доля</th></tr></thead><tbody>';
    (S.curve || []).forEach(function (p) {
      html += '<tr><td class="num">' + money(p.budget) + '</td><td class="num">' + p.zones + '</td><td class="num">' +
        money(p.cost) + '</td><td class="num">' + money(p.covered) + '</td><td class="num">' + pct(p.share) + '</td></tr>';
    });
    $('tbl-curve').innerHTML = html + '</tbody>';
  }

  // ── события интерфейса ───────────────────────────────────────────────────

  function wire() {
    $('rng-budget').addEventListener('input', onBudgetInput);
    $('btn-budget-reset').addEventListener('click', function () {
      $('rng-budget').value = S.declaredBudget;
      onBudgetInput();
    });

    Array.prototype.forEach.call($('strategy-switch').querySelectorAll('button'), function (btn) {
      btn.addEventListener('click', function () { setStrategy(btn.getAttribute('data-strategy')); });
    });

    $('chk-s1').addEventListener('change', syncLayers);
    Array.prototype.forEach.call(document.querySelectorAll('input[name=thematic]'), function (radio) {
      radio.addEventListener('change', function () { S.layer = radio.value; syncLayers(); });
    });
    $('rng-opacity').addEventListener('input', function () {
      S.opacity = Number($('rng-opacity').value) / 100;
      $('lbl-opacity').textContent = $('rng-opacity').value + '%';
      if (S.layer !== 'none' && S.overlays[S.layer]) S.overlays[S.layer].setOpacity(S.opacity);
    });
    $('chk-zones').addEventListener('change', function () {
      if (!S.zonesLayer) return;
      if ($('chk-zones').checked) { S.zonesLayer.addTo(S.map); reorder(); }
      else S.map.removeLayer(S.zonesLayer);
    });
    $('chk-assets').addEventListener('change', function () {
      if (!S.assetsLayer) return;
      if ($('chk-assets').checked) { S.assetsLayer.addTo(S.map); reorder(); }
      else S.map.removeLayer(S.assetsLayer);
    });
  }

  // ── старт ────────────────────────────────────────────────────────────────

  function boot() {
    wire();
    Promise.all([
      getJSON('/api/health'),
      getJSON('/api/run'),
      getJSON('/api/assets'),
      getJSON('/api/candidates'),
      getJSON('/api/sensitivity'),
      getJSON('/api/strategies')
    ]).then(function (res) {
      S.health = res[0];
      S.summary = res[1];
      S.assets = res[2];
      S.candidates = res[3];
      S.sensitivity = res[4];
      S.strategies = res[5];
      S.budget = toNum(S.strategies.budget_rub);
      if (S.budget === null) S.budget = toNum(S.summary.budget_rub);

      renderHeader();
      renderMeta();
      initMap();
      renderZones();
      renderAssets();
      renderStrategies();
      renderSensitivity();
      setupBudget();
      buildCurve();
      syncLayers();
      $('zone-body').innerHTML = '<div class="empty-hint">Нажмите зону на карте — здесь появятся цена, площадь, сроки и коэффициенты ПП РФ № 840.</div>';
    }).catch(function (err) {
      banner('Панель не загружена: ' + err.message +
        '. Проверьте, что сервис запущен (python -m src.api) и комплект доступен.');
      var pill = $('health-pill');
      pill.className = 'pill bad';
      pill.textContent = 'ошибка';

      // Без комплекта выгружать нечего: ссылка вела бы на 404, а органы управления
      // картой переключали бы пустоту. Гасим их, чтобы отказ читался как отказ, а не
      // как сломанная панель — на защите это разные впечатления.
      var bundle = $('bundle-link');
      if (bundle) {
        bundle.classList.add('disabled');
        bundle.setAttribute('aria-disabled', 'true');
        bundle.removeAttribute('href');
        bundle.title = 'Комплект не загружен — выгружать нечего';
      }
      var ctl = $('mapctl');
      if (ctl) {
        var inputs = ctl.querySelectorAll('input');
        for (var i = 0; i < inputs.length; i++) inputs[i].disabled = true;
      }
      var budget = $('rng-budget');
      if (budget) budget.disabled = true;
      var reset = $('btn-budget-reset');
      if (reset) reset.disabled = true;

      // Шапка иначе навсегда остаётся в состоянии «загрузка…».
      $('runline').innerHTML = '<span class="muted">комплект не загружен</span>';
    });
  }

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { if (S.curve) drawCurve(); }, 150);
  });

  document.addEventListener('DOMContentLoaded', boot);
})();
