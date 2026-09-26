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

  /**
   * Светофор приоритета проверки.
   *
   * Раньше здесь был плавный градиент от красного к бежевому, и он читался плохо:
   * между четвёртым и пятым объектом разницы на глаз нет, а решение принимается
   * именно между ними. Человек надёжно различает три состояния, а не десять
   * оттенков, поэтому приоритет разложен на три ступени по тому же рангу.
   *
   * Границы ступеней не случайны: верхняя треть портфеля — то, что проверяют в
   * первую очередь, нижняя — то, до чего доходят, если останутся деньги.
   */
  var PRIORITY_TIERS = [
    // Светофор, а не один тон разной силы. Возражение против красного понятное —
    // на карте паводка он рискует читаться как «беда» там, где речь про порядок
    // проверки. Но заказчик просил именно светофор, и довод у него сильнее: три
    // насыщенности одного цвета различает глаз, натренированный на графики, а
    // красный-жёлтый-зелёный узнаёт любой человек без объяснений.
    //
    // Компромисс в тонах: это приглушённые оттенки под светлый фон, а не сигнальные.
    // И цвет нигде не остаётся один — рядом всегда стоит слово, иначе ступень
    // пропадает при дальтонизме и в чёрно-белой печати.
    { key: 'high', title: 'высокий', color: '#c0392b', text: 'проверять в первую очередь' },
    { key: 'mid', title: 'средний', color: '#d98324', text: 'проверять, если позволяет бюджет' },
    { key: 'low', title: 'низкий', color: '#2f8f4e', text: 'можно отложить' },
    { key: 'none', title: 'нет оценки', color: '#8b8f9e', text: 'оценка не построена' }
  ];

  function priorityTier(rank, total) {
    if (rank === null || rank === undefined || rank === '') return PRIORITY_TIERS[3];
    var n = Number(rank);
    var count = Number(total) || 10;
    if (!isFinite(n) || n < 1) return PRIORITY_TIERS[3];
    if (n <= Math.ceil(count / 3)) return PRIORITY_TIERS[0];
    if (n <= Math.ceil((count * 2) / 3)) return PRIORITY_TIERS[1];
    return PRIORITY_TIERS[2];
  }

  function rankedCount() {
    var rows = ((S.assets || {}).features) || [];
    var n = 0;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i].properties.rank;
      if (r !== null && r !== undefined && r !== '') n++;
    }
    return n || 10;
  }

  function rankColor(rank) {
    return priorityTier(rank, rankedCount()).color;
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

  /**
   * Выбранный комплект живёт в адресной строке (?run=...), а не в памяти страницы.
   * Из-за этого ссылка на конкретный чип пересылается и открывается как есть — на
   * защите это удобнее, чем объяснять, что и где нажать. Пустая строка означает
   * комплект, открытый сервисом по умолчанию.
   */
  function currentRunId() {
    try {
      return new URLSearchParams(window.location.search).get('run') || '';
    } catch (e) {
      return '';
    }
  }

  /** Добавляет к адресу выбранный комплект, сохраняя уже имеющиеся параметры. */
  function withRun(path) {
    var runId = currentRunId();
    if (!runId) return path;
    return path + (path.indexOf('?') >= 0 ? '&' : '?') + 'run=' + encodeURIComponent(runId);
  }

  function getJSON(rawPath) {
    var path = withRun(rawPath);
    return fetch(path, { headers: { Accept: 'application/json' } }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error(body.error || body.detail || ('HTTP ' + r.status + ' ' + path));
        return body;
      });
    });
  }

  function fetchRaster(kind) {
    return fetch(withRun('/api/raster/' + kind + '.png')).then(function (r) {
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
    // zoomSnap: 0 — дробный зум. С целым шагом Leaflet не может подогнать кадр под
    // окно и оставляет его занимать чуть больше половины площади: чип 512×512 просто
    // не попадает ни в один целый уровень. Оператор при этом видит снимок в полтора
    // раза мельче, чем мог бы, и половина карты уходит под пустую подложку.
    S.map = L.map('map', {
      preferCanvas: true,
      zoomControl: true,
      minZoom: 3,
      zoomSnap: 0,
      zoomDelta: 0.5,
      wheelPxPerZoomLevel: 120
    });
    // Leaflet 1.9 вставляет в подпись собственный флаг. В панели оценки ущерба ему
    // делать нечего, а ссылка на библиотеку остаётся — лицензия требует именно её.
    S.map.attributionControl.setPrefix(
      '<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
    // Географический контекст. Тайлы приглушены фильтром в styles.css; если сети
    // нет, подложка просто не появится — слои запуска от неё не зависят.
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap · снимок Sentinel-1, Sen1Floods11'
    }).addTo(S.map);

    fitChip();
  }

  /** Показывает кадр целиком, во всю доступную площадь карты. */
  function fitChip() {
    var b = leafletBounds(S.summary && S.summary.bounds);
    if (b) S.map.fitBounds(b, { padding: [6, 6], animate: false });
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

  /** Лучший (то есть самый срочный) ранг среди объектов, попавших в зону. */
  function zoneTier(props) {
    // В каталоге covered_asset_ids — массив; в CSV-представлении — строка через «;».
    // Раньше массив превращался в «a04,a10» и не совпадал ни с одним объектом:
    // все заказанные зоны красились как «нет оценки».
    var raw = props.covered_asset_ids || [];
    var ids = Array.isArray(raw) ? raw.slice() : String(raw).split(';').filter(Boolean);
    if (!ids.length) return PRIORITY_TIERS[3];
    var rows = ((S.assets || {}).features) || [];
    var best = null;
    for (var i = 0; i < rows.length; i++) {
      var pr = rows[i].properties;
      if (ids.indexOf(pr.asset_id) < 0) continue;
      var r = pr.rank;
      if (r === null || r === undefined || r === '') continue;
      if (best === null || Number(r) < best) best = Number(r);
    }
    return priorityTier(best, rankedCount());
  }

  function zoneStyle(feature) {
    var id = feature.properties.candidate_id;
    var chosen = !!selectedSet()[id];
    var picked = S.selectedZone === id;
    if (chosen) {
      // Заказанная зона красится по срочности того, ради чего её заказали:
      // оператор видит на карте не «куплено/не куплено», а «куплено ради чего».
      var tier = zoneTier(feature.properties);
      return {
        color: picked ? '#1f2250' : '#3d4180',
        weight: picked ? 3 : 2,
        opacity: 1,
        dashArray: null,  // setStyle не сбрасывает пунктир сам — зона из каталога осталась бы пунктирной
        fillColor: tier.color,
        fillOpacity: picked ? 0.22 : 0.12
      };
    }
    return {
      color: picked ? '#1f2250' : '#ffffff',
      weight: picked ? 2.4 : 1.4,
      opacity: 0.95,
      fillColor: '#ffffff',
      fillOpacity: 0.02,
      dashArray: '5 4'
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

  /**
   * Цепочка «от пикселя до рубля» для одного объекта. Ничего не пересчитывает:
   * p, V, q и ущерб — из /api/assets, позиции и цены — из /api/strategies. Панель
   * только ставит их в одну строку, чтобы было видно, откуда берётся каждое число.
   */
  function assetChain(props, geometry, ok) {
    var c = (geometry && geometry.coordinates) || [];
    var thr = toNum(S.summary && S.summary.threshold);
    var h = '<div class="chain"><div class="chain-h">почему такой ущерб</div><ol>';
    if (!ok) {
      h += '<li>В пикселе объекта растр вероятности пуст или его окрестность покрыта нет-данными больше чем наполовину. ' +
        'Оценки нет — ранг и ущерб не выставляются и нулём не заменяются.</li></ol></div>';
      return h;
    }
    var p = toNum(props.p_flood);
    h += '<li><b>p = ' + fmt(p, 4) + '</b> — значение вероятностного растра основного метода в пикселе объекта' +
      (c.length ? ' (' + c[0].toFixed(5) + ', ' + c[1].toFixed(5) + ')' : '') + ', калибровано на validation.</li>';
    h += '<li><b>V = ' + fmt(props.asset_value_rub, 0) + ' ₽</b>, <b>q = ' + fmt(props.vulnerability_coef, 2) + '</b> — стоимость и уязвимость из условия кейса.</li>';
    h += '<li><b>EL = p × V × q</b> = ' + fmt(p, 4) + ' × ' + fmt(props.asset_value_rub, 0) + ' × ' + fmt(props.vulnerability_coef, 2) +
      ' = <b>' + money(props.expected_loss_rub) + ' ₽</b> → ранг ' + (props.rank == null ? DASH : props.rank) + ' из 10.</li>';
    if (thr !== null && p !== null) {
      h += '<li>На маске объект ' + (p >= thr ? '<b>в зоне затопления</b> (p ≥ ' : '<b>вне зоны затопления</b> (p < ') + fmt(thr, 2) +
        '), но на ущерб это не влияет: ущерб считается по вероятности, а не по маске, и низкое p его не обнуляет.</li>';
    }
    var plan = {};
    ((S.strategies && S.strategies.plans && S.strategies.plans[S.strategy]) || []).forEach(function (id) { plan[id] = true; });
    var price = {};
    ((S.strategies && S.strategies.positions) || []).forEach(function (pos) { if (pos.strategy === S.strategy) price[pos.candidate_id] = pos.cost_rub; });
    var zones = ((S.candidates && S.candidates.features) || []).filter(function (z) {
      return plan[z.properties.candidate_id] && (z.properties.covered_asset_ids || []).indexOf(props.asset_id) >= 0;
    });
    if (zones.length) {
      h += '<li>В стратегии ' + esc(S.strategy) + ' проверяется съёмкой: ' + zones.map(function (z) {
        var id = z.properties.candidate_id;
        return '<b>' + esc(id) + '</b>' + (price[id] != null ? ' — ' + money(price[id]) + ' ₽ в этой корзине' : '');
      }).join(', ') + ' ' + mark('scenario', 'сценарная ставка') + '.</li>';
    } else {
      h += '<li>В стратегии ' + esc(S.strategy) + ' заказанной съёмкой <b>не проверяется</b> — подтверждение только на месте.</li>';
    }
    return h + '</ol></div>';
  }

  function assetPopup(props, geometry) {
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
    html += assetChain(props, geometry, ok);
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
        // Номер объекта в круге. Верхняя треть по ущербу — заливка, остальные —
        // белый круг с обводкой; объект без оценки — пунктир. Это DOM, а не canvas:
        // номер должен читаться, а подписи на canvas Leaflet не рисует.
        var num = String(p.asset_id || '').replace(/^a0*/, '') || '?';
        var tier = ok ? priorityTier(p.rank, rankedCount()).key : 'none';
        return L.marker(latlng, {
          icon: L.divIcon({
            className: 'amark amark-' + tier,
            html: '<span>' + esc(num) + '</span>',
            iconSize: [26, 26],
            iconAnchor: [13, 13]
          }),
          riseOnHover: true,
          keyboard: false
        });
      },
      onEachFeature: function (feature, layer) {
        layer.bindTooltip(feature.properties.asset_id + ' · ' +
          (ASSET_TITLES[feature.properties.asset_class] || feature.properties.asset_class),
          { direction: 'top', offset: [0, -6] });
        layer.on('click', function () {
          layer.bindPopup(assetPopup(feature.properties, feature.geometry), { maxWidth: 340 }).openPopup();
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
            layer.bindPopup(assetPopup(layer.feature.properties, layer.feature.geometry), { maxWidth: 340 }).openPopup();
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
  // ── таймлайн «успеете ли к сроку» ────────────────────────────────────────
  //
  // Кейс требует, чтобы данные, недоступные к сроку решения, не увеличивали
  // оперативное покрытие. В расчёте это уже так (is_event_observation в
  // src/procurement/strategies.py), но на экране не было видно. Здесь то же правило
  // показано на датах из /api/candidates: наблюдение события, доступное не позже
  // срока, — идёт в покрытие; контекст до события и опоздавшие — нет.

  function plural(n, one, few, many) {
    var m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
  }

  function zoneTimelineStatus(zp, deadline) {
    if (zp.data_role !== 'event_observation') return { key: 'context', word: 'контекст — в покрытие не идёт' };
    if (deadline && zp.available_at && zp.available_at > deadline) return { key: 'late', word: 'не успевает — в покрытие не идёт' };
    return { key: 'ok', word: 'успевает — идёт в покрытие' };
  }

  function renderTimeline() {
    var box = $('timeline');
    if (!box || !S.candidates) return;
    var deadline = (S.summary && S.summary.decision_deadline) || '';
    var plan = {};
    ((S.strategies && S.strategies.plans && S.strategies.plans[S.strategy]) || []).forEach(function (id) { plan[id] = true; });
    var groups = {}, order = [];
    S.candidates.features.forEach(function (f) {
      var zp = f.properties;
      var key = [zp.data_role, zp.acquisition_type, zp.observation_at, zp.available_at].join('|');
      if (!groups[key]) {
        groups[key] = { zp: zp, status: zoneTimelineStatus(zp, deadline), total: 0, inPlan: 0 };
        order.push(key);
      }
      groups[key].total++;
      if (plan[zp.candidate_id]) groups[key].inPlan++;
    });
    var rows = order.map(function (k) { return groups[k]; })
      .sort(function (a, b) { return String(a.zp.observation_at).localeCompare(String(b.zp.observation_at)); });

    var dates = [deadline];
    rows.forEach(function (g) { dates.push(g.zp.observation_at, g.zp.available_at); });
    var ts = dates.filter(Boolean).map(function (d) { return Date.parse(d); }).filter(isFinite);
    if (!ts.length) { box.innerHTML = ''; return; }
    var t0 = Math.min.apply(null, ts), t1 = Math.max.apply(null, ts);
    var span = Math.max(t1 - t0, 86400000);
    t0 -= span * 0.06; t1 += span * 0.10;

    var W = Math.max(320, box.clientWidth), rowH = 34, top = 26, H = top + rows.length * rowH + 26;
    var ml = 12, mr = 12, iw = W - ml - mr;
    var X = function (d) { return ml + (Date.parse(d) - t0) / (t1 - t0) * iw; };
    var svg = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">';
    // месячные засечки
    var m = new Date(t0); m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1));
    while (m.getTime() < t1) {
      var mx = ml + (m.getTime() - t0) / (t1 - t0) * iw;
      svg += '<line class="t-grid" x1="' + mx + '" x2="' + mx + '" y1="' + (top - 6) + '" y2="' + (H - 20) + '"/>';
      svg += '<text class="t-tick" x="' + mx + '" y="' + (H - 6) + '" text-anchor="middle">' +
        m.toLocaleDateString('ru-RU', { month: 'short', year: '2-digit', timeZone: 'UTC' }) + '</text>';
      m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1));
    }
    rows.forEach(function (g, i) {
      var y = top + i * rowH + rowH / 2, x1 = X(g.zp.observation_at), x2 = X(g.zp.available_at || g.zp.observation_at);
      svg += '<line class="t-bar t-' + g.status.key + '" x1="' + x1 + '" x2="' + Math.max(x2, x1 + 0.01) + '" y1="' + y + '" y2="' + y + '"/>';
      svg += '<circle class="t-obs t-' + g.status.key + '" cx="' + x1 + '" cy="' + y + '" r="5"><title>съёмка ' + esc(g.zp.observation_at) + '</title></circle>';
      svg += '<rect class="t-avail t-' + g.status.key + '" x="' + (x2 - 4) + '" y="' + (y - 4) + '" width="8" height="8"><title>доступно ' + esc(g.zp.available_at) + '</title></rect>';
      // Подпись у самой полосы: без неё не понять, какая строка о чём, пока не
      // спустишься к таблице. Справа от кружка, а у правого края — слева.
      var right = x1 > W * 0.6;
      var label = (ACQ_WORD[g.zp.acquisition_type] || g.zp.acquisition_type) + ', ' + (SENSOR_WORD[g.zp.sensor_type] || g.zp.sensor_type) +
        ' · ' + g.total + ' ' + plural(g.total, 'зона', 'зоны', 'зон') + ' · ' + g.status.word;
      svg += '<text class="t-label" x="' + (right ? Math.min(x1, x2) - 10 : x1 + 10) + '" y="' + (y - 7) + '" text-anchor="' + (right ? 'end' : 'start') + '">' + esc(label) + '</text>';
    });
    if (deadline) {
      var dx = X(deadline);
      svg += '<line class="t-deadline" x1="' + dx + '" x2="' + dx + '" y1="' + 4 + '" y2="' + (H - 20) + '"/>';
      svg += '<text class="t-deadline-label" x="' + (dx - 6) + '" y="14" text-anchor="end">срок решения ' + esc(deadline) + '</text>';
    }
    box.innerHTML = svg + '</svg>';

    var html = '<thead><tr><th></th><th>данные</th><th>съёмка</th><th>доступно</th><th class="r">зон в каталоге</th><th class="r">в плане ' +
      esc(S.strategy) + '</th><th>к сроку</th></tr></thead><tbody>';
    rows.forEach(function (g) {
      var zp = g.zp;
      html += '<tr><td><span class="t-key t-' + g.status.key + '"></span></td>' +
        '<td>' + esc(SENSOR_WORD[zp.sensor_type] || zp.sensor_type || '') + ', ' + fmt(zp.resolution_m, 1) + ' м · ' +
        esc(ACQ_WORD[zp.acquisition_type] || zp.acquisition_type || '') + ' · ' + esc(ROLE_WORD[zp.data_role] || zp.data_role || '') + '</td>' +
        '<td>' + esc(zp.observation_at || DASH) + '</td><td>' + esc(zp.available_at || DASH) + '</td>' +
        '<td class="num">' + g.total + '</td><td class="num">' + g.inPlan + '</td>' +
        '<td>' + esc(g.status.word) + '</td></tr>';
    });
    $('tbl-timeline').innerHTML = html + '</tbody>';
    $('timeline-status').innerHTML = mark('scenario', 'сценарные даты', 'Даты съёмки и доступности сценарные: привязаны к дате события из метаданных Sen1Floods11, подтверждением поставщика не являются');
    $('timeline-note').innerHTML = 'Кружок — дата съёмки, квадрат — дата, к которой данные доступны. В оперативное покрытие идёт только наблюдение ' +
      'события, доступное не позже срока решения; архивный снимок до события — контекст, и высокое разрешение его в наблюдение паводка не превращает. ' +
      'Опоздавшие к сроку данные покрытие не увеличивают — так же считает и сервис.';
  }

  function renderStrategies() {
    renderTimeline();
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
    emit('strategy');
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
    var tick = threshold === null ? '' :
      '<i class="lg-tick" style="left:' + (threshold * 100).toFixed(1) + '%"></i>' +
      '<span class="lg-tau" style="left:' + (threshold * 100).toFixed(1) + '%">τ = ' + fmt(threshold, 2) + '</span>';
    var html = '';
    if (S.layer === 'probability') {
      html = '<div class="lg-scale"><div class="lg-bar lg-prob" style="background:' + gradient(PROB_STOPS) + '"></div>' + tick +
        '<span class="lg-0">0</span><span class="lg-1">1</span></div>' +
        '<div class="lg-text">вероятность временного затопления · τ зафиксирован на validation ' + mark('model', 'расчёт модели') + '</div>';
    } else if (S.layer === 'uncertainty') {
      html = '<div class="lg-scale"><div class="lg-bar" style="background:' + gradient(UNC_STOPS) + '"></div>' +
        '<span class="lg-0">0</span><span class="lg-1">1</span></div>' +
        '<div class="lg-text">неопределённость: 0,7 × энтропия вероятности + 0,3 × разброс трёх моделей ' + mark('model', 'расчёт модели') + '</div>';
    } else if (S.layer === 'mask') {
      html = '<div class="lg-mask"><span class="swatch" style="background:' + MASK_COLOR + '"></span>' +
        'временное затопление, p ≥ ' + (threshold === null ? DASH : fmt(threshold, 2)) + '</div>' +
        '<div class="lg-text">постоянная вода исключена на стороне разметки · на ущерб маска не влияет</div>';
    } else {
      html = '<div class="lg-text">радарная подложка Sentinel-1, канал VV, дБ · тематический слой выключен</div>';
    }
    // Точки объектов и заказанные зоны на карте покрашены по срочности проверки.
    // Цвет без подписи читается как украшение, поэтому ступени всегда перечислены
    // рядом — и словом тоже, чтобы работало при дальтонизме и в печати.
    html += '<div class="lg-prio">';
    for (var i = 0; i < 3; i++) {
      var t = PRIORITY_TIERS[i];
      html += '<span class="lg-prio-item" title="' + t.text + '">' +
        '<i style="background:' + t.color + '"></i>' + t.title + '</span>';
    }
    html += '</div>';
    $('legend').innerHTML = html;
  }

  // ── шапка и паспорт ──────────────────────────────────────────────────────

  function renderHeader() {
    var s = S.summary || {};
    var h = S.health || {};
    var parts = [];
    parts.push('<span class="kv">чип <b>' + esc(s.chip_id || DASH) + '</b></span>');
    parts.push('<span class="kv">событие <b>' + esc(s.event_id || DASH) + '</b></span>');
    parts.push('<span class="kv">съёмка <b>' + esc(s.observation_date || DASH) + '</b></span>');
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
    html += '<dt>дата съёмки S1</dt><dd>' + esc(s.observation_date || DASH) + '</dd>';
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
          (src.observation_date ? '<br>съёмка: ' + esc(src.observation_date) : '') +
          (src.license ? '<br>' + esc(src.license) : '') + '</li>';
      });
      html += '</ul></details>';
    }

    $('meta-body').innerHTML = html;
  }

  // ── бюджет ───────────────────────────────────────────────────────────────

  // ── масштаб события ──────────────────────────────────────────────────────

  /**
   * Одно событие в трёх единицах. Все три приходят из /api/run и посчитаны из
   * того же вероятностного растра, что и рублёвый ущерб: отдельной модели
   * «в штуках» нет, иначе числа начали бы расходиться между собой.
   */
  function renderImpact() {
    var imp = (S.summary && S.summary.impact) || {};
    var t = imp.totals || {};
    var q = (S.summary && S.summary.quality_pct) || {};

    if (!t.expected_loss_rub && !t.expected_objects) {
      $('impact-totals').innerHTML = '<div class="empty-hint">Оценка последствий не построена.</div>';
      $('impact-classes').innerHTML = '';
      $('impact-note').textContent = '';
      $('impact-quality').textContent = '—';
      return;
    }

    function cell(value, key, title) {
      return '<div class="impact-cell" title="' + esc(title || '') + '"><span class="v">' +
        value + '</span><span class="k">' + esc(key) + '</span></div>';
    }

    $('impact-totals').innerHTML =
      cell(money(t.expected_loss_rub) + ' ₽', 'ожидаемый ущерб',
        'Сумма p × V × q по десяти объектам портфеля') +
      cell(fmt(t.expected_objects, 2), 'объектов из ' + (t.total_objects || 10),
        'Сумма вероятностей: объект с вероятностью 0,4 даёт 0,4 ожидаемого объекта') +
      cell(fmt(t.expected_area_km2, 2) + ' км²', 'ожидаемое затопление',
        'Сумма вероятностей по пикселям снимка, умноженная на площадь пикселя');

    var rows = imp.by_class || [];
    var top = rows.slice(0, 6);
    var maxLoss = 0;
    top.forEach(function (r) { maxLoss = Math.max(maxLoss, Number(r.expected_loss_rub) || 0); });
    var html = '<thead><tr><th>объект</th><th class="n">ожид. шт.</th><th class="n">ущерб, ₽</th></tr></thead><tbody>';
    top.forEach(function (r) {
      var share = maxLoss ? (Number(r.expected_loss_rub) || 0) / maxLoss : 0;
      html += '<tr><td>' + esc(r.title_ru || r.asset_class) + '</td>' +
        '<td class="n">' + fmt(r.expected_objects, 2) + '</td>' +
        '<td class="n">' + money(r.expected_loss_rub) +
        // Дорожка фиксированной ширины: полоска в процентах от ячейки стояла в одну
        // строку с числом и у самого крупного объекта вылезала за край карточки.
        '<span class="ibar"><i style="width:' + Math.round(share * 100) + '%"></i></span></td></tr>';
    });
    html += '</tbody>';
    $('impact-classes').innerHTML = html;

    if (q.f1_pct !== null && q.f1_pct !== undefined) {
      $('impact-quality').textContent = 'точность ' + fmt(q.precision_pct, 1) + ' %';
      $('impact-quality').title =
        'На независимой проверке, часть test, ' + (q.chips || '—') + ' чипов: из того, что ' +
        'названо затоплением, действительно затоплено ' + fmt(q.precision_pct, 1) + ' %; ' +
        'из того, что затоплено, найдено ' + fmt(q.recall_pct, 1) + ' %; сводная F1 ' +
        fmt(q.f1_pct, 1) + ' %, совпадение областей IoU ' + fmt(q.iou_pct, 1) + ' %.';
    } else {
      $('impact-quality').textContent = 'точность не посчитана';
    }

    $('impact-note').textContent =
      'Ожидаемое число объектов и площадь — суммы вероятностей, а не счёт по маске. ' +
      'Округлять их до целого нельзя: получится предположение, выданное за факт.';
  }

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
    renderBudgetContext();
  }

  /**
   * Бюджет сам по себе — голое число, и на чекпоинте это прозвучало прямо:
   * шесть тысяч рублей выглядят несерьёзно, пока рядом не написано, к чему они
   * относятся. Поэтому под слайдером всегда стоит отношение к ожидаемому ущербу
   * и к двум опорным точкам: сколько надо, чтобы закрыть всё, и во что обошлась
   * бы закупка без разбора.
   */
  function renderBudgetContext() {
    var node = $('budget-context');
    if (!node) return;
    var imp = (S.summary && S.summary.impact) || {};
    var loss = toNum((imp.totals || {}).expected_loss_rub);
    var b = comparisonRow('B');
    var broad = toNum(b && b.decision_cost_rub);
    var parts = [];

    if (loss) {
      var share = (S.budget / loss) * 100;
      parts.push('Это <b>' + (share < 0.1 ? share.toFixed(3) : share.toFixed(1)) +
        ' %</b> ожидаемого ущерба <b>' + money(loss) + ' ₽</b>, в который обойдётся само ' +
        'событие. Съёмка не уменьшает ущерб — она уменьшает незнание о нём.');
    }
    var sat = (typeof curveSaturation === 'function') ? curveSaturation() : null;
    if (sat && sat.budget) {
      parts.push('Чтобы закрыть <b>весь</b> ожидаемый ущерб, достаточно <b>' +
        money(sat.budget) + ' ₽</b>.');
    }
    if (broad) {
      parts.push('Закупка без разбора стоила бы <b>' + money(broad) + ' ₽</b>.');
    }
    node.innerHTML = parts.join(' ');
  }

  /** Ставит слайдер на конкретную сумму и запускает пересчёт, как при перетаскивании. */
  function setBudget(value) {
    var rng = $('rng-budget');
    if (!rng || value === null || value === undefined) return;
    var v = Math.max(Number(rng.min), Math.min(Number(rng.max), Number(value)));
    rng.value = v;
    onBudgetInput();
  }

  function onBudgetInput() {
    S.budget = Number($('rng-budget').value);
    $('budget-num').textContent = fmt(S.budget, 0);
    emit('budget-input');
    renderBudgetContext();
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
        emit('strategies');
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
      // Точка насыщения известна только после кривой, а подпись под слайдером на
      // неё ссылается — поэтому контекст перерисовываем ещё раз.
      renderBudgetContext();
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

  // ── наряд на обследование ────────────────────────────────────────────────
  //
  // Рабочий документ для бригады: объекты в порядке приоритета, где они, что про них
  // известно и попадают ли они в заказанную съёмку. Все числа — из /api/assets и
  // /api/strategies как есть; панель только сортирует и раскладывает по строкам.
  // У объектов без оценки p, ущерб и ранг пустые — ноль здесь был бы ложью.

  function workOrderRows() {
    var features = (S.assets && S.assets.features) || [];
    var covered = coveredSet();
    var plan = {};
    ((S.strategies && S.strategies.plans && S.strategies.plans[S.strategy]) || []).forEach(function (id) { plan[id] = true; });
    var zonesFor = {};
    ((S.candidates && S.candidates.features) || []).forEach(function (z) {
      var zp = z.properties;
      if (!plan[zp.candidate_id]) return;
      (zp.covered_asset_ids || []).forEach(function (aid) { (zonesFor[aid] = zonesFor[aid] || []).push(zp.candidate_id); });
    });
    var rows = features.map(function (f) {
      var p = f.properties, c = (f.geometry && f.geometry.coordinates) || [];
      var ok = p.status === 'ok';
      return {
        rank: ok ? toNum(p.rank) : null,
        id: p.asset_id,
        type: ASSET_TITLES[p.asset_class] || p.asset_class || '',
        lon: toNum(c[0]), lat: toNum(c[1]),
        p: ok ? toNum(p.p_flood) : null,
        loss: ok ? toNum(p.expected_loss_rub) : null,
        unc: ok ? toNum(p.uncertainty) : null,
        status: ASSET_STATUS_WORD[p.status] || p.status || '',
        covered: !!covered[p.asset_id],
        zones: (zonesFor[p.asset_id] || []).join(' '),
        value: toNum(p.asset_value_rub),
        q: toNum(p.vulnerability_coef)
      };
    });
    // Сначала оценённые по рангу, затем без оценки — по номеру: их приоритет неизвестен.
    rows.sort(function (a, b) {
      if (a.rank !== null && b.rank !== null) return a.rank - b.rank;
      if (a.rank !== null) return -1;
      if (b.rank !== null) return 1;
      return String(a.id).localeCompare(String(b.id));
    });
    return rows;
  }

  function orderStamp() {
    var s = S.summary || {};
    return {
      run: s.run_id || '', chip: s.chip_id || '', event: s.event_id || '',
      deadline: s.decision_deadline || '', strategy: S.strategy, budget: S.budget,
      made: new Date().toLocaleString('ru-RU')
    };
  }

  function csvCell(v) {
    if (v === null || v === undefined) return '';
    var t = String(v);
    return /[",\n\r]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
  }

  function downloadWorkOrderCsv() {
    var st = orderStamp();
    var head = ['priority', 'asset_id', 'asset_type', 'lon', 'lat', 'p_flood', 'expected_loss_rub',
      'uncertainty', 'status', 'covered_by_survey', 'survey_zones', 'asset_value_rub', 'vulnerability_coef'];
    var lines = [head.join(',')];
    workOrderRows().forEach(function (r) {
      lines.push([r.rank, r.id, r.type,
        r.lon === null ? null : r.lon.toFixed(6), r.lat === null ? null : r.lat.toFixed(6),
        r.p === null ? null : r.p.toFixed(4), r.loss === null ? null : r.loss.toFixed(2),
        r.unc === null ? null : r.unc.toFixed(3), r.status,
        r.covered ? 'yes' : 'no', r.zones, r.value, r.q].map(csvCell).join(','));
    });
    // BOM — чтобы Excel открыл кириллицу в UTF-8, а не в кодировке системы.
    var blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'vodopol_naryad_' + (st.run || 'run') + '_' + st.strategy + '.csv';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  function openWorkOrderPrint() {
    var st = orderStamp();
    var rows = workOrderRows();
    var h = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Наряд на обследование — ' + esc(st.chip) + '</title><style>' +
      'body{font:12px/1.4 system-ui,"Segoe UI",Arial,sans-serif;color:#111;margin:18mm 14mm}' +
      'h1{font-size:18px;margin:0 0 4px}.sub{color:#555;margin:0 0 12px}' +
      'dl{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;margin:0 0 14px}dt{color:#555}dd{margin:0;font-weight:600}' +
      'table{width:100%;border-collapse:collapse;font-size:11.5px}th,td{border:1px solid #bbb;padding:4px 6px;vertical-align:top}' +
      'th{background:#f1f1f1;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}' +
      'td.done{width:70px}tr.nocover td{background:#fff8e6}.note{color:#444;font-size:10.5px;margin:10px 0 0}' +
      '.sign{display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;margin-top:28px}.sign div{border-top:1px solid #333;padding-top:3px;color:#555;font-size:10.5px}' +
      '@media print{body{margin:10mm}button{display:none}}</style></head><body>' +
      '<button onclick="print()" style="float:right">Печать</button>' +
      '<h1>Наряд на обследование</h1><p class="sub">Водополь · оценка ущерба от паводка · документ сформирован ' + esc(st.made) + '</p>' +
      '<dl><dt>запуск</dt><dd>' + esc(st.run) + '</dd><dt>чип / событие</dt><dd>' + esc(st.chip) + ' · ' + esc(st.event) + '</dd>' +
      '<dt>срок решения</dt><dd>' + esc(st.deadline) + ' (сценарная дата)</dd>' +
      '<dt>стратегия съёмки</dt><dd>' + esc(STRATEGY_TITLE[st.strategy] || st.strategy) + '</dd>' +
      '<dt>бюджет решения</dt><dd>' + money(st.budget) + ' ₽</dd></dl>' +
      '<table><thead><tr><th>№</th><th>объект</th><th>координаты, °<br>долгота, широта</th><th>p затопления</th>' +
      '<th>ожидаемый ущерб, ₽</th><th>неопр.</th><th>статус</th><th>покрыт съёмкой</th><th>отметка бригады</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      h += '<tr class="' + (r.covered ? '' : 'nocover') + '"><td class="n">' + (r.rank === null ? '—' : r.rank) + '</td>' +
        '<td><b>' + esc(r.id) + '</b> · ' + esc(r.type) + '</td>' +
        '<td class="n">' + (r.lon === null ? '—' : r.lon.toFixed(6) + ', ' + r.lat.toFixed(6)) + '</td>' +
        '<td class="n">' + (r.p === null ? '—' : fmt(r.p, 4)) + '</td>' +
        '<td class="n">' + (r.loss === null ? '—' : money(r.loss)) + '</td>' +
        '<td class="n">' + (r.unc === null ? '—' : fmt(r.unc, 3)) + '</td>' +
        '<td>' + esc(r.status) + '</td>' +
        '<td>' + (r.covered ? 'да' + (r.zones ? '<br><small>' + esc(r.zones) + '</small>' : '') : 'нет — только наземно') + '</td>' +
        '<td class="done"></td></tr>';
    });
    h += '</tbody></table>' +
      '<p class="note">Порядок — по рангу ожидаемого ущерба: EL = p × V × q, p — выход модели по вероятностному растру, ' +
      'V и q заданы условием кейса. Объекты без оценки идут в конце, их p, ущерб и ранг пустые — это «неизвестно», а не ноль. ' +
      'Отметка «покрыт съёмкой» относится к выбранной стратегии и считается по сценарной ставке съёмки: при другой ставке состав заказа может измениться. ' +
      'Строки, не покрытые съёмкой, выделены — по ним подтверждение возможно только на месте.</p>' +
      '<div class="sign"><div>выдал</div><div>принял, бригадир</div><div>дата обследования</div></div>' +
      '</body></html>';
    var w = window.open('', '_blank');
    if (!w) { banner('Браузер заблокировал новое окно: разрешите всплывающие окна для этой страницы или выгрузите наряд в CSV.', 'warn'); return; }
    w.document.open(); w.document.write(h); w.document.close();
  }

  // ── заявка на съёмку ─────────────────────────────────────────────────────
  //
  // Корзина выбранной стратегии как документ заказа — то, что обсуждают на
  // технико-экономическом заседании. Все цифры — позиции из /api/strategies
  // (procurement_plan.csv) и свойства зон из /api/candidates, как есть. Итог берётся из
  // сравнения стратегий, а не складывается в панели: скидка Р считается на корзину
  // целиком, и сумма заранее посчитанных цен была бы другим числом.

  var SENSOR_WORD = { sar: 'радар (SAR)', optical: 'оптика' };
  var ACQ_WORD = { new: 'новая съёмка', operational: 'оперативный архив', archive: 'архив' };
  var ROLE_WORD = { event_observation: 'наблюдение события', context: 'контекст до события' };
  var USAGE_WORD = { internal: 'внутреннее', limited: 'ограниченное', unrestricted: 'без ограничений' };

  function zoneRing(feature) {
    var g = feature && feature.geometry;
    if (!g) return '';
    var ring = g.type === 'Polygon' ? g.coordinates[0] : g.type === 'MultiPolygon' ? g.coordinates[0][0] : [];
    var seen = {}, out = [];
    ring.forEach(function (c) {
      var k = c[0].toFixed(5) + ' ' + c[1].toFixed(5);
      if (!seen[k]) { seen[k] = true; out.push(k); }
    });
    return out.join('; ');
  }

  function openProcurementRequest() {
    var st = orderStamp();
    var s = S.summary || {};
    var positions = ((S.strategies && S.strategies.positions) || []).filter(function (p) { return p.strategy === S.strategy; });
    var cmp = comparisonRow(S.strategy) || {};
    var byId = {};
    ((S.candidates && S.candidates.features) || []).forEach(function (f) { byId[f.properties.candidate_id] = f; });
    var feasible = cmp.budget_feasible === true;
    var counterfactual = !feasible && S.strategy === 'B';
    var rateSource = positions.length && byId[positions[0].candidate_id] ? byId[positions[0].candidate_id].properties.base_rate_source : '';

    var h = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Заявка на съёмку — ' + esc(st.chip) + ' — ' + esc(st.strategy) + '</title><style>' +
      'body{font:12px/1.4 system-ui,"Segoe UI",Arial,sans-serif;color:#111;margin:16mm 12mm}' +
      'h1{font-size:18px;margin:0 0 4px}.sub{color:#555;margin:0 0 12px}h2{font-size:13px;margin:16px 0 6px}' +
      'dl{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;margin:0 0 10px}dt{color:#555}dd{margin:0;font-weight:600}' +
      '.stamp{border:2px solid #b42318;color:#b42318;padding:6px 10px;font-weight:700;margin:8px 0;display:inline-block}' +
      '.scen{background:#fff4e0;border-left:3px solid #d98a00;padding:6px 10px;margin:8px 0;font-size:11px}' +
      'table{width:100%;border-collapse:collapse;font-size:11px}th,td{border:1px solid #bbb;padding:4px 5px;vertical-align:top}' +
      'th{background:#f1f1f1;text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}' +
      'td.geo{font-size:9.5px;color:#444;max-width:180px}tfoot td{font-weight:700;background:#f7f7f7}' +
      '.note{color:#444;font-size:10.5px;margin:10px 0 0}.sign{display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;margin-top:26px}' +
      '.sign div{border-top:1px solid #333;padding-top:3px;color:#555;font-size:10.5px}@media print{body{margin:9mm}button{display:none}}' +
      '</style></head><body><button onclick="print()" style="float:right">Печать</button>' +
      '<h1>Заявка на дополнительную съёмку</h1><p class="sub">Водополь · обоснование заказа ДЗЗ · документ сформирован ' + esc(st.made) + '</p>' +
      '<dl><dt>запуск</dt><dd>' + esc(st.run) + '</dd><dt>чип / событие</dt><dd>' + esc(st.chip) + ' · ' + esc(st.event) + '</dd>' +
      '<dt>срок решения</dt><dd>' + esc(st.deadline) + ' (сценарная дата)</dd>' +
      '<dt>стратегия</dt><dd>' + esc(STRATEGY_TITLE[st.strategy] || st.strategy) + '</dd>' +
      '<dt>объявленный бюджет</dt><dd>' + money(st.budget) + ' ₽</dd>' +
      '<dt>правовая основа</dt><dd>' + esc(s.legal_edition || '') + ', РП = Б × К × О × П × Т × Р</dd></dl>';
    if (counterfactual) {
      h += '<div class="stamp">КОНТРФАКТИЧЕСКАЯ: стоимость выше бюджета, к исполнению не предлагается</div>';
    }
    h += '<div class="scen"><b>Ставка Б сценарная.</b> ' + esc(rateSource || 'Подтверждённый действующий размер БРЕ не установлен.') +
      ' Цены ниже — расчёт по этой ставке, а не коммерческое предложение поставщика; даты съёмки и доступности тоже сценарные.</div>';

    if (!positions.length) {
      h += '<p>В стратегии ' + esc(st.strategy) + ' платных позиций нет: решение принимается по открытым данным.</p>';
    } else {
      h += '<h2>Позиции заказа</h2><table><thead><tr><th>№</th><th>зона</th><th>контур, долгота широта</th><th>данные</th>' +
        '<th>обработка / использование</th><th>съёмка / доступно</th><th>Б, ₽/км²</th><th>К, км²</th><th>О</th><th>П</th><th>Т</th><th>Р</th>' +
        '<th>цена за км², ₽</th><th>цена позиции, ₽</th></tr></thead><tbody>';
      positions.forEach(function (p, i) {
        var z = byId[p.candidate_id], zp = z ? z.properties : {};
        h += '<tr><td class="n">' + (i + 1) + '</td><td><b>' + esc(p.candidate_id) + '</b>' +
          (zp.covered_asset_ids && zp.covered_asset_ids.length ? '<br><small>объекты: ' + esc(zp.covered_asset_ids.join(', ')) + '</small>' : '') + '</td>' +
          '<td class="geo">' + esc(zoneRing(z)) + '</td>' +
          '<td>' + esc(SENSOR_WORD[zp.sensor_type] || zp.sensor_type || '') + ', ' + fmt(zp.resolution_m, 1) + ' м<br><small>' +
          esc(ACQ_WORD[zp.acquisition_type] || zp.acquisition_type || '') + ' · ' + esc(ROLE_WORD[zp.data_role] || zp.data_role || '') + '</small></td>' +
          '<td>' + esc(p.processing_level || '') + ' / ' + esc(USAGE_WORD[p.usage_type] || p.usage_type || '') +
          (p.guaranteed_purchase ? '<br><small>гарантированный выкуп</small>' : '') + '</td>' +
          '<td>' + esc(zp.observation_at || '') + '<br>' + esc(zp.available_at || '') + '</td>' +
          '<td class="n">' + money(p.base_rate_rub_km2) + '</td><td class="n">' + fmt(p.area_km2, 3) + '</td>' +
          '<td class="n">' + fmt(p.processing_coef, 2) + '</td><td class="n">' + fmt(p.usage_coef, 2) + '</td>' +
          '<td class="n">' + fmt(p.freshness_coef, 2) + '</td><td class="n">' + fmt(p.discount_coef, 2) + '</td>' +
          '<td class="n">' + money(p.unit_price_rub_km2) + '</td><td class="n">' + money(p.cost_rub) + '</td></tr>';
      });
      h += '</tbody><tfoot><tr><td colspan="13">итого по заявке, стоимость данных</td><td class="n">' + money(cmp.data_cost_rub) + '</td></tr>' +
        '<tr><td colspan="13">прочие затраты</td><td class="n">' + money(cmp.other_cost_rub) + '</td></tr>' +
        '<tr><td colspan="13">полная стоимость решения · бюджет ' + money(st.budget) + ' ₽ · ' + (feasible ? 'укладывается' : 'выше бюджета') + '</td><td class="n">' + money(cmp.decision_cost_rub) + '</td></tr></tfoot></table>';
      h += '<p class="note">Покрытие корзиной: ' + money(cmp.covered_expected_loss_rub) + ' ₽ ожидаемого ущерба, ' + pct(cmp.coverage_share) +
        ' портфеля. Скидка Р считается на площадь группы корзины целиком, поэтому цена позиции зависит от состава заявки: ' +
        'убрать или добавить зону — значит пересчитать все позиции, а не вычесть одну строку. Итог взят из расчёта сервиса, а не сложен в документе.</p>';
    }
    h += '<div class="sign"><div>подготовил</div><div>согласовал</div><div>утвердил</div></div></body></html>';
    var w = window.open('', '_blank');
    if (!w) { banner('Браузер заблокировал новое окно: разрешите всплывающие окна для этой страницы.', 'warn'); return; }
    w.document.open(); w.document.write(h); w.document.close();
  }

  // ── события интерфейса ───────────────────────────────────────────────────

  function wire() {
    $('rng-budget').addEventListener('input', onBudgetInput);
    $('btn-order-csv').addEventListener('click', downloadWorkOrderCsv);
    $('btn-order-print').addEventListener('click', openWorkOrderPrint);
    $('btn-request').addEventListener('click', openProcurementRequest);
    // Две опорные точки бюджета, на которые чаще всего хотят посмотреть: сколько
    // нужно, чтобы вопросов не осталось, и во что обошлась бы закупка без разбора.
    var jumpFull = $('btn-budget-full');
    if (jumpFull) jumpFull.addEventListener('click', function () {
      var sat = curveSaturation();
      if (sat && sat.budget) setBudget(sat.budget);
    });
    var jumpBroad = $('btn-budget-broad');
    if (jumpBroad) jumpBroad.addEventListener('click', function () {
      var b = comparisonRow('B');
      var cost = toNum(b && b.decision_cost_rub);
      if (cost) setBudget(cost);
    });

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

  // ── выбор комплекта ──────────────────────────────────────────────────────

  /**
   * Наполняет список чипов. Рядом с идентификатором пишем роль события в
   * протоколе: «обучение» или «независимая проверка» — это единственное, что
   * оператору действительно нужно знать, выбирая, на чём смотреть результат.
   */
  function renderRunPicker() {
    var select = $('run-select');
    if (!select) return;
    getJSON('/api/runs').then(function (data) {
      var runs = (data && data.runs) || [];
      // Обзорная карта помечает кадры, по которым комплект уже собран, поэтому
      // список нужен не только этому выпадающему меню.
      S.runs = runs;
      if (!runs.length) {
        select.innerHTML = '<option value="">комплектов нет</option>';
        select.disabled = true;
        return;
      }
      var chosen = currentRunId() || (data && data.current) || '';
      select.innerHTML = runs.map(function (r) {
        var label = (r.chip_id || r.run_id);
        if (r.observation_date) label += ' · ' + r.observation_date;
        if (r.part) label += ' · ' + r.part;
        // Цель пишем только когда она не основная: иначе она повторялась бы в
        // каждой строке и ничего не различала.
        if (r.purpose && r.purpose !== 'response') label += ' · ' + (r.purpose_title || r.purpose);
        return '<option value="' + esc(r.run_id) + '"' +
          (r.run_id === chosen ? ' selected' : '') + '>' + esc(label) + '</option>';
      }).join('');
      select.disabled = false;
      select.title = 'Комплект запуска. Смена чипа перечитывает готовые файлы; ' +
        'модель при этом не переобучается.';
    }).catch(function () {
      select.innerHTML = '<option value="">список недоступен</option>';
      select.disabled = true;
    });

    select.addEventListener('change', function () {
      var value = select.value;
      if (!value) return;
      // Перезагружаем страницу с новым параметром вместо точечного обновления:
      // у панели полтора десятка связанных представлений, слои карты и два холста,
      // и выборочный сброс любого из них — это тихий рассинхрон чисел на экране.
      window.location.search = '?run=' + encodeURIComponent(value);
    });
  }

  // ── сборка комплекта ─────────────────────────────────────────────────────

  var BUILD = { chips: [], writable: false, timer: null, jobId: '' };

  function buildState(text, kind) {
    var node = $('build-state');
    if (!node) return;
    node.className = 'buildstate' + (kind ? ' ' + kind : '');
    node.textContent = text || '';
  }

  /** Список чипов в выпадающий список окна, с учётом выбранного события. */
  function fillChipList() {
    var event = $('build-event').value;
    var list = $('build-chip-list');
    var items = BUILD.chips.filter(function (c) { return !event || c.event_id === event; });
    list.innerHTML = items.map(function (c) {
      var note = c.part || '';
      if (c.run_id) note += (note ? ', ' : '') + 'комплект уже собран';
      return '<option value="' + esc(c.chip_id) + '"' +
        (note ? ' label="' + esc(note) + '"' : '') + '></option>';
    }).join('');
    $('build-chip-hint').textContent = 'доступно чипов: ' + items.length;
  }

  /** Подсказка под полем: роль события и то, что комплект уже есть. */
  function describeChip() {
    var value = ($('build-chip').value || '').trim();
    var hint = $('build-chip-hint');
    if (!value) { fillChipList(); return; }
    var found = null;
    for (var i = 0; i < BUILD.chips.length; i++) {
      if (BUILD.chips[i].chip_id === value) { found = BUILD.chips[i]; break; }
    }
    if (!found) {
      hint.textContent = 'такого чипа нет среди выкачанных';
      return;
    }
    var parts = [found.event_id];
    if (found.part) parts.push(found.part);
    if (found.run_id) parts.push('комплект уже собран, сборка перезапишет его свежим');
    hint.textContent = parts.join(' · ');
  }

  function loadChips() {
    return getJSON('/api/chips').then(function (data) {
      BUILD.chips = (data && data.chips) || [];
      BUILD.writable = !!(data && data.writable);
      var events = [];
      BUILD.chips.forEach(function (c) {
        if (events.indexOf(c.event_id) < 0) events.push(c.event_id);
      });
      events.sort();
      $('build-event').innerHTML = '<option value="">все события</option>' +
        events.map(function (e) { return '<option value="' + esc(e) + '">' + esc(e) + '</option>'; }).join('');
      fillChipList();

      if (!BUILD.writable) {
        // Честно гасим кнопку вместо падения на середине прогона: каталог
        // комплектов смонтирован только на чтение, писать результат некуда.
        $('btn-build-start').disabled = true;
        buildState(data.reason || 'сборка здесь недоступна', 'bad');
      }
      if (data && data.busy && data.busy.status === 'running') {
        watchBuild(data.busy.job_id);
      }
    });
  }

  function appendLog(lines) {
    var log = $('build-log');
    log.classList.remove('hidden');
    log.textContent = (lines || []).join('\n');
    log.scrollTop = log.scrollHeight;
  }

  /** Опрос хода сборки. Секунда — прогон идёт минуту, чаще спрашивать незачем. */
  function watchBuild(jobId) {
    BUILD.jobId = jobId;
    $('btn-build-start').disabled = true;
    buildState('идёт сборка…');
    if (BUILD.timer) clearInterval(BUILD.timer);
    BUILD.timer = setInterval(function () {
      getJSON('/api/build/' + encodeURIComponent(jobId)).then(function (job) {
        appendLog(job.lines);
        if (job.status === 'running') {
          buildState('идёт сборка, ' + Math.round(job.elapsed_sec) + ' с');
          return;
        }
        clearInterval(BUILD.timer);
        BUILD.timer = null;
        $('btn-build-start').disabled = !BUILD.writable;
        if (job.status === 'done') {
          buildState('готово за ' + Math.round(job.elapsed_sec) + ' с, открываю…', 'ok');
          window.location.search = '?run=' + encodeURIComponent(job.run_id);
        } else {
          buildState(job.error || 'сборка не удалась', 'bad');
        }
      }).catch(function (err) {
        clearInterval(BUILD.timer);
        BUILD.timer = null;
        $('btn-build-start').disabled = !BUILD.writable;
        buildState('связь со сборкой потеряна: ' + err.message, 'bad');
      });
    }, 1000);
  }

  function startBuild() {
    var chip = ($('build-chip').value || '').trim();
    var budget = Number($('build-budget').value);
    if (!chip) { buildState('выберите чип', 'bad'); return; }
    if (!isFinite(budget) || budget < 0) { buildState('бюджет должен быть числом от нуля', 'bad'); return; }
    $('btn-build-start').disabled = true;
    buildState('запускаю…');
    $('build-log').textContent = '';
    fetch(withRun('/api/build'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ chip_id: chip, budget_rub: budget })
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) throw new Error(body.error || body.detail || ('HTTP ' + r.status));
        return body;
      });
    }).then(function (job) {
      watchBuild(job.job_id);
    }).catch(function (err) {
      $('btn-build-start').disabled = !BUILD.writable;
      buildState(err.message, 'bad');
    });
  }

  function wireBuild() {
    var open = $('btn-build-open');
    var dialog = $('build-dialog');
    if (!open || !dialog) return;
    open.addEventListener('click', function () {
      buildState('');
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else dialog.setAttribute('open', 'open');
      if (!BUILD.chips.length) {
        loadChips().catch(function (err) {
          buildState('каталог чипов недоступен: ' + err.message, 'bad');
        });
      }
    });
    $('build-event').addEventListener('change', fillChipList);
    $('build-chip').addEventListener('input', describeChip);
    $('btn-build-start').addEventListener('click', startBuild);
  }

  // ── старт ────────────────────────────────────────────────────────────────

  function boot() {
    wire();
    wireBuild();
    renderRunPicker();
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

      var bundleLink = $('bundle-link');
      if (bundleLink) bundleLink.href = withRun('/api/bundle.zip');

      renderHeader();
      renderMeta();
      renderImpact();
      initMap();
      renderZones();
      renderAssets();
      renderStrategies();
      renderSensitivity();
      setupBudget();
      buildCurve();
      syncLayers();
      emit('boot');
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
      var picker = $('run-select');
      if (picker) picker.disabled = true;
      var ctl = $('mapctl');
      if (ctl) {
        var inputs = ctl.querySelectorAll('input');
        for (var i = 0; i < inputs.length; i++) inputs[i].disabled = true;
      }
      var budget = $('rng-budget');
      if (budget) budget.disabled = true;
      var reset = $('btn-budget-reset');
      if (reset) reset.disabled = true;
      ['btn-order-csv', 'btn-order-print', 'btn-request'].forEach(function (id) { if ($(id)) $(id).disabled = true; });

      // Шапка иначе навсегда остаётся в состоянии «загрузка…».
      $('runline').innerHTML = '<span class="muted">комплект не загружен</span>';
      emit('error');
    });
  }

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (S.curve) drawCurve();
      renderTimeline();
      // Карта пересобирает размеры сама, но подогнать кадр под новую площадь
      // она не догадается — иначе после разворота окна снимок остаётся мелким.
      if (S.map) { S.map.invalidateSize(); fitChip(); }
    }, 150);
  });

  function emit(reason) {
    document.dispatchEvent(new CustomEvent('vp:update', { detail: reason }));
  }

  // Интерфейс для pages.js: страницы из макета дорисовываются там, но работают
  // на тех же данных и тех же функциях, что и проверенные блоки этого файла.
  window.VODOPOL = {
    S: S, $: $, esc: esc, toNum: toNum, fmt: fmt, money: money, pct: pct, mark: mark,
    getJSON: getJSON, withRun: withRun, currentRunId: currentRunId,
    setStrategy: setStrategy, setBudget: setBudget, comparisonRow: comparisonRow,
    selectedSet: selectedSet, coveredSet: coveredSet, priorityTier: priorityTier,
    rankedCount: rankedCount, fitChip: fitChip, showZoneCard: showZoneCard, restyleZones: restyleZones,
    assetPopup: assetPopup, fullPurchaseCost: fullPurchaseCost, plural: plural,
    ASSET_TITLES: ASSET_TITLES, SENSOR_WORD: SENSOR_WORD, ACQ_WORD: ACQ_WORD,
    ROLE_WORD: ROLE_WORD, USAGE_WORD: USAGE_WORD, STRATEGY_TITLE: STRATEGY_TITLE,
    PRIORITY_TIERS: PRIORITY_TIERS, DASH: DASH
  };

  document.addEventListener('DOMContentLoaded', boot);
})();
