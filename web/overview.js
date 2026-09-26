/**
 * Обзорная карта: весь набор сразу и Россия на одном экране.
 *
 * Зачем отдельная карта. Основная карта показывает один кадр 512×512 — это
 * рабочий экран оператора, и подробность там нужна. Но человек, который видит
 * панель впервые, первым делом спрашивает не «какая вероятность в этом пикселе»,
 * а «где это всё вообще и при чём тут мы». Ответ на такой вопрос — обзор, и
 * открываться он должен сразу, без единого клика.
 *
 * Про Россию, честно и прямо. В официальном наборе кейса российских событий нет:
 * Sen1Floods11 собран по одиннадцати катастрофическим паводкам 2016–2019 годов в
 * Азии, Африке, Америке и Испании. Поэтому карта не рисует несуществующих данных.
 * Она показывает две вещи рядом: где у нас есть размеченные кадры и где решение
 * нужно применять. Sentinel-1 снимает Россию тем же радаром и с тем же шагом
 * сетки, что и территории набора, а вход у модели — два канала в децибелах,
 * поэтому подставляется российский снимок тем же флагом --s1-file. Это проверено
 * прогоном, а не обещано.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var MAP = null;
  var CHIPS = null;
  var LAYERS = {};

  /** Цвет кадра по его роли в протоколе — тот же смысл, что в списке комплектов. */
  var PART_STYLE = {
    train: { color: '#6b7280', title: 'обучение' },
    validation: { color: '#2563eb', title: 'настройка порога' },
    test: { color: '#c0392b', title: 'независимая проверка' },
    holdout: { color: '#7c3aed', title: 'отложенное событие' },
    '': { color: '#9ca3af', title: 'вне протокола' }
  };

  /**
   * Российские территории, где половодье и паводки повторяются.
   *
   * Это не реестр и не прогноз: список иллюстративный, собран по крупным событиям
   * последних лет, о которых знает любой, кто следит за темой. Данных по ним у нас
   * нет и мы их не рисуем — точка на карте означает «сюда решение применяется»,
   * а не «здесь что-то посчитано». Разница принципиальная, и в подписи она есть.
   */
  var RU_SPOTS = [
    { name: 'Тулун, Иркутская область', lat: 54.56, lon: 100.58, note: 'Ия, 2019' },
    { name: 'Благовещенск, Амурская область', lat: 50.29, lon: 127.53, note: 'Амур, 2013' },
    { name: 'Комсомольск-на-Амуре', lat: 50.55, lon: 137.01, note: 'Амур, 2013' },
    { name: 'Орск, Оренбургская область', lat: 51.20, lon: 58.57, note: 'Урал, 2024' },
    { name: 'Оренбург', lat: 51.77, lon: 55.10, note: 'Урал, 2024' },
    { name: 'Курган', lat: 55.44, lon: 65.34, note: 'Тобол, 2024' },
    { name: 'Ишим, Тюменская область', lat: 56.11, lon: 69.49, note: 'Ишим, 2024' },
    { name: 'Ленск, Якутия', lat: 60.73, lon: 114.90, note: 'Лена, заторы льда' },
    { name: 'Якутск', lat: 62.03, lon: 129.73, note: 'Лена, заторы льда' },
    { name: 'Великий Устюг, Вологодская область', lat: 60.76, lon: 46.31, note: 'Сухона и Юг, заторы' },
    { name: 'Крымск, Краснодарский край', lat: 44.93, lon: 37.99, note: 'Адагум, 2012' },
    { name: 'Уссурийск, Приморский край', lat: 43.80, lon: 131.95, note: 'Раздольная' }
  ];

  /** Рамка России целиком: по ней карта показывает страну одним движением. */
  var RU_BOUNDS = [[41.2, 19.6], [77.0, 190.0]];

  function chipBounds(b) { return [[b[1], b[0]], [b[3], b[2]]]; }

  function esc(v) {
    return String(v === null || v === undefined ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /** Комплекты, которые уже собраны: их кадры на обзоре выделяются. */
  function builtByChip() {
    var out = {};
    var runs = (window.VODOPOL && window.VODOPOL.S && window.VODOPOL.S.runs) || [];
    runs.forEach(function (r) { if (r.chip_id) out[r.chip_id] = r; });
    return out;
  }

  function ensureMap() {
    if (MAP) return MAP;
    MAP = L.map('overview-map', {
      preferCanvas: true,
      zoomControl: true,
      minZoom: 2,
      zoomSnap: 0,
      worldCopyJump: true
    });
    MAP.attributionControl.setPrefix(
      '<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 12,
      attribution: '© OpenStreetMap · кадры Sen1Floods11'
    }).addTo(MAP);
    MAP.setView([30, 60], 2);
    return MAP;
  }

  function drawChips() {
    var map = ensureMap();
    var built = builtByChip();
    var group = L.layerGroup().addTo(map);
    LAYERS.chips = group;

    var all = [];
    CHIPS.forEach(function (c) {
      var style = PART_STYLE[c.part] || PART_STYLE[''];
      var isBuilt = !!built[c.chip_id];
      var rect = L.rectangle(chipBounds(c.bounds), {
        color: style.color,
        weight: isBuilt ? 2 : 0.6,
        opacity: isBuilt ? 1 : 0.65,
        fillColor: style.color,
        fillOpacity: isBuilt ? 0.35 : 0.12
      });
      rect.bindTooltip(
        '<b>' + esc(c.chip_id) + '</b><br>' + esc(style.title) +
        (isBuilt ? '<br>комплект собран — нажмите, чтобы открыть'
                 : '<br>комплект не собран — нажмите, чтобы собрать'),
        { sticky: true }
      );
      rect.on('click', function () { pickChip(c, built[c.chip_id]); });
      rect.addTo(group);
      all.push(rect.getBounds());

      // Кадр — это 0,046° по стороне, на обзорном зуме он меньше пикселя и просто
      // не виден. Поэтому рядом с точным прямоугольником рисуется точка с
      // постоянным размером в пикселях: она показывает, где кадр есть, а
      // прямоугольник показывает, какой он, когда карту приближают.
      var dot = L.circleMarker(rect.getBounds().getCenter(), {
        radius: isBuilt ? 5 : 3,
        color: style.color,
        weight: isBuilt ? 2 : 1,
        fillColor: style.color,
        fillOpacity: isBuilt ? 0.95 : 0.55
      });
      dot.bindTooltip(
        '<b>' + esc(c.chip_id) + '</b><br>' + esc(style.title) +
        (isBuilt ? '<br>комплект собран — нажмите, чтобы открыть'
                 : '<br>комплект не собран — нажмите, чтобы собрать'),
        { sticky: true }
      );
      dot.on('click', function () { pickChip(c, built[c.chip_id]); });
      dot.addTo(group);
    });
    LAYERS.allBounds = all;
  }

  function drawRussia() {
    var map = ensureMap();
    var group = L.layerGroup().addTo(map);
    LAYERS.russia = group;
    // Рамка страны: без неё дюжина точек читается как случайная россыпь, а не как
    // «вот территория, ради которой всё это». Линия пунктирная и без заливки —
    // это указание области применения, а не данные.
    L.rectangle(RU_BOUNDS, {
      color: '#0f766e',
      weight: 1.4,
      opacity: 0.7,
      dashArray: '6 5',
      fill: false,
      interactive: false
    }).addTo(group);

    RU_SPOTS.forEach(function (s) {
      L.circleMarker([s.lat, s.lon], {
        radius: 7,
        color: '#0f766e',
        weight: 2,
        fillColor: '#14b8a6',
        fillOpacity: 0.9
      }).bindTooltip(
        '<b>' + esc(s.name) + '</b><br>' + esc(s.note) +
        '<br><i>размеченных данных набора здесь нет</i>',
        { sticky: true }
      ).addTo(group);
    });
  }

  /** Нажатие на кадр: открыть готовый комплект или предложить собрать. */
  function pickChip(chip, run) {
    if (run && run.run_id) {
      window.location.search = '?run=' + encodeURIComponent(run.run_id);
      return;
    }
    var dialog = $('build-dialog');
    var field = $('build-chip');
    if (!dialog || !field) return;
    if ($('btn-build-open')) $('btn-build-open').click();
    field.value = chip.chip_id;
    field.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function fit(which) {
    var map = ensureMap();
    if (which === 'ru') { map.fitBounds(RU_BOUNDS, { padding: [10, 10] }); return; }
    if (which === 'chip') {
      var b = (window.VODOPOL && window.VODOPOL.S && window.VODOPOL.S.summary || {}).bounds;
      if (b) { map.fitBounds(chipBounds(b), { padding: [40, 40] }); return; }
    }
    if (LAYERS.allBounds && LAYERS.allBounds.length) {
      var acc = L.latLngBounds(LAYERS.allBounds[0].getSouthWest(), LAYERS.allBounds[0].getNorthEast());
      for (var i = 1; i < LAYERS.allBounds.length; i++) acc.extend(LAYERS.allBounds[i]);
      // Россия в кадр входит всегда, даже когда в ней нет ни одного кадра набора:
      // вопрос «а где тут мы» задают первым, и ответ не должен требовать прокрутки.
      if (which !== 'data') acc.extend(RU_BOUNDS);
      map.fitBounds(acc, { padding: [16, 16] });
    }
  }

  function renderLegend() {
    var node = $('overview-legend');
    if (!node) return;
    var counts = {};
    CHIPS.forEach(function (c) { counts[c.part] = (counts[c.part] || 0) + 1; });
    var html = '';
    ['train', 'validation', 'test', 'holdout'].forEach(function (key) {
      var st = PART_STYLE[key];
      html += '<span class="ov-item"><i style="background:' + st.color + '"></i>' +
        esc(st.title) + ' · ' + (counts[key] || 0) + '</span>';
    });
    html += '<span class="ov-item"><i style="background:#14b8a6;border-radius:50%"></i>' +
      'территории России, где половодье повторяется</span>';
    node.innerHTML = html;
  }

  function renderNote() {
    var node = $('overview-note');
    if (!node) return;
    node.innerHTML =
      '<b>446 размеченных кадров по 11 событиям</b> — это всё, что содержит официальный ' +
      'набор кейса Sen1Floods11: катастрофические паводки 2016–2019 годов в Азии, Африке, ' +
      'Америке и Испании. <b>Российских событий в наборе нет</b>, и рисовать их мы не стали: ' +
      'нарисованные данные — это не данные.' +
      '<br><br>' +
      'Бирюзовым отмечены территории, где половодье повторяется и где решение нужно ' +
      'применять. Данных по ним у нас нет, и точка означает «сюда это ставится», ' +
      'а не «здесь посчитано». Разница принципиальная.' +
      '<br><br>' +
      'Что для этого уже готово: Sentinel-1 снимает Россию тем же радаром, с тем же шагом ' +
      'сетки и теми же каналами VV и VH, что и территории набора. Вход у модели — два ' +
      'канала в децибелах, происхождение файла ей безразлично, и свой снимок подставляется ' +
      'флагом <code>--s1-file</code>. Это не обещание: полный прогон на снимке вне набора ' +
      'проверен и проходит валидатор комплекта десять из десяти. Чего не хватает — ' +
      'размеченных российских сцен, чтобы <i>измерить</i> качество переноса, а не заявить его.';
  }

  /** Публичная точка входа: вызывается роутером при переходе на страницу обзора. */
  function show() {
    ensureMap();
    if (!CHIPS) return;
    setTimeout(function () { MAP.invalidateSize(); fit('all'); }, 60);
  }

  function boot() {
    if (!$('overview-map')) return;
    fetch('chip_index.json?v=1', { headers: { Accept: 'application/json' } })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        CHIPS = (data && data.chips) || [];
        ensureMap();
        drawChips();
        drawRussia();
        renderLegend();
        renderNote();
        $('overview-count').textContent = CHIPS.length;
        setTimeout(function () { MAP.invalidateSize(); fit('all'); }, 80);
      })
      .catch(function (err) {
        var node = $('overview-note');
        if (node) node.textContent = 'Каталог кадров не загрузился: ' + err.message;
      });

    var b;
    if ((b = $('ov-fit-all'))) b.addEventListener('click', function () { fit('data'); });
    if ((b = $('ov-fit-ru'))) b.addEventListener('click', function () { fit('ru'); });
    if ((b = $('ov-fit-chip'))) b.addEventListener('click', function () { fit('chip'); });
  }

  window.VODOPOL_OVERVIEW = { show: show, fit: fit };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
