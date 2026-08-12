/* tfvn-tarot dataset viewer — Raw view + export builder (todo B.5)
 *
 * Zero-toolchain vanilla JS per the B.1 view contract:
 *   window.Views.raw = function (containerEl, app) { ... }
 *
 * Sections:
 *   - Dataset picker: built from GET /api/catalog (kind == "jsonl") mapped to
 *     the /api/rows dataset ids (anchor_readings -> anchor, generated ->
 *     raw_generated, generated_sep -> raw_generated_sep, ifd_scores ->
 *     raw_ifd_scores), plus the synthetic `all_sft` union (no catalog file).
 *   - Lazy JSON block list: GET /api/rows/{id}?page=N&page_size=100 per page —
 *     never more than one page in the DOM, collapsible pretty-printed rows,
 *     copy button per row.
 *   - Whole-file download: plain <a href="/api/export/{id}"> — Content-
 *     Disposition drives the browser filename.
 *   - Export builder: filter form (same params as the rows endpoint) building
 *     a plain <a href="/api/export/{id}?params"> kept in sync on every input.
 *   - Single-record view: click a row id -> GET /api/rows/{dataset_id}/{pk};
 *     pk per dataset: SFT/raw/ifd -> example_id; cards/spines/compact ->
 *     card_id/orientation; vn_upright -> card_id; spreads -> spread_id;
 *     anchor -> anchor_id. 404 -> toast (via app.fetchJSON) + a "not found"
 *     note in the detail panel.
 *
 * All data interpolated through app.esc(); JSON bodies via textContent.
 */
(function () {
  'use strict';

  var PAGE_SIZE = 100;

  /* catalog jsonl basename -> filtering dataset id (the registry uses short
   * ids; the catalog uses file paths). */
  var ID_RENAMES = {
    anchor_readings: 'anchor',
    generated: 'raw_generated',
    generated_sep: 'raw_generated_sep',
    ifd_scores: 'raw_ifd_scores'
  };

  /* synthetic union first; the rest in a stable, readable order. */
  var UNION_DATASETS = ['all_sft'];
  var KNOWN_ORDER = [
    'all_sft', 'filtered_core', 'filtered_bulk', 'anchor', 'cards', 'vn_spine',
    'english_spine', 'vn_upright', 'compact_cards', 'spreads',
    'raw_generated', 'raw_generated_sep', 'raw_ifd_scores'
  ];

  /* datasets whose primary key is card_id + orientation (two rows per card). */
  var CARD_ORIENTATION_IDS = {
    cards: true, vn_spine: true, english_spine: true, compact_cards: true
  };

  /* ------------------------------------------------------------------ */
  /* pure helpers                                                        */
  /* ------------------------------------------------------------------ */

  /* Primary-key path for the single-record endpoint, per dataset kind. */
  function pkPath(datasetId, row) {
    if (CARD_ORIENTATION_IDS[datasetId]) {
      return row.card_id + '/' + row.orientation;
    }
    if (datasetId === 'vn_upright') return String(row.card_id);
    if (datasetId === 'spreads') return String(row.spread_id);
    if (datasetId === 'anchor') return String(row.anchor_id);
    return String(row.example_id);
  }

  /* Short human label under the pk, key-presence driven. */
  function rowTitle(datasetId, row) {
    if (CARD_ORIENTATION_IDS[datasetId]) {
      return (row.name_en ? row.name_en + ' \u2014 ' : '') + row.orientation;
    }
    if (datasetId === 'vn_upright') return row.name_en || String(row.card_id);
    if (datasetId === 'spreads') return row.name_en || String(row.spread_id);
    if (datasetId === 'anchor') {
      return (row.name_en || row.card_id) + ' \u2014 ' + row.orientation;
    }
    var bits = [];
    if (row.task_type) bits.push(row.task_type);
    if (row.spread_name_vi) bits.push(row.spread_name_vi);
    if (row.prompt_slot) bits.push(row.prompt_slot);
    if (typeof row.ifd_score === 'number') bits.push('ifd ' + row.ifd_score.toFixed(3));
    return bits.join(' \u00b7 ');
  }

  /* Catalog artifacts (kind == jsonl) -> ordered dataset ids. */
  function catalogDatasetIds(artifacts) {
    var ids = [];
    var seen = {};
    artifacts.forEach(function (a) {
      if (!a || a.kind !== 'jsonl' || !a.path) return;
      var base = String(a.path).split('/').pop().replace(/\.jsonl$/, '');
      var id = ID_RENAMES[base] || base;
      if (!seen[id]) { seen[id] = true; ids.push(id); }
    });
    UNION_DATASETS.forEach(function (u) {
      if (!seen[u]) { seen[u] = true; ids.unshift(u); }
    });
    ids.sort(function (x, y) {
      var xi = KNOWN_ORDER.indexOf(x);
      var yi = KNOWN_ORDER.indexOf(y);
      return (xi < 0 ? KNOWN_ORDER.length : xi) - (yi < 0 ? KNOWN_ORDER.length : yi);
    });
    return ids;
  }

  /* Clipboard copy with execCommand fallback (file:// / insecure contexts). */
  function fallbackCopy(text, onOk, onErr) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      if (document.execCommand('copy')) onOk();
      else onErr();
    } catch (err) {
      onErr();
    }
    ta.remove();
  }

  function copyText(text, app) {
    function ok() { app.toast('copied', 'ok'); }
    function err() { app.toast('copy failed'); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(ok, function () {
        fallbackCopy(text, ok, err);
      });
    } else {
      fallbackCopy(text, ok, err);
    }
  }

  function pretty(row) { return JSON.stringify(row, null, 2); }

  /* ------------------------------------------------------------------ */
  /* the view                                                            */
  /* ------------------------------------------------------------------ */

  window.Views.raw = function (containerEl, app) {
    var state = { dataset: 'all_sft', page: 1, total: 0, rows: [] };
    var reqToken = 0;

    /* element refs (filled in buildScaffold) */
    var els = {};

    function esc(v) { return app.esc(v); }

    /* ---------------------------------------------------------------- */
    /* DOM scaffolding                                                    */
    /* ---------------------------------------------------------------- */

    function buildScaffold() {
      containerEl.innerHTML =
        '<style>' +
        '.raw-row{display:flex;flex-direction:column;border:1px solid var(--border);' +
        'border-radius:var(--radius);background:var(--bg-panel);margin-bottom:8px;}' +
        '.raw-row-head{display:flex;align-items:center;gap:10px;padding:6px 10px;' +
        'flex-wrap:wrap;}' +
        '.raw-row-sub{flex:1;min-width:120px;font-size:12px;overflow:hidden;' +
        'text-overflow:ellipsis;white-space:nowrap;}' +
        '.raw-toggle{font-size:11px;width:16px;padding:0;text-align:center;}' +
        '.raw-body{margin:0 10px 10px;max-height:320px;}' +
        '.raw-detail{margin-bottom:16px;}' +
        '.raw-detail pre{margin-top:10px;}' +
        '.raw-empty{padding:20px;color:var(--text-dim);text-align:center;}' +
        '.export-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));' +
        'gap:10px;margin-bottom:12px;}' +
        '.export-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}' +
        '.export-url{font-family:var(--mono);font-size:11px;color:var(--text-faint);' +
        'word-break:break-all;}' +
        '</style>' +

        '<div class="panel" style="margin-bottom:16px">' +
        '  <div class="panel-title">Raw view \u2014 dataset</div>' +
        '  <div style="display:flex;align-items:end;gap:12px;flex-wrap:wrap">' +
        '    <div class="field" style="min-width:240px">' +
        '      <label for="raw-dataset">Dataset (jsonl)</label>' +
        '      <select id="raw-dataset"></select>' +
        '    </div>' +
        '    <div class="field">' +
        '      <span class="small faint" id="raw-dataset-info"></span>' +
        '    </div>' +
        '    <a class="btn" id="raw-download-whole" href="#" download>Download whole file</a>' +
        '  </div>' +
        '</div>' +

        '<div class="panel" style="margin-bottom:16px">' +
        '  <div class="panel-title">Export builder \u2014 filtered JSONL</div>' +
        '  <form class="export-form" id="raw-export-form" autocomplete="off">' +
        '    <div class="field"><label for="f-tier">tier</label>' +
        '      <select id="f-tier"><option value="">any</option><option value="core">core</option>' +
        '      <option value="bulk">bulk</option></select></div>' +
        '    <div class="field"><label for="f-task-type">task_type</label>' +
        '      <input id="f-task-type" type="text" placeholder="safety"></div>' +
        '    <div class="field"><label for="f-register">register</label>' +
        '      <input id="f-register" type="text" placeholder="casual"></div>' +
        '    <div class="field"><label for="f-length-band">length_band</label>' +
        '      <input id="f-length-band" type="text"></div>' +
        '    <div class="field"><label for="f-querent-context">querent_context</label>' +
        '      <input id="f-querent-context" type="text"></div>' +
        '    <div class="field"><label for="f-spread-id">spread_id</label>' +
        '      <input id="f-spread-id" type="text"></div>' +
        '    <div class="field"><label for="f-card-id">card_id</label>' +
        '      <input id="f-card-id" type="number" min="0" max="77"></div>' +
        '    <div class="field"><label for="f-orientation">orientation</label>' +
        '      <select id="f-orientation"><option value="">any</option>' +
        '      <option value="upright">upright</option>' +
        '      <option value="reversed">reversed</option></select></div>' +
        '    <div class="field"><label for="f-ifd-min">ifd_min</label>' +
        '      <input id="f-ifd-min" type="number" step="any"></div>' +
        '    <div class="field"><label for="f-ifd-max">ifd_max</label>' +
        '      <input id="f-ifd-max" type="number" step="any"></div>' +
        '    <div class="field" style="grid-column:1/-1"><label for="f-q">q (substring search)</label>' +
        '      <input id="f-q" type="search" placeholder="search question/reading text"></div>' +
        '  </form>' +
        '  <div class="export-actions">' +
        '    <a class="btn btn-primary" id="raw-download-filtered" href="#" download>Download filtered JSONL</a>' +
        '    <span class="export-url" id="raw-export-url"></span>' +
        '  </div>' +
        '</div>' +

        '<div class="panel raw-detail" id="raw-detail" hidden>' +
        '  <div class="panel-title" style="display:flex;align-items:center;gap:10px">' +
        '    <span id="raw-detail-title">Record</span>' +
        '    <span style="flex:1"></span>' +
        '    <button type="button" class="btn-link" id="raw-detail-close">close</button>' +
        '  </div>' +
        '  <pre class="json-block" id="raw-detail-body"></pre>' +
        '</div>' +

        '<div class="panel">' +
        '  <div class="panel-title" style="display:flex;align-items:baseline;gap:10px">' +
        '    <span>Rows</span>' +
        '    <span class="small faint" id="raw-list-info"></span>' +
        '  </div>' +
        '  <div id="raw-list"></div>' +
        '  <div class="pager" id="raw-pager" hidden>' +
        '    <button type="button" class="btn" id="raw-prev">\u2190 prev</button>' +
        '    <span id="raw-page-info" class="mono"></span>' +
        '    <button type="button" class="btn" id="raw-next">next \u2192</button>' +
        '  </div>' +
        '</div>';

      els.dataset = containerEl.querySelector('#raw-dataset');
      els.datasetInfo = containerEl.querySelector('#raw-dataset-info');
      els.downloadWhole = containerEl.querySelector('#raw-download-whole');
      els.exportForm = containerEl.querySelector('#raw-export-form');
      els.downloadFiltered = containerEl.querySelector('#raw-download-filtered');
      els.exportUrl = containerEl.querySelector('#raw-export-url');
      els.detail = containerEl.querySelector('#raw-detail');
      els.detailTitle = containerEl.querySelector('#raw-detail-title');
      els.detailBody = containerEl.querySelector('#raw-detail-body');
      els.detailClose = containerEl.querySelector('#raw-detail-close');
      els.list = containerEl.querySelector('#raw-list');
      els.listInfo = containerEl.querySelector('#raw-list-info');
      els.pager = containerEl.querySelector('#raw-pager');
      els.prev = containerEl.querySelector('#raw-prev');
      els.next = containerEl.querySelector('#raw-next');
      els.pageInfo = containerEl.querySelector('#raw-page-info');
    }

    /* ---------------------------------------------------------------- */
    /* actions                                                           */
    /* ---------------------------------------------------------------- */

    function queryParams() {
      var fields = [
        ['tier', els.exportForm.querySelector('#f-tier').value],
        ['task_type', els.exportForm.querySelector('#f-task-type').value],
        ['register', els.exportForm.querySelector('#f-register').value],
        ['length_band', els.exportForm.querySelector('#f-length-band').value],
        ['querent_context', els.exportForm.querySelector('#f-querent-context').value],
        ['spread_id', els.exportForm.querySelector('#f-spread-id').value],
        ['card_id', els.exportForm.querySelector('#f-card-id').value],
        ['orientation', els.exportForm.querySelector('#f-orientation').value],
        ['ifd_min', els.exportForm.querySelector('#f-ifd-min').value],
        ['ifd_max', els.exportForm.querySelector('#f-ifd-max').value],
        ['q', els.exportForm.querySelector('#f-q').value]
      ];
      var parts = [];
      fields.forEach(function (f) {
        var v = String(f[1]).trim();
        if (v !== '') parts.push(f[0] + '=' + encodeURIComponent(v));
      });
      return parts;
    }

    function exportUrl() {
      var qs = queryParams();
      return '/api/export/' + encodeURIComponent(state.dataset) +
        (qs.length ? '?' + qs.join('&') : '');
    }

    function updateExportLinks() {
      var url = exportUrl();
      els.downloadFiltered.setAttribute('href', url);
      els.exportUrl.textContent = url;
    }

    function updateWholeLink() {
      els.downloadWhole.setAttribute('href', '/api/export/' + encodeURIComponent(state.dataset));
      els.datasetInfo.textContent =
        state.rows.length ? state.dataset + ' \u2014 ' + state.total + ' rows total' : state.dataset;
    }

    function renderDetail(row, notFound, idPath) {
      if (notFound) {
        els.detailTitle.textContent = 'record not found: ' + state.dataset + '/' + idPath;
        els.detailBody.textContent = 'No row with that primary key in this dataset.';
      } else {
        els.detailTitle.textContent =
          'record \u2014 ' + state.dataset + '/' + idPath + ' \u00b7 ' + rowTitle(state.dataset, row);
        els.detailBody.textContent = pretty(row);
      }
      els.detail.hidden = false;
      els.detail.scrollIntoView({ block: 'nearest' });
    }

    /* Single record from the canonical endpoint (/api/rows/{id}/{pk}). */
    function loadRecord(idPath) {
      app.spinner.show(containerEl);
      app.fetchJSON('/api/rows/' + encodeURIComponent(state.dataset) + '/' +
        idPath.split('/').map(encodeURIComponent).join('/'))
        .then(function (row) { renderDetail(row, false, idPath); })
        .catch(function () { renderDetail(null, true, idPath); })
        .then(function () { app.spinner.hide(containerEl); });
    }

    function buildRowBlock(row) {
      var idPath = pkPath(state.dataset, row);
      var label = rowTitle(state.dataset, row);

      var wrap = document.createElement('div');
      wrap.className = 'raw-row';

      var head = document.createElement('div');
      head.className = 'raw-row-head';

      var toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'btn-link raw-toggle';
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-label', 'Expand row');
      toggle.textContent = '\u25b8';

      var idLink = document.createElement('a');
      idLink.className = 'raw-row-id';
      idLink.href = '#';
      idLink.textContent = idPath;
      idLink.title = 'show exact raw record';
      idLink.addEventListener('click', function (ev) {
        ev.preventDefault();
        loadRecord(idPath);
      });

      var sub = document.createElement('span');
      sub.className = 'raw-row-sub muted';
      sub.textContent = label;

      var copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'btn-link';
      copyBtn.textContent = 'copy';
      copyBtn.addEventListener('click', function () { copyText(pretty(row), app); });

      head.appendChild(toggle);
      head.appendChild(idLink);
      head.appendChild(sub);
      head.appendChild(copyBtn);

      var body = document.createElement('pre');
      body.className = 'json-block raw-body';
      body.hidden = true;
      body.textContent = pretty(row);

      toggle.addEventListener('click', function () {
        body.hidden = !body.hidden;
        toggle.setAttribute('aria-expanded', String(!body.hidden));
        toggle.textContent = body.hidden ? '\u25b8' : '\u25be';
      });

      wrap.appendChild(head);
      wrap.appendChild(body);
      return wrap;
    }

    function renderList() {
      els.list.innerHTML = '';
      var frag = document.createDocumentFragment();
      state.rows.forEach(function (row) { frag.appendChild(buildRowBlock(row)); });
      els.list.appendChild(frag);

      var totalPages = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
      els.listInfo.textContent = state.dataset + ' \u00b7 ' + state.total + ' rows \u00b7 page ' +
        state.page + ' of ' + totalPages + ' \u00b7 ' + state.rows.length + ' loaded';
      els.pageInfo.textContent = state.page + ' / ' + totalPages;
      els.prev.disabled = state.page <= 1;
      els.next.disabled = state.page >= totalPages;
      els.pager.hidden = state.total === 0;
      if (state.total === 0) {
        var empty = document.createElement('div');
        empty.className = 'raw-empty';
        empty.textContent = 'No rows match \u2014 this dataset is empty or (raw files) missing on disk.';
        els.list.appendChild(empty);
      }
      updateWholeLink();
    }

    function loadPage(page) {
      var token = ++reqToken;
      app.spinner.show(containerEl);
      var url = '/api/rows/' + encodeURIComponent(state.dataset) +
        '?page=' + page + '&page_size=' + PAGE_SIZE;
      app.fetchJSON(url)
        .then(function (data) {
          if (token !== reqToken) return;
          state.page = data.page || page;
          state.total = data.total || 0;
          state.rows = data.rows || [];
          renderList();
        })
        .catch(function () {
          if (token !== reqToken) return;
          /* app.fetchJSON already toasted; leave a readable empty state */
          state.rows = [];
          state.total = 0;
          renderList();
        })
        .then(function () { app.spinner.hide(containerEl); });
    }

    function selectDataset(datasetId) {
      state.dataset = datasetId;
      els.dataset.value = datasetId;
      els.detail.hidden = true;
      reqToken++;
      state.rows = [];
      state.total = 0;
      els.list.innerHTML = '';
      els.pager.hidden = true;
      updateExportLinks();
      loadPage(1);
    }

    /* ---------------------------------------------------------------- */
    /* init                                                              */
    /* ---------------------------------------------------------------- */

    function pickDatasetFromHash(ids) {
      var qs = location.hash.split('?')[1];
      if (!qs) return null;
      var params = new URLSearchParams(qs);
      var want = params.get('dataset_id');
      if (want && ids.indexOf(want) !== -1) return want;
      return null;
    }

    function renderPicker(ids) {
      els.dataset.innerHTML = '';
      ids.forEach(function (id) {
        var opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        els.dataset.appendChild(opt);
      });
      els.dataset.addEventListener('change', function () {
        selectDataset(els.dataset.value);
      });
    }

    function init() {
      buildScaffold();
      els.prev.addEventListener('click', function () {
        if (state.page > 1) loadPage(state.page - 1);
      });
      els.next.addEventListener('click', function () {
        if (state.rows.length) loadPage(state.page + 1);
      });
      els.detailClose.addEventListener('click', function () { els.detail.hidden = true; });
      els.exportForm.addEventListener('input', updateExportLinks);
      els.exportForm.addEventListener('submit', function (ev) { ev.preventDefault(); });
      updateExportLinks();

      app.spinner.show(containerEl);
      app.fetchJSON('/api/catalog')
        .then(function (data) {
          var artifacts = (data && data.artifacts) || [];
          var ids = catalogDatasetIds(artifacts);
          if (!ids.length) ids = KNOWN_ORDER.slice();
          renderPicker(ids);
          var fromHash = pickDatasetFromHash(ids);
          var initial = fromHash || (ids.indexOf('all_sft') !== -1 ? 'all_sft' : ids[0]);
          state.dataset = initial;
          els.dataset.value = initial;
          updateExportLinks();
          loadPage(1);
        })
        .catch(function () {
          /* catalog failed (toast already shown): fall back to the known set */
          renderPicker(KNOWN_ORDER.slice());
          state.dataset = 'all_sft';
          els.dataset.value = 'all_sft';
          updateExportLinks();
          loadPage(1);
        })
        .then(function () { app.spinner.hide(containerEl); });
    }

    init();
  };
})();
