/* tfvn-tarot SFT dataset explorer (todo B.4)
 *
 * Filterable / searchable / paginated table over GET /api/rows/all_sft with a
 * row-detail panel: question_vi + reading_vi (line breaks preserved), a
 * critique panel rendered ONLY when the row carries a `critique` object
 * (key-presence, never tier-based), matched_pair_id / safety_category /
 * grounding_defect metadata, provenance chips and a raw-JSON tab.
 *
 * Key-presence discipline: rows lacking `ifd_score` render "—", rows lacking
 * a `critique` object render no critique panel. No branch keys off tier.
 */
(function () {
  'use strict';

  var PAGE_SIZES = [20, 50, 100, 200];
  var TIERS = [['', 'all tiers'], ['core', 'core'], ['bulk', 'bulk']];
  var TASK_TYPES = [
    ['', 'all'],
    ['correction', 'correction'],
    ['explanation', 'explanation'],
    ['grounding', 'grounding'],
    ['reading', 'reading'],
    ['safety', 'safety']
  ];
  var REGISTERS = [['', 'all'], ['casual', 'casual'], ['formal', 'formal'], ['warm', 'warm']];
  var LENGTH_BANDS = [['', 'all'], ['ngắn', 'ngắn'], ['đầy_đủ', 'đầy_đủ']];
  var ORIENTATIONS = [['', 'all'], ['upright', 'upright'], ['reversed', 'reversed']];
  var TEXT_FILTERS = [
    ['querent_context', 'querent_context', 'e.g. love, career'],
    ['spread_id', 'spread_id', 'e.g. spread_three'],
    ['card_id', 'card_id', 'card id 0-77'],
    ['q', 'q', 'search question / reading / glosses']
  ];

  window.Views.dataset = function (containerEl, app) {
    var esc = app.esc;

    var state = {
      page: 1,
      pageSize: 50,
      filters: {
        tier: '', task_type: '', register: '', length_band: '',
        querent_context: '', spread_id: '', card_id: '', orientation: '',
        ifd_min: '', ifd_max: '', q: ''
      },
      rows: [],
      total: 0,
      detailRow: null,
      detailTab: 'meaning'
    };

    var el = { form: null, tbody: null, pager: null, pagerTotal: null, detail: null };

    /* ------------------------------------------------------------------ */
    /* small builders                                                      */
    /* ------------------------------------------------------------------ */
    function buildSelect(name, options) {
      var s = document.createElement('select');
      s.name = name;
      options.forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o[0];
        opt.textContent = o[1];
        s.appendChild(opt);
      });
      return s;
    }

    function buildInput(type, name, placeholder, step) {
      var i = document.createElement('input');
      i.type = type;
      i.name = name;
      if (placeholder) i.placeholder = placeholder;
      if (step) i.step = step;
      return i;
    }

    function buildField(labelText, control) {
      var field = document.createElement('div');
      field.className = 'field';
      var label = document.createElement('label');
      label.textContent = labelText;
      field.appendChild(label);
      field.appendChild(control);
      return field;
    }

    /* ------------------------------------------------------------------ */
    /* filter bar                                                          */
    /* ------------------------------------------------------------------ */
    function buildFilterBar() {
      var form = document.createElement('form');
      form.className = 'filter-bar';
      form.setAttribute('aria-label', 'SFT row filters');

      var groups = [
        buildField('Tier', buildSelect('tier', TIERS)),
        buildField('Task type', buildSelect('task_type', TASK_TYPES)),
        buildField('Register', buildSelect('register', REGISTERS)),
        buildField('Length band', buildSelect('length_band', LENGTH_BANDS)),
        buildField('Orientation', buildSelect('orientation', ORIENTATIONS))
      ];
      TEXT_FILTERS.forEach(function (tf) {
        groups.push(buildField(tf[0], buildInput('text', tf[1], tf[2])));
      });
      groups.push(buildField('IFD min', buildInput('number', 'ifd_min', 'any', 'any')));
      groups.push(buildField('IFD max', buildInput('number', 'ifd_max', 'any', 'any')));

      groups.forEach(function (g) { form.appendChild(g); });

      var actions = document.createElement('div');
      actions.className = 'field';
      actions.style.alignSelf = 'end';
      var applyBtn = document.createElement('button');
      applyBtn.type = 'submit';
      applyBtn.className = 'btn btn-primary';
      applyBtn.textContent = 'Apply filters';
      var resetBtn = document.createElement('button');
      resetBtn.type = 'button';
      resetBtn.className = 'btn';
      resetBtn.textContent = 'Reset';
      resetBtn.setAttribute('data-action', 'reset-filters');
      actions.appendChild(applyBtn);
      actions.appendChild(resetBtn);
      form.appendChild(actions);

      form.addEventListener('submit', function (ev) {
        ev.preventDefault();
        if (!readFilters()) return;
        state.page = 1;
        fetchRows();
      });

      return form;
    }

    /* ------------------------------------------------------------------ */
    /* filter read + client-side validation                                */
    /* ------------------------------------------------------------------ */
    function readFilters() {
      var fd = new FormData(el.form);
      var f = state.filters;
      [
        'tier', 'task_type', 'register', 'length_band',
        'querent_context', 'spread_id', 'card_id', 'orientation',
        'ifd_min', 'ifd_max', 'q'
      ].forEach(function (k) {
        var v = fd.get(k);
        f[k] = typeof v === 'string' ? v.trim() : '';
      });

      var bad = [];
      ['ifd_min', 'ifd_max'].forEach(function (k) {
        if (f[k] !== '') {
          var n = Number(f[k]);
          if (isNaN(n)) bad.push(k);
          else f[k] = String(n);
        }
      });
      if (f.card_id !== '') {
        var ci = Number(f.card_id);
        if (isNaN(ci)) bad.push('card_id');
        else f.card_id = String(Math.floor(ci));
      }
      if (bad.length) {
        app.toast('Invalid value for: ' + bad.join(', ') + ' (numbers only)', 'error');
        return false;
      }
      return true;
    }

    function resetFilters() {
      state.filters = {
        tier: '', task_type: '', register: '', length_band: '',
        querent_context: '', spread_id: '', card_id: '', orientation: '',
        ifd_min: '', ifd_max: '', q: ''
      };
      var inputs = el.form.querySelectorAll('input, select');
      for (var i = 0; i < inputs.length; i++) inputs[i].value = '';
      state.page = 1;
      fetchRows();
    }

    /* ------------------------------------------------------------------ */
    /* query building / fetching                                           */
    /* ------------------------------------------------------------------ */
    function buildQuery() {
      var f = state.filters;
      var params = ['page=' + state.page, 'page_size=' + state.pageSize];
      var map = {
        tier: f.tier, task_type: f.task_type, register: f.register,
        length_band: f.length_band, querent_context: f.querent_context,
        spread_id: f.spread_id, orientation: f.orientation,
        ifd_min: f.ifd_min, ifd_max: f.ifd_max, q: f.q
      };
      if (f.card_id !== '') map.card_id = f.card_id;
      Object.keys(map).forEach(function (k) {
        var v = map[k];
        if (v !== '' && v != null) params.push(k + '=' + encodeURIComponent(v));
      });
      return params.join('&');
    }

    function fetchRows() {
      app.spinner.show(containerEl);
      app.fetchJSON('/api/rows/all_sft?' + buildQuery()).then(function (data) {
        state.rows = Array.isArray(data.rows) ? data.rows : [];
        state.total = typeof data.total === 'number' ? data.total : 0;
        fillTable();
        fillPager();
        state.detailRow = null;
        hideDetail();
      }).catch(function () {
        /* fetchJSON already toasted the failure */
      }).then(function () {
        app.spinner.hide(containerEl);
      });
    }

    /* ------------------------------------------------------------------ */
    /* table rendering                                                     */
    /* ------------------------------------------------------------------ */
    function fmtIfd(row) {
      var s = row.ifd_score;
      if (typeof s !== 'number' || isNaN(s) || !isFinite(s)) {
        return '<span class="faint">\u2014</span>';
      }
      return '<span class="num">' + s.toFixed(4) + '</span>';
    }

    function critiqueBadge(row) {
      var c = row.critique;
      if (!c || typeof c !== 'object') return '<span class="faint">\u2014</span>';
      var v = c.verdict;
      var cls = v === 'pass' ? ' ok' : (v === 'fix' ? ' fail' : '');
      return '<span class="badge' + cls + '">' + esc(String(v == null ? 'critique' : v)) + '</span>';
    }

    function cardsHtml(row) {
      var used = Array.isArray(row.cards_used) ? row.cards_used : [];
      if (!used.length) return '<span class="faint">\u2014</span>';
      return used.map(function (c) {
        var name = c && c.name_en != null ? esc(c.name_en) : '';
        var ori = c && c.orientation ? esc(c.orientation) : '';
        var badge = ori
          ? ' <span class="badge' + (c.orientation === 'reversed' ? ' amber' : '') + '">' + ori + '</span>'
          : '';
        return '<div>' + name + badge + '</div>';
      }).join('');
    }

    function rowHtml(row) {
      return '<tr data-example="' + esc(row.example_id) + '">' +
        '<td><code>' + esc(row.example_id) + '</code></td>' +
        '<td><span class="badge">' + esc(row.task_type) + '</span></td>' +
        '<td>' + (row.matched_pair_id != null && row.matched_pair_id !== ''
          ? '<code>' + esc(row.matched_pair_id) + '</code>'
          : '<span class="faint">\u2014</span>') + '</td>' +
        '<td>' + (row.spread_name_vi ? esc(row.spread_name_vi) : '<span class="faint">\u2014</span>') + '</td>' +
        '<td>' + cardsHtml(row) + '</td>' +
        '<td>' + (row.querent_context ? esc(row.querent_context) : '<span class="faint">\u2014</span>') + '</td>' +
        '<td>' + (row.register ? esc(row.register) : '<span class="faint">\u2014</span>') + '</td>' +
        '<td>' + (row.length_band ? esc(row.length_band) : '<span class="faint">\u2014</span>') + '</td>' +
        '<td>' + fmtIfd(row) + '</td>' +
        '<td>' + critiqueBadge(row) + '</td>' +
        '</tr>';
    }

    var COLUMNS = [
      'example_id', 'task_type', 'matched_pair_id', 'spread_name_vi',
      'cards_used', 'querent_context', 'register', 'length_band',
      'ifd_score', 'critique'
    ];

    function fillTable() {
      if (!state.rows.length) {
        el.tbody.innerHTML = '<tr><td colspan="' + COLUMNS.length + '" class="muted">' +
          'No rows match the current filters.</td></tr>';
        return;
      }
      el.tbody.innerHTML = state.rows.map(rowHtml).join('');
    }

    /* ------------------------------------------------------------------ */
    /* pager                                                               */
    /* ------------------------------------------------------------------ */
    function fillPager() {
      var total = state.total;
      var totalPages = Math.max(1, Math.ceil(total / state.pageSize));
      var page = Math.min(state.page, totalPages);

      el.pager.innerHTML =
        '<button type="button" class="btn" data-page="prev"' + (page <= 1 ? ' disabled' : '') + '>\u2190 prev</button>' +
        '<span>Page <strong data-role="page-num">' + page + '</strong> / ' + totalPages + '</span>' +
        '<span data-role="pager-total">' + total + ' rows</span>' +
        '<button type="button" class="btn" data-page="next"' + (page >= totalPages ? ' disabled' : '') + '>next \u2192</button>' +
        '<label class="small" style="margin-left:8px">page size ' +
        '<select data-role="page-size" aria-label="page size">' +
        PAGE_SIZES.map(function (s) {
          return '<option value="' + s + '"' + (s === state.pageSize ? ' selected' : '') + '>' + s + '</option>';
        }).join('') +
        '</select></label>';

      el.pager.querySelector('[data-role="page-size"]').addEventListener('change', function (ev) {
        state.pageSize = parseInt(ev.target.value, 10) || 50;
        state.page = 1;
        fetchRows();
      });
    }

    /* ------------------------------------------------------------------ */
    /* row detail                                                          */
    /* ------------------------------------------------------------------ */
    function critiquePanel(row) {
      var c = row.critique;
      if (!c || typeof c !== 'object') return '';
      var v = c.verdict;
      var cls = v === 'pass' ? ' ok' : (v === 'fix' ? ' fail' : '');
      var axes = [
        ['answers_question', c.answers_question],
        ['faithful', c.faithful],
        ['orientation_ok', c.orientation_ok],
        ['vietnamese_natural', c.vietnamese_natural]
      ];
      var issues = Array.isArray(c.issues) ? c.issues : [];
      var html = '<div class="panel" style="margin:12px 0">' +
        '<div class="panel-title">Critique <span class="badge' + cls + '">' + esc(String(v == null ? '\u2014' : v)) + '</span></div>' +
        '<div class="grid cols-2 small">';
      axes.forEach(function (a) {
        if (a[1] == null) return;
        html += '<div><strong>' + esc(a[0]) + ':</strong> ' + esc(String(a[1])) + '</div>';
      });
      html += '</div>';
      if (issues.length) {
        html += '<ul class="small" style="margin:8px 0 0;padding-left:18px">' +
          issues.map(function (i) { return '<li>' + esc(String(i)) + '</li>'; }).join('') +
          '</ul>';
      }
      html += '</div>';
      return html;
    }

    function metaHtml(row) {
      var items = [
        ['matched_pair_id', row.matched_pair_id],
        ['safety_category', row.safety_category],
        ['grounding_defect', row.grounding_defect],
        ['wrong_claim', row.wrong_claim],
        ['critique_applied', row.critique_applied],
        ['rotated_axis', row.rotated_axis],
        ['matched_member', row.matched_member]
      ];
      return items.map(function (m) {
        var v = m[1];
        if (v == null || v === '') v = '\u2014';
        var display = typeof v === 'object' ? esc(JSON.stringify(v)) : esc(String(v));
        return '<div class="field"><label>' + esc(m[0]) + '</label><span>' + display + '</span></div>';
      }).join('');
    }

    function provenanceHtml(row) {
      var p = row.provenance;
      if (!Array.isArray(p) || !p.length) return '<span class="faint">\u2014</span>';
      return p.map(function (x) { return '<span class="badge">' + esc(String(x)) + '</span>'; }).join(' ');
    }

    function detailMeaning(row) {
      var html = '<div class="field" style="margin:10px 0">' +
        '<label>Question (question_vi)</label>' +
        '<div style="white-space:pre-wrap;background:var(--bg-inset);border:1px solid var(--border);border-radius:var(--radius);padding:8px 10px">' +
        esc(row.question_vi) + '</div></div>' +
        '<div class="field" style="margin:10px 0">' +
        '<label>Reading (reading_vi)</label>' +
        '<div style="white-space:pre-wrap;background:var(--bg-inset);border:1px solid var(--border);border-radius:var(--radius);padding:8px 10px">' +
        esc(row.reading_vi) + '</div></div>';

      if (row.reading_vi_original != null && row.reading_vi_original !== '') {
        html += '<div class="field" style="margin:10px 0">' +
          '<label>Reading original (reading_vi_original)</label>' +
          '<div style="white-space:pre-wrap;background:var(--bg-inset);border:1px solid var(--border);border-radius:var(--radius);padding:8px 10px">' +
          esc(row.reading_vi_original) + '</div></div>';
      }

      html += critiquePanel(row);

      html += '<h4>Metadata</h4>' +
        '<div class="grid cols-3 small" style="margin-bottom:10px">' + metaHtml(row) + '</div>';

      html += '<h4>Provenance</h4><div style="display:flex;flex-wrap:wrap;gap:6px">' +
        provenanceHtml(row) + '</div>';

      return html;
    }

    function renderDetail() {
      var row = state.detailRow;
      if (!row) return;
      var head = '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px">' +
        '<h3 style="margin:0">' + esc(row.example_id) +
        ' <span class="badge">' + esc(row.task_type || '') + '</span>' +
        (row.tier ? ' <span class="badge billed">tier: ' + esc(row.tier) + '</span>' : '') +
        '</h3>' +
        '<span style="margin-left:auto;display:flex;gap:6px">' +
        '<button type="button" class="btn-link"' + (state.detailTab === 'meaning' ? ' style="text-decoration:underline"' : '') + ' data-tab="meaning">Meaning</button>' +
        '<button type="button" class="btn-link"' + (state.detailTab === 'raw' ? ' style="text-decoration:underline"' : '') + ' data-tab="raw">Raw JSON</button>' +
        '<button type="button" class="btn-link" data-action="close-detail">close</button>' +
        '</span></div>';
      var body = state.detailTab === 'raw'
        ? '<pre class="json-block">' + esc(JSON.stringify(row, null, 2)) + '</pre>'
        : detailMeaning(row);
      el.detail.innerHTML = head + body;
      el.detail.style.display = '';
    }

    function showDetail(row) {
      state.detailRow = row;
      state.detailTab = 'meaning';
      renderDetail();
      if (el.detail.scrollIntoView) el.detail.scrollIntoView({ block: 'nearest' });
    }

    function hideDetail() {
      state.detailRow = null;
      if (el.detail) el.detail.style.display = 'none';
    }

    /* ------------------------------------------------------------------ */
    /* delegated interactions                                              */
    /* ------------------------------------------------------------------ */
    function onContainerClick(ev) {
      var t = ev.target;
      if (!t || !t.closest) return;

      var tr = t.closest('tr[data-example]');
      if (tr && el.tbody.contains(tr)) {
        var id = tr.getAttribute('data-example');
        for (var i = 0; i < state.rows.length; i++) {
          if (String(state.rows[i].example_id) === id) { showDetail(state.rows[i]); return; }
        }
        return;
      }

      var tabBtn = t.closest('[data-tab]');
      if (tabBtn && state.detailRow) {
        state.detailTab = tabBtn.getAttribute('data-tab');
        renderDetail();
        return;
      }

      if (t.closest('[data-action="close-detail"]')) { hideDetail(); return; }
      if (t.closest('[data-action="reset-filters"]')) { resetFilters(); return; }

      var pageBtn = t.closest('[data-page]');
      if (pageBtn) {
        var dir = pageBtn.getAttribute('data-page');
        var totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
        if (dir === 'prev' && state.page > 1) { state.page -= 1; fetchRows(); }
        if (dir === 'next' && state.page < totalPages) { state.page += 1; fetchRows(); }
      }
    }

    /* ------------------------------------------------------------------ */
    /* init                                                                */
    /* ------------------------------------------------------------------ */
    containerEl.innerHTML = '';

    el.form = buildFilterBar();
    containerEl.appendChild(el.form);

    var wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    var table = document.createElement('table');
    table.className = 'data';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    COLUMNS.forEach(function (c) {
      var th = document.createElement('th');
      th.textContent = c;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    el.tbody = document.createElement('tbody');
    table.appendChild(el.tbody);
    wrap.appendChild(table);
    containerEl.appendChild(wrap);

    el.pager = document.createElement('div');
    el.pager.className = 'pager';
    containerEl.appendChild(el.pager);

    el.detail = document.createElement('div');
    el.detail.className = 'panel';
    el.detail.setAttribute('data-role', 'row-detail');
    el.detail.style.display = 'none';
    containerEl.appendChild(el.detail);

    containerEl.addEventListener('click', onContainerClick);

    fetchRows();
  };
})();
