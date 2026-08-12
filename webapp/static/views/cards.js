/* tfvn-tarot dataset viewer — KB explorer / Cards view (B.3)
 *
 * Data sources:
 *   /api/rows/cards     page, page_size, q, card_id, orientation, ...
 *   /api/rows/spreads   (secondary table, all 21 layouts)
 *   /api/rows/vn_spine  (reference only — the detail panel is driven by the
 *                        cards dataset, which already carries meaning_vi)
 *
 * Filtering note: the A.4 API exposes q / orientation / card_id as query
 * params but NOT arcana / suit / vi_provenance (verified in
 * src/tfvn/webapp/filtering.py::RowsParams). cards.jsonl is 156 rows total,
 * so this view fetches ONE page_size=200 request (the API cap) and applies
 * ALL filters client-side for uniform behaviour — see learnings.md.
 *
 * View contract (app.js): window.Views.cards = function (containerEl, app).
 * Every interpolated value goes through app.esc(). All data rendered is
 * HTML-escaped; row data comes from the read-only /api endpoints.
 */
(function () {
  'use strict';

  var TABLE_PAGE = 25; // rows per page in the main table (client-side pager)

  var DOMAIN_LABELS = {
    title_main: 'Main',
    title_love: 'Love',
    title_work: 'Work',
    title_money: 'Money',
    title_health: 'Health'
  };

  window.Views.cards = function (containerEl, app) {
    var esc = app.esc;

    /* ---------------- state ---------------- */
    var allRows = [];       // every cards.jsonl row (156)
    var filtered = [];      // rows passing the current filters
    var tablePage = 1;
    var state = { arcana: 'all', suit: 'all', orientation: 'all', provenance: 'all', q: '' };

    /* ---------------- small helpers ---------------- */
    // esc() the value, or render an em dash when null / empty.
    function dash(v) {
      return (v == null || String(v) === '') ? '&mdash;' : esc(v);
    }

    function badge(kind, text) {
      return '<span class="badge ' + esc(kind) + '">' + esc(text) + '</span>';
    }

    /* ---------------- filtering (all client-side) ---------------- */
    function matches(row) {
      if (state.arcana !== 'all' && row.arcana !== state.arcana) return false;
      if (state.suit !== 'all') {
        if (state.suit === 'none') { if (row.suit != null) return false; }
        else if (row.suit !== state.suit) return false;
      }
      if (state.orientation !== 'all' && row.orientation !== state.orientation) return false;
      if (state.provenance !== 'all' && row.vi_provenance !== state.provenance) return false;
      if (state.q) {
        var needle = state.q.toLowerCase();
        var hay = [
          row.name_en, row.name_vi_gloss, row.arcana, row.suit,
          row.polarity_axis, row.meaning_en, row.meaning_vi,
          (row.keywords_en || []).join(' '), (row.keywords_vi || []).join(' ')
        ].join(' ').toLowerCase();
        if (hay.indexOf(needle) === -1) return false;
      }
      return true;
    }

    /* ---------------- main table ---------------- */
    function renderTable() {
      filtered = allRows.filter(matches);
      var pages = Math.max(1, Math.ceil(filtered.length / TABLE_PAGE));
      if (tablePage > pages) tablePage = pages;
      var start = (tablePage - 1) * TABLE_PAGE;
      var slice = filtered.slice(start, start + TABLE_PAGE);
      var tbody = root.querySelector('[data-b3="cards-body"]');

      if (!slice.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="muted">No cards match the current filters.</td></tr>';
      } else {
        tbody.innerHTML = slice.map(function (row) {
          var len = (row.meaning_vi || '').length;
          return '<tr data-card="' + esc(row.card_id) + '" tabindex="0" role="button" ' +
            'aria-label="View ' + esc(row.name_en) + ' upright and reversed meanings">' +
            '<td class="mono">' + esc(row.card_id) + '</td>' +
            '<td>' + esc(row.name_en) + '</td>' +
            '<td>' + (row.arcana === 'major' ? badge('amber', 'major') : badge('', 'minor')) + '</td>' +
            '<td>' + dash(row.suit) + '</td>' +
            '<td>' + (row.orientation === 'upright' ? badge('ok', 'upright') : badge('billed', 'reversed')) + '</td>' +
            '<td>' + dash(row.polarity_axis) + '</td>' +
            '<td>' + badge('', row.vi_provenance) + '</td>' +
            '<td class="num">' + esc((row.keywords_en || []).length) + '</td>' +
            '<td class="num">' + (len ? esc(len) : '&mdash;') + '</td>' +
            '</tr>';
        }).join('');
      }

      Array.prototype.forEach.call(tbody.querySelectorAll('tr[data-card]'), function (tr) {
        function open() {
          openDetail(Number(tr.getAttribute('data-card')));
        }
        tr.addEventListener('click', open);
        tr.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
        });
      });

      var pager = root.querySelector('[data-b3="pager"]');
      var shown = slice.length ? (start + 1) + '&ndash;' + (start + slice.length) : '0';
      pager.innerHTML =
        '<span class="mono" data-b3="pager-info">' + shown + ' of ' + esc(filtered.length) +
        ' &middot; page ' + esc(tablePage) + ' of ' + esc(pages) + '</span>' +
        '<button type="button" class="btn" data-b3="prev" ' + (tablePage <= 1 ? 'disabled' : '') + '>&larr; Prev</button>' +
        '<button type="button" class="btn" data-b3="next" ' + (tablePage >= pages ? 'disabled' : '') + '>Next &rarr;</button>';
    }

    /* ---------------- detail panel (upright vs reversed) ---------------- */
    function metaRows(row) {
      function kv(k, v) {
        return '<tr><th>' + esc(k) + '</th><td>' + dash(v) + '</td></tr>';
      }
      return '<table class="data"><tbody>' +
        kv('Element', row.element) +
        kv('Planet', row.planet) +
        kv('Zodiac', row.zodiac) +
        kv('Number', row.number) +
        kv('Source IDs', (row.source_ids || []).join(', ')) +
        kv('Forbidden claims', (row.forbidden_claims || []).join(', ')) +
        kv('Yes / No', row.yes_no) +
        kv('Polarity axis', row.polarity_axis) +
        kv('vi attribution', row.vi_orientation_attribution) +
        '</tbody></table>';
    }

    function meaningPanel(row, label) {
      if (!row) {
        return '<div class="panel"><div class="muted">No ' + esc(label) + ' row for this card.</div></div>';
      }
      var domain = row.domain_vi || {};
      var domainHtml;
      if (!Object.keys(domain).length) {
        domainHtml = '<div class="muted small">No domain_vi on this ' + esc(row.orientation) +
          ' row (synthetic reversed meanings carry no domain_vi).</div>';
      } else {
        domainHtml = '<div class="grid cols-2">' + Object.keys(DOMAIN_LABELS).map(function (k) {
          var v = domain[k];
          if (v == null || v === '') return '';
          return '<div class="panel b3-domain"><h4>' + esc(DOMAIN_LABELS[k]) +
            '</h4><p class="small">' + esc(v) + '</p></div>';
        }).join('') + '</div>';
      }
      var kwEn = (row.keywords_en || []).map(function (k) { return badge('', k); }).join(' ') || '&mdash;';
      var kwVi = (row.keywords_vi || []).map(function (k) { return badge('', k); }).join(' ') || '&mdash;';
      var orient = row.orientation === 'upright' ? badge('ok', 'upright') : badge('billed', 'reversed');
      return '<div class="panel">' +
        '<h3>' + esc(row.name_en) + ' ' + orient + ' ' + badge('', row.vi_provenance || '') + '</h3>' +
        '<h4>Meaning (EN)</h4><p>' + dash(row.meaning_en) + '</p>' +
        '<h4>Meaning (VI)</h4><p>' + dash(row.meaning_vi) + '</p>' +
        '<h4>Keywords (EN)</h4><p>' + kwEn + '</p>' +
        '<h4>Keywords (VI)</h4><p>' + kwVi + '</p>' +
        '<h4>domain_vi</h4>' + domainHtml +
        '<h4>Metadata</h4>' + metaRows(row) +
        '</div>';
    }

    function renderDetail(detailEl, rows) {
      var upright = null;
      var reversed = null;
      rows.forEach(function (r) {
        if (r.orientation === 'upright') upright = r;
        else reversed = r;
      });
      if (!upright && !reversed) {
        detailEl.innerHTML = '<div class="muted">No rows found for this card.</div>';
        return;
      }
      var card = upright || reversed;
      var header = '<h2>' + esc(card.name_en) + ' ' +
        (card.arcana === 'major' ? badge('amber', 'major') : badge('', 'minor')) +
        ' <span class="muted small">card ' + esc(card.card_id) + ' &middot; ' +
        (dash(card.suit) !== '&mdash;' ? 'suit ' + esc(card.suit) + ' &middot; ' : '') +
        'element ' + esc(card.element) + (card.planet ? ' &middot; ' + esc(card.planet) : '') +
        (card.zodiac ? ' &middot; ' + esc(card.zodiac) : '') + '</span></h2>';

      var tabs =
        '<div class="b3-tabs">' +
        '<button type="button" class="btn btn-primary" data-b3="tab-compare">Compare</button>' +
        '<button type="button" class="btn" data-b3="tab-raw">Raw JSON</button>' +
        '</div>';

      var compare =
        '<div data-b3="panel-compare" class="grid cols-2 b3-compare">' +
        meaningPanel(upright, 'upright') +
        meaningPanel(reversed, 'reversed') +
        '</div>';

      var raw =
        '<div data-b3="panel-raw" hidden>' +
        '<div class="json-block">' + esc(JSON.stringify({ upright: upright, reversed: reversed }, null, 2)) + '</div>' +
        '</div>';

      detailEl.innerHTML = header + tabs + compare + raw;

      var btnCompare = detailEl.querySelector('[data-b3="tab-compare"]');
      var btnRaw = detailEl.querySelector('[data-b3="tab-raw"]');
      var panelCompare = detailEl.querySelector('[data-b3="panel-compare"]');
      var panelRaw = detailEl.querySelector('[data-b3="panel-raw"]');
      btnCompare.addEventListener('click', function () {
        panelRaw.hidden = true;
        panelCompare.hidden = false;
        btnRaw.classList.remove('btn-primary');
        btnCompare.classList.add('btn-primary');
      });
      btnRaw.addEventListener('click', function () {
        panelCompare.hidden = true;
        panelRaw.hidden = false;
        btnCompare.classList.remove('btn-primary');
        btnRaw.classList.add('btn-primary');
      });
    }

    function openDetail(cardId) {
      var detailEl = root.querySelector('[data-b3="detail"]');
      detailEl.hidden = false;
      detailEl.innerHTML = '';
      app.spinner.show(detailEl);
      app.fetchJSON('/api/rows/cards?card_id=' + cardId + '&page_size=200')
        .then(function (res) {
          app.spinner.hide(detailEl);
          renderDetail(detailEl, res.rows || []);
        })
        .catch(function () {
          app.spinner.hide(detailEl);
          detailEl.innerHTML = '<div class="view-error">Failed to load card ' + esc(cardId) + '.</div>';
        });
      detailEl.scrollIntoView({ block: 'nearest' });
    }

    /* ---------------- spreads table ---------------- */
    function renderSpreads(rows) {
      var tbody = root.querySelector('[data-b3="spreads-body"]');
      if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="muted">No spreads.</td></tr>';
        return;
      }
      tbody.innerHTML = rows.map(function (s) {
        return '<tr>' +
          '<td class="mono">' + esc(s.spread_id) + '</td>' +
          '<td>' + esc(s.name_en) + '</td>' +
          '<td>' + dash(s.name_vi) + '</td>' +
          '<td class="num">' + esc(s.cards_drawn) + '</td>' +
          '<td>' + badge('', s.difficulty) + '</td>' +
          '</tr>';
      }).join('');
    }

    /* ---------------- chrome ---------------- */
    containerEl.innerHTML = '';
    var root = document.createElement('div');
    root.innerHTML =
      '<h1>Knowledge base <span class="muted small">&mdash; kb/cards.jsonl &middot; 156 rows (78 cards &times; upright/reversed)</span></h1>' +

      '<div class="filter-bar">' +
      '<div class="field"><label for="b3-arcana">Arcana</label>' +
      '<select id="b3-arcana" data-b3="f-arcana">' +
      '<option value="all">all</option><option value="major">major</option><option value="minor">minor</option>' +
      '</select></div>' +
      '<div class="field"><label for="b3-suit">Suit</label>' +
      '<select id="b3-suit" data-b3="f-suit">' +
      '<option value="all">all</option><option value="none">none (major)</option>' +
      '<option value="cups">cups</option><option value="pentacles">pentacles</option>' +
      '<option value="swords">swords</option><option value="wands">wands</option>' +
      '</select></div>' +
      '<div class="field"><label for="b3-orientation">Orientation</label>' +
      '<select id="b3-orientation" data-b3="f-orientation">' +
      '<option value="all">all</option><option value="upright">upright</option><option value="reversed">reversed</option>' +
      '</select></div>' +
      '<div class="field"><label for="b3-provenance">vi_provenance</label>' +
      '<select id="b3-provenance" data-b3="f-provenance">' +
      '<option value="all">all</option><option value="source">source</option>' +
      '<option value="synthetic">synthetic</option><option value="synthetic_no_anchor">synthetic_no_anchor</option>' +
      '</select></div>' +
      '<div class="field"><label for="b3-q">Search</label>' +
      '<input type="search" id="b3-q" data-b3="f-q" placeholder="name, meaning, keywords&hellip;">' +
      '</div>' +
      '</div>' +

      '<p class="small muted" data-b3="note">Filters are applied client-side over one page_size=200 fetch ' +
      '(the A.4 API exposes q / orientation / card_id but not arcana / suit / vi_provenance).</p>' +

      '<div class="table-wrap"><table class="data">' +
      '<thead><tr><th>id</th><th>name_en</th><th>arcana</th><th>suit</th><th>orientation</th>' +
      '<th>polarity_axis</th><th>vi_provenance</th><th class="num">keywords_en</th><th class="num">meaning_vi len</th></tr></thead>' +
      '<tbody data-b3="cards-body"></tbody>' +
      '</table></div>' +

      '<div class="pager" data-b3="pager"></div>' +

      '<div data-b3="detail" hidden></div>' +

      '<div class="panel">' +
      '<div class="panel-title">Spreads &mdash; kb/spreads.jsonl &middot; layouts</div>' +
      '<div class="table-wrap"><table class="data">' +
      '<thead><tr><th>spread_id</th><th>name_en</th><th>name_vi</th><th class="num">cards_drawn</th><th>difficulty</th></tr></thead>' +
      '<tbody data-b3="spreads-body"></tbody>' +
      '</table></div>' +
      '</div>';

    containerEl.appendChild(root);

    /* ---------------- wire filters + pager ---------------- */
    root.querySelector('[data-b3="f-arcana"]').addEventListener('change', function (e) {
      state.arcana = e.target.value; tablePage = 1; renderTable();
    });
    root.querySelector('[data-b3="f-suit"]').addEventListener('change', function (e) {
      state.suit = e.target.value; tablePage = 1; renderTable();
    });
    root.querySelector('[data-b3="f-orientation"]').addEventListener('change', function (e) {
      state.orientation = e.target.value; tablePage = 1; renderTable();
    });
    root.querySelector('[data-b3="f-provenance"]').addEventListener('change', function (e) {
      state.provenance = e.target.value; tablePage = 1; renderTable();
    });
    root.querySelector('[data-b3="f-q"]').addEventListener('input', function (e) {
      state.q = e.target.value; tablePage = 1; renderTable();
    });
    root.querySelector('[data-b3="pager"]').addEventListener('click', function (e) {
      if (e.target.getAttribute('data-b3') === 'prev') { tablePage -= 1; renderTable(); }
      else if (e.target.getAttribute('data-b3') === 'next') { tablePage += 1; renderTable(); }
    });

    /* ---------------- load data ---------------- */
    app.spinner.show(containerEl);
    Promise.all([
      app.fetchJSON('/api/rows/cards?page=1&page_size=200'),
      app.fetchJSON('/api/rows/spreads?page=1&page_size=200')
    ]).then(function (results) {
      app.spinner.hide(containerEl);
      allRows = results[0].rows || [];
      renderSpreads(results[1].rows || []);
      renderTable();
      if (!allRows.length) {
        root.querySelector('[data-b3="note"]').textContent =
          'No rows returned from /api/rows/cards.';
      }
    }).catch(function () {
      app.spinner.hide(containerEl);
      root.querySelector('[data-b3="note"]').textContent =
        'Failed to load cards — see the error toast above.';
    });
  };
})();
