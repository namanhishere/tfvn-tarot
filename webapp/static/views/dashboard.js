/* tfvn-tarot dataset viewer — Dashboard view (B.2)
 *
 * Fetches /api/catalog, /api/hashcheck, /api/stats in parallel and renders:
 *   1. headline stat cards (total SFT examples, core/bulk split, top
 *      task_type / register / querent_context, cards_drawn mode,
 *      % reversed, safety pair count)
 *   2. hash integrity panel (green/red per check + method shown; red
 *      .banner-fail banner when any check mismatches — the stale-data signal)
 *   3. artifact catalog table (path, kind, tag, rows, size KB, sha256 short,
 *      badge per tag)
 *
 * Partial-failure policy: every section renders what succeeded; a failed
 * fetch has already toasted via app.fetchJSON and the section shows a compact
 * "unavailable" note instead — never a blank page, never NaN/undefined.
 * All interpolated data goes through textContent (app.esc-equivalent); the
 * only innerHTML-free template bits are static strings.
 */
(function () {
  'use strict';

  window.Views = window.Views || {};
  window.Views.dashboard = function (containerEl, app) {
    containerEl.setAttribute('data-view', 'dashboard');
    containerEl.innerHTML = '';
    app.spinner.show(containerEl);

    // View-specific styles: .banner-fail mirrors the theme's red banner
    // (styles.css only ships .banner-ok / .banner-warn / .banner-error).
    var style = document.createElement('style');
    style.textContent = [
      '.banner-fail { border-color: rgba(224,101,92,.55); background: rgba(224,101,92,.12); color: var(--fail); }',
      '.dash-card .stat-value { overflow-wrap: anywhere; }',
      '.dash-mono { font-family: var(--mono); font-size: 12px; }',
      '.dash-unavail { border: 1px dashed var(--border-strong); border-radius: var(--radius); padding: 12px 14px; color: var(--text-dim); background: var(--bg-panel); margin-bottom: 16px; }'
    ].join('\n');
    containerEl.appendChild(style);

    // null = pending, object = ok, false = failed (toast already shown)
    var state = { catalog: null, hashcheck: null, stats: null };

    function settled() {
      return state.catalog !== null && state.hashcheck !== null && state.stats !== null;
    }

    function maybeRender() {
      if (!settled()) return;
      app.spinner.hide(containerEl);
      if (state.hashcheck) renderHash(state.hashcheck);
      else if (state.hashcheck === false) unavail('Hash integrity');
      if (state.stats) renderStats(state.stats);
      else if (state.stats === false) unavail('Dataset headline stats');
      if (state.catalog) renderCatalog(state.catalog);
      else if (state.catalog === false) unavail('Artifact catalog');
    }

    function grab(key, path) {
      app.fetchJSON(path).then(
        function (data) { state[key] = data; },
        function () { state[key] = false; }
      ).then(maybeRender);
    }

    grab('catalog', '/api/catalog');
    grab('hashcheck', '/api/hashcheck');
    grab('stats', '/api/stats');

    /* ---------------------------------------------------------------- */
    /* DOM + formatting helpers                                          */
    /* ---------------------------------------------------------------- */

    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text != null) node.textContent = String(text);
      return node;
    }

    function fmtInt(n) {
      if (n == null || isNaN(n)) return '—';
      return Number(n).toLocaleString('en-US');
    }

    function fmtKB(bytes) {
      if (bytes == null || isNaN(bytes) || bytes <= 0) return '—';
      return (bytes / 1024).toFixed(1);
    }

    function shaShort(sha) {
      return sha ? String(sha).slice(0, 8) : '—';
    }

    function argmax(dist) {
      var best = null;
      var bestCount = -1;
      for (var key in dist) {
        if (Object.prototype.hasOwnProperty.call(dist, key) && dist[key] > bestCount) {
          best = key;
          bestCount = dist[key];
        }
      }
      return best == null ? null : { key: best, count: bestCount };
    }

    function sumDist(dist) {
      var total = 0;
      for (var key in dist) {
        if (Object.prototype.hasOwnProperty.call(dist, key)) total += dist[key];
      }
      return total;
    }

    function section(title) {
      var panel = el('section', 'panel');
      panel.appendChild(el('h2', 'panel-title', title));
      containerEl.appendChild(panel);
      return panel;
    }

    function unavail(title) {
      var panel = section(title);
      panel.appendChild(el(
        'p', 'dash-unavail',
        'Section unavailable — the endpoint failed (see the error toast above).'
      ));
    }

    /* ---------------------------------------------------------------- */
    /* Hash integrity panel                                               */
    /* ---------------------------------------------------------------- */

    var CHECK_NAMES = {
      cards: 'cards.jsonl',
      datasets: 'filtered_core + filtered_bulk'
    };

    function renderHash(hc) {
      var checks = hc.checks || [];
      var anyFail = !hc.cards_match || !hc.dataset_match;
      var panel = section('Hash integrity');

      var failed = checks.filter(function (c) { return !c.matches; })
        .map(function (c) { return CHECK_NAMES[c.id] || c.id; });

      var banner = el('div', 'banner ' + (anyFail ? 'banner-fail' : 'banner-ok'));
      if (anyFail) {
        banner.appendChild(el('strong', null, 'STALE DATA'));
        banner.appendChild(el(
          'span', null,
          ' — recorded digests do not match the live files for: ' + failed.join(', ') +
          '. The dataset may have been rebuilt without updating the hash files.'
        ));
      } else {
        banner.appendChild(el('strong', null, 'Hashes match'));
        banner.appendChild(el(
          'span', null,
          ' — every recorded digest verified against the live files.'
        ));
      }
      panel.appendChild(banner);

      var table = el('table', 'data');
      var thead = document.createElement('thead');
      var headRow = document.createElement('tr');
      ['Target', 'Method', 'Status', 'Computed', 'Recorded'].forEach(function (h) {
        headRow.appendChild(el('th', null, h));
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      var tbody = document.createElement('tbody');
      checks.forEach(function (check) {
        var tr = document.createElement('tr');
        tr.appendChild(el('td', 'dash-mono', CHECK_NAMES[check.id] || check.id));
        tr.appendChild(el('td', 'dash-mono', check.method || 'canonical'));
        var statusCell = el('td');
        statusCell.appendChild(el(
          'span', 'badge ' + (check.matches ? 'ok' : 'fail'),
          check.matches ? 'matches' : 'MISMATCH'
        ));
        tr.appendChild(statusCell);
        tr.appendChild(el('td', 'dash-mono', shaShort(check.computed_digest)));
        tr.appendChild(el('td', 'dash-mono', shaShort(check.recorded_digest)));
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);

      var wrap = el('div', 'table-wrap');
      wrap.appendChild(table);
      panel.appendChild(wrap);
    }

    /* ---------------------------------------------------------------- */
    /* Headline stat cards                                                */
    /* ---------------------------------------------------------------- */

    function statCard(value, label, dataStat) {
      var card = el('div', 'stat-card dash-card');
      var v = el('div', 'stat-value', value);
      if (dataStat) v.setAttribute('data-stat', dataStat);
      card.appendChild(v);
      card.appendChild(el('div', 'stat-label', label));
      return card;
    }

    function renderStats(st) {
      var panel = section('Dataset headline stats');
      var grid = el('div', 'grid cols-4');
      var dist = st.distributions || {};

      var total = st.source && st.source.total != null ? st.source.total
        : (st.tier_counts ? (st.tier_counts.core || 0) + (st.tier_counts.bulk || 0) : null);
      grid.appendChild(statCard(fmtInt(total), 'Total SFT examples (core + bulk)', 'total'));

      var core = st.tier_counts ? (st.tier_counts.core || 0) : 0;
      var bulk = st.tier_counts ? (st.tier_counts.bulk || 0) : 0;
      grid.appendChild(statCard(fmtInt(core) + ' / ' + fmtInt(bulk), 'Core / bulk split', 'split'));

      var topTask = argmax(dist.task_type);
      grid.appendChild(statCard(
        topTask ? topTask.key : '—',
        topTask ? 'Top task_type · ' + fmtInt(topTask.count) + ' rows' : 'Top task_type',
        'top-task'
      ));

      var topReg = argmax(dist.register);
      grid.appendChild(statCard(
        topReg ? topReg.key : '—',
        topReg ? 'Top register · ' + fmtInt(topReg.count) + ' rows' : 'Top register',
        'top-register'
      ));

      var topCtx = argmax(dist.querent_context);
      grid.appendChild(statCard(
        topCtx ? topCtx.key : '—',
        topCtx ? 'Top querent_context · ' + fmtInt(topCtx.count) + ' rows' : 'Top querent_context',
        'top-context'
      ));

      var mode = argmax(dist.cards_drawn);
      grid.appendChild(statCard(
        mode ? mode.key : '—',
        mode ? 'Cards drawn (mode) · ' + fmtInt(mode.count) + ' rows' : 'Cards drawn (mode)',
        'cards-drawn'
      ));

      var revPct = st.total_reversed_percent;
      grid.appendChild(statCard(
        revPct == null ? '—' : Number(revPct).toFixed(2) + '%',
        '% reversed card mentions',
        'reversed'
      ));

      var safetyTotal = sumDist(dist.safety_category);
      grid.appendChild(statCard(
        safetyTotal ? fmtInt(Math.floor(safetyTotal / 2)) : '—',
        safetyTotal ? 'Safety pairs (' + fmtInt(safetyTotal) + ' safety rows ÷ 2)' : 'Safety pairs',
        'safety-pairs'
      ));

      panel.appendChild(grid);
    }

    /* ---------------------------------------------------------------- */
    /* Artifact catalog table                                             */
    /* ---------------------------------------------------------------- */

    var TAG_BADGE = {
      kb: 'ok',
      dataset: 'badge-slow',
      anchor: 'billed',
      raw: 'badge',
      report: 'amber',
      hash: 'badge'
    };

    function renderCatalog(cat) {
      var panel = section('Artifact catalog');
      var table = el('table', 'data');
      var thead = document.createElement('thead');
      var headRow = document.createElement('tr');
      ['Path', 'Kind', 'Tag', 'Rows', 'Size KB', 'sha256'].forEach(function (h) {
        headRow.appendChild(el('th', null, h));
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      var tbody = document.createElement('tbody');
      (cat.artifacts || []).forEach(function (a) {
        var tr = document.createElement('tr');
        tr.appendChild(el('td', 'dash-mono', a.path));
        tr.appendChild(el('td', null, a.kind || '—'));
        var tagCell = el('td');
        tagCell.appendChild(el('span', 'badge ' + (TAG_BADGE[a.tag] || 'badge'), a.tag || '—'));
        tr.appendChild(tagCell);
        tr.appendChild(el('td', 'num', a.rows == null ? '—' : fmtInt(a.rows)));
        tr.appendChild(el('td', 'num', fmtKB(a.size_bytes)));
        tr.appendChild(el('td', 'dash-mono', shaShort(a.sha256)));
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);

      var wrap = el('div', 'table-wrap');
      wrap.appendChild(table);
      panel.appendChild(wrap);
    }
  };
})();
