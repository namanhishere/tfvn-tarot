/* tfvn-tarot dataset viewer — Statistics view (B.6)
 *
 * Renders the single /api/stats payload as CSS-bar-chart sections (per plan
 * B.6: no chart libraries, pure CSS bars). Data source is the one GET
 * /api/stats JSON; every interpolated value is escaped via app.esc().
 *
 * Sections:
 *   1. Overview stat cards (source/tier/card mentions/reversed %/IFD mean)
 *   2. SFT distributions: task_type, register, length_band, querent_context,
 *      spread_id, cards_drawn
 *   3. Per-card orientation matrix (78 rows, upright vs reversed bars,
 *      sorted by total frequency descending)
 *   4. IFD histogram (10 bins, bin edges in a title tooltip)
 *   5. Critique / safety / grounding distributions: critique_applied,
 *      critique_verdict, safety_category, grounding_defect, wrong_claim,
 *      provenance
 *   6. KB stats: arcana/suit/orientation/vi_provenance/polarity_axis/
 *      vi_orientation_attribution counts, meaning_en & meaning_vi char-length
 *      min/mean/max, domain_vi coverage per key
 *   7. Anchor stats (per-card_id + per-orientation)
 *   8. Spreads stats (by_spread_id table + cards_drawn/difficulty)
 *   9. Splits summary (train/val/test sizes + split x task_type cross-tab)
 *
 * Defensive: any distribution key absent from the JSON renders an "no data"
 * note in its panel instead of crashing; a 0 count renders an empty bar
 * (width 0) with the number shown — never NaN.
 */
window.Views.stats = function (containerEl, app) {
  'use strict';

  var esc = app.esc;

  /* Canonical card names in card_id order 0-77 (src/tfvn/aliases.py).
   * /api/stats carries card_ids only; this static map supplies labels. */
  var CARD_NAMES = ["The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor", "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit", "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance", "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World", "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands", "Six of Wands", "Seven of Wands", "Eight of Wands", "Nine of Wands", "Ten of Wands", "Page of Wands", "Knight of Wands", "Queen of Wands", "King of Wands", "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups", "Six of Cups", "Seven of Cups", "Eight of Cups", "Nine of Cups", "Ten of Cups", "Page of Cups", "Knight of Cups", "Queen of Cups", "King of Cups", "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords", "Six of Swords", "Seven of Swords", "Eight of Swords", "Nine of Swords", "Ten of Swords", "Page of Swords", "Knight of Swords", "Queen of Swords", "King of Swords", "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles", "Five of Pentacles", "Six of Pentacles", "Seven of Pentacles", "Eight of Pentacles", "Nine of Pentacles", "Ten of Pentacles", "Page of Pentacles", "Knight of Pentacles", "Queen of Pentacles", "King of Pentacles"];

  /* ---------------------------------------------------------------- */
  /* helpers                                                           */
  /* ---------------------------------------------------------------- */

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : 0; }

  function fmt(v) { return num(v).toLocaleString('en-US'); }

  function pct(count, max) { return max > 0 ? (num(count) / max) * 100 : 0; }

  /* A .bar-fill span; width set inline as % of max. 0 count -> width 0. */
  function fill(count, max, kind) {
    var c = num(count);
    var w = c === 0 ? 'width:0;min-width:0' : 'width:' + pct(c, max).toFixed(2) + '%';
    var kindAttr = kind ? ' data-kind="' + esc(kind) + '"' : '';
    return '<span class="bar-fill"' + kindAttr + ' style="' + w + '"></span>';
  }

  /* One .bar row: label | track>fill | value. */
  function barRow(label, count, max, kind) {
    return '<div class="bar">' +
      '<span class="bar-label" title="' + esc(label) + '">' + esc(label) + '</span>' +
      '<span class="bar-track">' + fill(count, max, kind) + '</span>' +
      '<span class="bar-value">' + fmt(count) + '</span>' +
      '</div>';
  }

  /* A panel titled `title` whose body is `body` (raw HTML). */
  function panel(title, body) {
    return '<div class="panel">' +
      '<div class="panel-title">' + esc(title) + '</div>' + body +
      '</div>';
  }

  /* A distribution panel: bars sorted by count descending. `extra` rows
   * (e.g. a total line) append below the bars. Missing dist -> note. */
  function distPanel(title, dist, extra) {
    if (!dist || typeof dist !== 'object') {
      return panel(title, '<div class="muted small">No data — key absent from /api/stats.</div>');
    }
    var entries = [];
    for (var key in dist) {
      if (Object.prototype.hasOwnProperty.call(dist, key)) entries.push([key, num(dist[key])]);
    }
    entries.sort(function (a, b) { return b[1] - a[1]; });
    var max = entries.length ? entries[0][1] : 0;
    var html = entries.map(function (e) { return barRow(e[0], e[1], max); }).join('');
    if (!html) html = '<div class="muted small">Empty distribution.</div>';
    return panel(title, html + (extra || ''));
  }

  /* Table row with an inline bar (for .data tables / matrices). */
  function barCell(count, max, kind) {
    return '<span class="bar-track" style="display:inline-block;vertical-align:middle;width:64px">' +
      fill(count, max, kind) + '</span>' +
      '<span class="bar-value" style="display:inline-block;min-width:38px;text-align:right">' +
      fmt(count) + '</span>';
  }

  /* ---------------------------------------------------------------- */
  /* section builders                                                  */
  /* ---------------------------------------------------------------- */

  function overview(d) {
    var cells = [
      ['Rows', fmt(d.source ? d.source.total : 0)],
      ['Core', fmt(d.tier_counts ? d.tier_counts.core : 0)],
      ['Bulk', fmt(d.tier_counts ? d.tier_counts.bulk : 0)],
      ['Card mentions', fmt(d.per_card ? d.per_card.total_card_mentions : 0)],
      ['Reversed', (typeof d.total_reversed_percent === 'number' ? d.total_reversed_percent : 0) + '%'],
      ['IFD mean', d.ifd && typeof d.ifd.mean === 'number' ? d.ifd.mean : '—'],
    ];
    return '<div class="grid cols-3" style="margin-bottom:16px">' +
      cells.map(function (c) {
        return '<div class="stat-card"><div class="stat-value">' + esc(c[1]) +
          '</div><div class="stat-label">' + esc(c[0]) + '</div></div>';
      }).join('') +
      '</div>';
  }

  function distributions(d) {
    var dist = d.distributions || {};
    var order = ['task_type', 'register', 'length_band', 'querent_context', 'spread_id', 'cards_drawn'];
    return '<div class="grid cols-2">' +
      order.map(function (k) { return distPanel(k, dist[k]); }).join('') +
      '</div>';
  }

  function perCardMatrix(d) {
    var per = d.per_card || {};
    var freq = per.frequency || {};
    var mix = per.orientation_mix || {};
    var rows = [];
    for (var id = 0; id < CARD_NAMES.length; id++) {
      var idStr = String(id);
      var cell = mix[idStr] || {};
      var u = num(cell.upright);
      var r = num(cell.reversed);
      var total = num(freq[idStr]);
      if (total === 0 && (u > 0 || r > 0)) total = u + r;
      rows.push({ id: id, name: CARD_NAMES[id], upright: u, reversed: r, total: total });
    }
    rows.sort(function (a, b) {
      return b.total - a.total || a.id - b.id;
    });
    var max = rows.length ? rows[0].total : 0;
    var body = rows.map(function (row) {
      return '<tr>' +
        '<td>' + esc(row.id) + ' · ' + esc(row.name) + '</td>' +
        '<td>' + barCell(row.upright, max, 'ok') + '</td>' +
        '<td>' + barCell(row.reversed, max, 'billed') + '</td>' +
        '<td class="num">' + fmt(row.total) + '</td>' +
        '</tr>';
    }).join('');
    return '<div class="panel">' +
      '<div class="panel-title">Per-card orientation (sorted by total frequency)</div>' +
      '<div class="table-wrap" style="max-height:520px;overflow:auto">' +
      '<table class="data"><thead><tr>' +
      '<th>Card</th><th>Upright</th><th>Reversed</th><th>Total</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>' +
      '</div></div>';
  }

  function ifdSection(d) {
    var ifd = d.ifd || {};
    var hist = Array.isArray(ifd.histogram) ? ifd.histogram : [];
    var edges = Array.isArray(ifd.bin_edges) ? ifd.bin_edges : [];
    var max = hist.length ? Math.max.apply(null, hist) : 0;
    var bars = hist.map(function (count, i) {
      var lo = typeof edges[i] === 'number' ? edges[i].toFixed(4) : '?';
      var hi = typeof edges[i + 1] === 'number' ? edges[i + 1].toFixed(4) : '?';
      var label = 'bin ' + (i + 1);
      return '<div class="bar">' +
        '<span class="bar-label" title="' + esc(lo + ' … ' + hi) + '">' + esc(label) +
        ' <span class="faint small">[' + esc(lo) + ', ' + esc(hi) + ')</span></span>' +
        '<span class="bar-track">' + fill(count, max) + '</span>' +
        '<span class="bar-value">' + fmt(count) + '</span>' +
        '</div>';
    }).join('');
    if (!hist.length) {
      bars = '<div class="muted small">No data — key absent from /api/stats.</div>';
    }
    var stats = '<div class="grid cols-4" style="margin-bottom:10px">' +
      [['count', ifd.count], ['min', ifd.min], ['max', ifd.max], ['mean', ifd.mean]]
        .map(function (c) {
          return '<div class="stat-card"><div class="stat-value">' +
            (c[1] === null || typeof c[1] === 'undefined' ? '—' : esc(c[1])) +
            '</div><div class="stat-label">' + esc(c[0]) + '</div></div>';
        }).join('') +
      '</div>';
    return '<div class="panel"><div class="panel-title">IFD score histogram (10 bins)</div>' +
      stats + bars + '</div>';
  }

  function critiqueSafety(d) {
    var dist = d.distributions || {};
    var order = ['critique_applied', 'critique_verdict', 'safety_category', 'grounding_defect', 'wrong_claim', 'provenance'];
    return '<div class="grid cols-2">' +
      order.map(function (k) { return distPanel(k, dist[k]); }).join('') +
      '</div>';
  }

  function kbStats(d) {
    var kb = d.kb || {};
    var countPanels = ['arcana', 'suit', 'orientation', 'vi_provenance', 'polarity_axis', 'vi_orientation_attribution']
      .map(function (k) { return distPanel(k, kb[k]); }).join('');

    function lengthPanel(label, s) {
      s = s || {};
      var body = '<div class="grid cols-3">' +
        [['count', s.count], ['min', s.min], ['mean', s.mean], ['max', s.max]]
          .map(function (c) {
            return '<div class="stat-card"><div class="stat-value">' +
              (c[1] === null || typeof c[1] === 'undefined' ? '—' : esc(c[1])) +
              '</div><div class="stat-label">' + esc(c[0]) + '</div></div>';
          }).join('') +
        '</div>';
      return panel(label + ' char-length (min/mean/max)', body);
    }

    function coveragePanel() {
      var cov = kb.domain_vi_coverage;
      if (!cov || typeof cov !== 'object') {
        return panel('domain_vi coverage', '<div class="muted small">No data — key absent from /api/stats.</div>');
      }
      var keys = Object.keys(cov);
      var orientations = [];
      keys.forEach(function (key) {
        var per = cov[key] || {};
        Object.keys(per).forEach(function (o) {
          if (orientations.indexOf(o) === -1) orientations.push(o);
        });
      });
      var rows = keys.map(function (key) {
        var per = cov[key] || {};
        return '<tr><td>' + esc(key) + '</td>' +
          orientations.map(function (o) {
            return '<td class="num">' + fmt(per[o]) + '</td>';
          }).join('') + '</tr>';
      }).join('');
      var head = orientations.map(function (o) { return '<th>' + esc(o) + '</th>'; }).join('');
      return '<div class="panel"><div class="panel-title">domain_vi coverage (non-empty values)</div>' +
        '<div class="table-wrap"><table class="data"><thead><tr><th>key</th>' + head +
        '</tr></thead><tbody>' + rows + '</tbody></table></div></div>';
    }

    return '<div class="grid cols-2">' +
      countPanels +
      lengthPanel('meaning_en', kb.meaning_en) +
      lengthPanel('meaning_vi', kb.meaning_vi) +
      coveragePanel() +
      '</div>';
  }

  function anchorSection(d) {
    var anchor = d.anchor || {};
    var body;
    if (!anchor || typeof anchor !== 'object') {
      body = '<div class="muted small">No data — key absent from /api/stats.</div>';
    } else {
      var maxCard = 0;
      var byCard = anchor.by_card_id || {};
      Object.keys(byCard).forEach(function (k) {
        if (num(byCard[k]) > maxCard) maxCard = num(byCard[k]);
      });
      var rows = Object.keys(byCard).sort(function (a, b) { return num(byCard[b]) - num(byCard[a]) || num(a) - num(b); })
        .map(function (id) {
          return '<tr><td>' + esc(id) + ' · ' + esc(CARD_NAMES[num(id)] || id) + '</td>' +
            '<td>' + barCell(byCard[id], maxCard) + '</td></tr>';
        }).join('');
      var orient = anchor.by_orientation || {};
      var maxO = Math.max(num(orient.upright), num(orient.reversed), 1);
      var orientBars = ['upright', 'reversed'].map(function (o) {
        return barRow(o, orient[o], maxO, o === 'reversed' ? 'billed' : 'ok');
      }).join('');
      body = '<div class="grid cols-2">' +
        '<div><div class="small muted" style="margin-bottom:6px">Rows: <b>' + fmt(anchor.count) +
        '</b></div>' + orientBars + '</div>' +
        '<div><div class="small muted" style="margin-bottom:6px">Per card</div>' +
        '<div class="table-wrap"><table class="data"><thead><tr><th>Card</th><th>Rows</th></tr></thead>' +
        '<tbody>' + rows + '</tbody></table></div></div>' +
        '</div>';
    }
    return panel('Anchor readings (30-row gold set)', body);
  }

  function spreadsSection(d) {
    var spreads = d.spreads || {};
    if (!spreads || typeof spreads !== 'object') {
      return panel('Spreads', '<div class="muted small">No data — key absent from /api/stats.</div>');
    }
    var byId = spreads.by_spread_id || {};
    var ids = Object.keys(byId).sort(function (a, b) { return num(a.replace(/\D/g, '')) - num(b.replace(/\D/g, '')); });
    var tableRows = ids.map(function (id) {
      var s = byId[id] || {};
      return '<tr><td>' + esc(id) + '</td>' +
        '<td>' + esc(s.name_en === null || typeof s.name_en === 'undefined' ? '—' : s.name_en) + '</td>' +
        '<td>' + esc(s.name_vi === null || typeof s.name_vi === 'undefined' ? '—' : s.name_vi) + '</td>' +
        '<td class="num">' + (typeof s.cards_drawn === 'number' ? s.cards_drawn : '—') + '</td>' +
        '<td>' + esc(s.difficulty === null || typeof s.difficulty === 'undefined' ? '—' : s.difficulty) + '</td>' +
        '<td class="num">' + (typeof s.num_positions === 'number' ? s.num_positions : '—') + '</td>' +
        '</tr>';
    }).join('');
    var body = '<div class="grid cols-2">' +
      distPanel('cards_drawn (spreads)', spreads.cards_drawn) +
      distPanel('difficulty', spreads.difficulty) +
      '</div>' +
      '<div class="table-wrap" style="margin-top:12px"><table class="data"><thead><tr>' +
      '<th>spread_id</th><th>name_en</th><th>name_vi</th><th class="num">cards_drawn</th>' +
      '<th>difficulty</th><th class="num">positions</th></tr></thead>' +
      '<tbody>' + (tableRows || '<tr><td colspan="6" class="muted">No spreads.</td></tr>') +
      '</tbody></table></div>';
    return panel('Spreads (' + fmt(spreads.count) + ')', body);
  }

  function splitsSection(d) {
    var splits = d.splits || {};
    if (!splits || typeof splits !== 'object') {
      return panel('Splits', '<div class="muted small">No data — key absent from /api/stats.</div>');
    }
    var counts = splits.counts || {};
    var keys = Object.keys(counts);
    var max = keys.length ? Math.max.apply(null, keys.map(function (k) { return num(counts[k]); })) : 0;
    var sizeBars = keys.map(function (k) { return barRow(k, counts[k], max); }).join('');
    var summary = '<div class="grid cols-3">' +
      [['train', counts.train], ['val', counts.val], ['test', counts.test]]
        .map(function (c) {
          return '<div class="stat-card"><div class="stat-value">' + fmt(c[1]) +
            '</div><div class="stat-label">' + esc(c[0]) + '</div></div>';
        }).join('') +
      '</div>';
    var cross = splits.by_task_type || {};
    var splitsOrder = ['train', 'val', 'test'].filter(function (s) { return cross[s]; });
    var taskTypes = [];
    splitsOrder.forEach(function (s) {
      Object.keys(cross[s]).forEach(function (t) {
        if (taskTypes.indexOf(t) === -1) taskTypes.push(t);
      });
    });
    var crossRows = splitsOrder.map(function (s) {
      return '<tr><td>' + esc(s) + '</td>' +
        taskTypes.map(function (t) {
          return '<td class="num">' + fmt(cross[s][t]) + '</td>';
        }).join('') +
        '<td class="num">' + fmt(counts[s]) + '</td></tr>';
    }).join('');
    var crossTable = '<div class="table-wrap"><table class="data"><thead><tr><th>split</th>' +
      taskTypes.map(function (t) { return '<th class="num">' + esc(t) + '</th>'; }).join('') +
      '<th class="num">total</th></tr></thead><tbody>' +
      (crossRows || '<tr><td class="muted">No cross-tab.</td></tr>') + '</tbody></table></div>';
    return '<div class="panel"><div class="panel-title">Splits summary</div>' +
      '<div class="grid cols-2" style="margin-bottom:12px">' +
      '<div>' + sizeBars + '</div><div>' + summary +
      '<div class="small muted" style="margin-top:8px">unmatched: ' + fmt(splits.unmatched_rows) +
      ' · total rows: ' + fmt(splits.total_rows) + '</div></div></div>' +
      crossTable + '</div>';
  }

  /* ---------------------------------------------------------------- */
  /* render                                                            */
  /* ---------------------------------------------------------------- */

  app.spinner.show(containerEl);
  app.fetchJSON('/api/stats').then(function (d) {
    var html =
      '<h1>Statistics</h1>' +
      overview(d) +
      '<h3 style="margin-top:20px">SFT distributions</h3>' +
      distributions(d) +
      '<div style="margin-top:16px">' + perCardMatrix(d) + '</div>' +
      '<div style="margin-top:16px">' + ifdSection(d) + '</div>' +
      '<h3 style="margin-top:20px">Critique / safety / grounding</h3>' +
      critiqueSafety(d) +
      '<h3 style="margin-top:20px">KB (156-row bilingual cards)</h3>' +
      kbStats(d) +
      '<div style="margin-top:16px">' + anchorSection(d) + '</div>' +
      '<div style="margin-top:16px">' + spreadsSection(d) + '</div>' +
      '<div style="margin-top:16px">' + splitsSection(d) + '</div>';
    containerEl.innerHTML = html;
    app.spinner.hide(containerEl);
  }).catch(function (err) {
    app.spinner.hide(containerEl);
    var div = document.createElement('div');
    div.className = 'view-error';
    div.innerHTML = '<h2>Failed to load statistics</h2><p>' + esc(err && err.message ? err.message : String(err)) + '</p>';
    containerEl.appendChild(div);
  });
};
