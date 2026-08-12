/* tfvn-tarot dataset viewer — Reports view (B.7)
 *
 * Lists the six pipeline reports (GET /api/reports) and renders each one's
 * contents (GET /api/reports/{report_id}) with a per-report CURATED panel
 * plus a generic JSON renderer as the fallback for any unexpected shape
 * (plan B.7: "generic renderer is the fallback for any file" — it never
 * crashes: nested dicts -> tables, scalars -> key-value, lists of scalars ->
 * chips, lists of dicts -> tables, booleans -> ok/fail badges).
 *
 * Curated panels:
 *   filter_report    — layer keep-rate bars (kept/dropped per layer:
 *                      l1_dedup, l1_programmatic, l2_ifd, l3_deita,
 *                      l4_judge) + reasons table + tiers/input_rows +
 *                      acceptance flags
 *   coverage_report  — dedup cascade bars + acceptance checklist; the
 *                      known-failing flag `safety_pairs_ge_5_each: false`
 *                      renders AMBER (plan acceptance criterion)
 *   split_stats      — sizes + acceptance + per-split breakdown (splits.json
 *                      joined via /api/stats; splits_path shown from the list
 *                      entry metadata)
 *   ablation_report  — summary panels (conditions, floor verdict, baseline,
 *                      corpus metrics)
 *   w2_2_gate_report — gate summary: failed_gate = 0 and
 *                      negative_control_rejection_rate >= floor highlighted;
 *                      pass/fail banner derived from both
 *   spreads_discrimination_report — per-spread top-1 rates table
 *
 * Report selection switches content; each fetch shows the spinner; failures
 * toast (app.fetchJSON contract) and render an error panel. All interpolated
 * data is escaped via app.esc(). styles.css is untouched — a scoped <style>
 * block (reusing the :root tokens) covers the view's own layout.
 */
window.Views.reports = function (containerEl, app) {
  'use strict';

  var esc = app.esc;

  /* Short selector labels; the page h2 uses the backend's full title. */
  var SHORT = {
    filter_report: 'Filter report',
    coverage_report: 'Coverage report',
    split_stats: 'Split statistics',
    ablation_report: 'Ablation report',
    w2_2_gate_report: 'W2.2 gate report',
    spreads_discrimination_report: 'Spreads discrimination'
  };

  /* ---------------------------------------------------------------- */
  /* helpers                                                           */
  /* ---------------------------------------------------------------- */

  function num(v) { return typeof v === 'number' && isFinite(v) ? v : 0; }

  function fmt(v) { return num(v).toLocaleString('en-US'); }

  /* 4-decimal trimmed float; integers render plain. */
  function f4(v) {
    var n = num(v);
    if (n === Math.round(n)) return fmt(n);
    return String(n.toFixed(4)).replace(/0+$/, '').replace(/\.$/, '');
  }

  function pctRate(v) { return (num(v) * 100).toFixed(1) + '%'; }

  function badge(text, kind) {
    return '<span class="badge ' + (kind || '') + '">' + esc(text) + '</span>';
  }

  function chips(values) {
    if (!values || !values.length) return '<span class="faint">—</span>';
    return '<span class="rpt-chips">' + values.map(function (v) {
      return '<span class="rpt-chip">' + esc(String(v)) + '</span>';
    }).join('') + '</span>';
  }

  function panel(title, body) {
    return '<div class="panel"><div class="panel-title">' + esc(title) +
      '</div>' + body + '</div>';
  }

  function statCard(label, value) {
    return '<div class="stat-card"><div class="stat-value">' + value +
      '</div><div class="stat-label">' + esc(label) + '</div></div>';
  }

  function statCards(cells) {
    return '<div class="grid cols-4" style="margin-bottom:16px">' +
      cells.map(function (c) { return statCard(c[0], c[1]); }).join('') +
      '</div>';
  }

  function table(headers, rows) {
    if (!rows || !rows.length) {
      return '<div class="table-wrap"><table class="data"><tbody>' +
        '<tr><td class="muted">No rows.</td></tr></tbody></table></div>';
    }
    return '<div class="table-wrap"><table class="data"><thead><tr>' +
      headers.map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') +
      '</tr></thead><tbody>' + rows.join('') + '</tbody></table></div>';
  }

  /* key-value rows; values are pre-rendered (already escaped) HTML. */
  function kvTable(entries) {
    return table(['key', 'value'], entries.map(function (e) {
      return '<tr><td class="rpt-key">' + esc(e[0]) + '</td><td>' + e[1] + '</td></tr>';
    }));
  }

  /* A .bar-fill segment for the stacked keep/drop bar. */
  function segFill(pct, kind) {
    var w = pct > 0 ? pct.toFixed(2) + '%' : '0px';
    return '<span class="bar-fill" data-kind="' + esc(kind) + '"' +
      ' style="display:inline-block;min-width:0;width:' + w + '"></span>';
  }

  /* kept (ok) vs dropped (fail) stacked bar row. */
  function keepDropBar(label, kept, dropped) {
    var total = kept + dropped;
    var keptPct = total > 0 ? (kept / total) * 100 : 0;
    var droppedPct = total > 0 ? (dropped / total) * 100 : 0;
    return '<div class="bar">' +
      '<span class="bar-label" title="' + esc(label + ' — kept ' + fmt(kept) + ', dropped ' + fmt(dropped)) + '">' +
      esc(label) + '</span>' +
      '<span class="bar-track">' + segFill(keptPct, 'ok') + segFill(droppedPct, 'fail') + '</span>' +
      '<span class="bar-value" style="line-height:1.35">' +
      '<span style="color:var(--ok)">' + fmt(kept) + '</span><br>' +
      '<span style="color:var(--fail)">' + fmt(dropped) + '</span>' +
      '</span></div>';
  }

  /* Single-count bar relative to `max`. 0 count -> empty bar. */
  function countBar(label, count, max, kind) {
    var c = num(count);
    var w = c === 0 ? 'width:0;min-width:0' : 'width:' + ((c / max) * 100).toFixed(2) + '%';
    var kindAttr = kind ? ' data-kind="' + esc(kind) + '"' : '';
    return '<div class="bar">' +
      '<span class="bar-label" title="' + esc(label) + '">' + esc(label) + '</span>' +
      '<span class="bar-track"><span class="bar-fill"' + kindAttr + ' style="' + w + '"></span></span>' +
      '<span class="bar-value">' + fmt(c) + '</span></div>';
  }

  function boolBadge(v, amberOnFalse) {
    if (v === true) return badge('true', 'ok');
    if (v === false) return badge('false', amberOnFalse ? 'amber' : 'fail');
    return esc(String(v));
  }

  /* Acceptance checklist; amberOnFalse renders false flags AMBER (coverage
   * report: the known-failing safety_pairs_ge_5_each flag). */
  function acceptanceTable(flags, amberOnFalse) {
    var entries = [];
    for (var k in flags) {
      if (Object.prototype.hasOwnProperty.call(flags, k)) {
        entries.push([k, boolBadge(flags[k], amberOnFalse)]);
      }
    }
    return kvTable(entries);
  }

  function rawToggle(d) {
    var pretty;
    try { pretty = JSON.stringify(d, null, 2); } catch (e) { pretty = String(d); }
    return '<details class="rpt-raw"><summary>Raw JSON</summary>' +
      '<pre class="json-block">' + esc(pretty) + '</pre></details>';
  }

  /* ---------------------------------------------------------------- */
  /* generic JSON renderer (fallback — never crashes)                  */
  /* ---------------------------------------------------------------- */

  function renderObject(obj) {
    var keys = Object.keys(obj);
    if (!keys.length) return '<span class="faint">—</span>';
    var rows = keys.map(function (k) {
      return '<tr><td class="rpt-key">' + esc(k) + '</td><td>' +
        renderValue(obj[k]) + '</td></tr>';
    }).join('');
    return '<div class="table-wrap"><table class="data"><tbody>' + rows + '</tbody></table></div>';
  }

  function renderArray(arr) {
    if (!arr.length) return '<span class="faint">—</span>';
    var allScalar = arr.every(function (x) { return x === null || typeof x !== 'object'; });
    if (allScalar) return chips(arr);
    var allDicts = arr.every(function (x) { return x && typeof x === 'object' && !Array.isArray(x); });
    if (allDicts) {
      var keys = [];
      arr.forEach(function (d) {
        Object.keys(d).forEach(function (k) { if (keys.indexOf(k) === -1) keys.push(k); });
      });
      var rows = arr.map(function (d) {
        return '<tr>' + keys.map(function (k) {
          return '<td>' + renderValue(d[k]) + '</td>';
        }).join('') + '</tr>';
      }).join('');
      return '<div class="table-wrap"><table class="data"><thead><tr>' +
        keys.map(function (k) { return '<th>' + esc(k) + '</th>'; }).join('') +
        '</tr></thead><tbody>' + rows + '</tbody></table></div>';
    }
    /* mixed / nested arrays -> index table */
    var rows2 = arr.map(function (x, i) {
      return '<tr><td class="rpt-key">' + (i + 1) + '</td><td>' + renderValue(x) + '</td></tr>';
    }).join('');
    return '<div class="table-wrap"><table class="data"><tbody>' + rows2 + '</tbody></table></div>';
  }

  function renderValue(v) {
    if (v === null || typeof v === 'undefined') return '<span class="faint">—</span>';
    var t = typeof v;
    if (t === 'boolean') return boolBadge(v, false);
    if (t === 'number') return '<span class="rpt-num">' + f4(v) + '</span>';
    if (t === 'string') {
      return v.length > 400 ? '<span class="rpt-long">' + esc(v) + '</span>' : esc(v);
    }
    if (Array.isArray(v)) return renderArray(v);
    if (t === 'object') return renderObject(v);
    return esc(String(v));
  }

  function genericView(d) {
    return '<div class="rpt-generic"><h3>Report contents</h3>' +
      renderValue(d) +
      '<div class="small muted" style="margin-top:8px">' +
      'Generic JSON renderer — this report has no curated panel for its shape.</div>' +
      '</div>';
  }

  /* ---------------------------------------------------------------- */
  /* curated panels                                                    */
  /* ---------------------------------------------------------------- */

  function filterReport(d) {
    var layers = d.layers || {};
    var tiers = d.tiers || {};
    var inputRows = num(d.input_rows);
    var finalKept = layers.l4_judge ? num(layers.l4_judge.kept)
      : (layers.l3_deita ? num(layers.l3_deita.kept) : inputRows);
    var cards = statCards([
      ['Input rows', fmt(inputRows)],
      ['Core rows', fmt(tiers.core_rows)],
      ['Bulk rows', fmt(tiers.bulk_rows)],
      ['Output rows', fmt(finalKept)]
    ]);

    /* keep/drop per layer, derived deterministically: an explicit `dropped`
     * key wins; else previous kept - this kept (input_rows for the first
     * layer). l4_judge has no dropped key -> 13571 - 13471 = 100 (= its
     * decision_fail + judge_errors). */
    var order = ['l1_dedup', 'l1_programmatic', 'l2_ifd', 'l3_deita', 'l4_judge'];
    var prevKept = inputRows;
    var bars = [];
    order.forEach(function (name) {
      var ly = layers[name];
      if (!ly || typeof ly !== 'object') return;
      var hasKept = typeof ly.kept === 'number';
      var hasDropped = typeof ly.dropped === 'number';
      if (!hasKept && !hasDropped) return;
      var kept = hasKept ? num(ly.kept) : (prevKept - num(ly.dropped));
      var dropped = hasDropped ? num(ly.dropped) : (prevKept - kept);
      if (kept < 0) kept = 0;
      if (dropped < 0) dropped = 0;
      bars.push({ name: name, kept: kept, dropped: dropped });
      if (hasKept) prevKept = kept;
    });
    var barsHtml = bars.map(function (b) { return keepDropBar(b.name, b.kept, b.dropped); }).join('');
    var layersPanel = panel('Filter layer keep-rate (kept / dropped)',
      barsHtml || '<div class="muted small">No layer data.</div>');
    var details = layersDetails(layers);
    var acc = acceptanceTable(d.acceptance || {}, false);
    var outputs = chips(Array.isArray(d.outputs) ? d.outputs : []);
    return cards + layersPanel +
      (details ? '<div style="margin-top:16px">' + details + '</div>' : '') +
      '<div style="margin-top:16px">' + panel('Acceptance flags', acc) + '</div>' +
      '<div style="margin-top:16px">' + panel('Outputs', outputs) + '</div>' +
      '<div style="margin-top:16px">' + kvTable([['schema', esc(d.schema || '—')]]) + '</div>';
  }

  function layersDetails(layers) {
    var parts = [];

    var dedup = layers.l1_dedup;
    if (dedup && typeof dedup === 'object') {
      var dedupRows = ['exact_sha256', 'minhash_ge_0.85', 'embedding_ge_0.92', 'structural']
        .filter(function (k) { return k in dedup; })
        .map(function (k) { return [k, fmt(dedup[k])]; });
      if (dedupRows.length) {
        parts.push(panel('l1_dedup — cascade counters (dropped)', kvTable(dedupRows)));
      }
    }

    var prog = layers.l1_programmatic;
    if (prog && typeof prog === 'object') {
      if (prog.reasons && typeof prog.reasons === 'object') {
        var reasons = Object.keys(prog.reasons).map(function (k) {
          return [k, fmt(prog.reasons[k])];
        });
        parts.push(panel('l1_programmatic — drop reasons', kvTable(reasons)));
      }
      var warns = prog.collision_warns_deferred_to_l4_faithfulness;
      if (warns && typeof warns === 'object') {
        var warnChips = Object.keys(warns).map(function (k) {
          return k + ': ' + warns[k];
        });
        parts.push(panel('l1_programmatic — collision warns deferred to L4', chips(warnChips)));
      }
    }

    var l2 = layers.l2_ifd;
    if (l2 && typeof l2 === 'object') {
      parts.push(panel('l2_ifd', kvTable([
        ['keep_fraction', f4(l2.keep_fraction)],
        ["Cohen's d", f4(l2.cohens_d)],
        ['model', esc(l2.model || '—')],
        ['kept mean IFD', f4(l2.kept_mean_ifd)],
        ['rejected mean IFD', f4(l2.rejected_mean_ifd)],
        ['measurably_different', boolBadge(l2.measurably_different, false)],
        ['protected_pairs_restored', fmt(l2.protected_pairs_restored)]
      ])));
    }

    var l3 = layers.l3_deita;
    if (l3 && l3.method) {
      parts.push(panel('l3_deita — method', '<div class="small">' + esc(l3.method) + '</div>'));
    }

    var l4 = layers.l4_judge;
    if (l4 && typeof l4 === 'object') {
      var axisRows = [];
      var perAxis = l4.per_axis;
      if (perAxis && typeof perAxis === 'object') {
        axisRows = Object.keys(perAxis).map(function (ax) {
          var a = perAxis[ax] || {};
          return '<tr><td class="rpt-key">' + esc(ax) + '</td>' +
            '<td class="num">' + fmt(a.pass) + '</td>' +
            '<td class="num">' + fmt(a.fail) + '</td></tr>';
        });
      }
      parts.push(panel('l4_judge — per-axis pass/fail', table(['axis', 'pass', 'fail'], axisRows)));
      parts.push(panel('l4_judge', kvTable([
        ['judge', esc(l4.judge || '—')],
        ['decision_pass', fmt(l4.decision_pass)],
        ['decision_fail', fmt(l4.decision_fail)],
        ['judge_errors', fmt(l4.judge_errors)],
        ['gating_axes', chips(l4.gating_axes || [])],
        ['excluded_axes', chips(l4.excluded_axes || [])],
        ['protected_matched_pairs_kept_despite_fail', fmt(l4.protected_matched_pairs_kept_despite_fail)],
        ['systematic_defect', boolBadge(l4.systematic_defect, false)]
      ])));
    }

    if (!parts.length) return '';
    return '<div class="grid cols-2">' + parts.map(function (p) {
      return '<div>' + p + '</div>';
    }).join('') + '</div>';
  }

  function coverageReport(d) {
    var rowsIn = num(d.rows_in);
    var rowsAfter = num(d.rows_after_dedup);
    var cards = statCards([
      ['Rows in', fmt(rowsIn)],
      ['Rows after dedup', fmt(rowsAfter)],
      ['Core rows after dedup', fmt(d.core_rows_after_dedup)],
      ['Dedup removed', fmt(rowsIn - rowsAfter)]
    ]);

    var cascade = d.dedup_cascade || {};
    var cKeys = ['exact_sha256', 'minhash_ge_0.85', 'embedding_ge_0.92', 'structural']
      .filter(function (k) { return k in cascade; });
    var maxCascade = cKeys.length ? Math.max.apply(null, cKeys.map(function (k) { return num(cascade[k]); })) : 0;
    var barsHtml = cKeys.map(function (k) {
      return countBar(k, cascade[k], maxCascade || 1, num(cascade[k]) > 0 ? 'fail' : '');
    }).join('');
    var cascadePanel = panel('Dedup cascade (rows dropped per stage)',
      barsHtml || '<div class="muted small">No cascade data.</div>');

    var accPanel = panel('Coverage acceptance', acceptanceTable(d.acceptance || {}, true) +
      '<div class="small muted" style="margin-top:8px">Amber = known-failing flag accepted by the pipeline.</div>');

    var cov = d.coverage || {};
    var covPanels = '';
    if (cov.card_orientation_in_core && typeof cov.card_orientation_in_core === 'object') {
      var co = cov.card_orientation_in_core;
      covPanels += '<div style="margin-top:12px">' + panel('Card × orientation in core', kvTable([
        ['expected', fmt(co.expected)],
        ['present', fmt(co.present)],
        ['missing', chips(co.missing || [])]
      ])) + '</div>';
    }
    if (cov.spreads_in_core && typeof cov.spreads_in_core === 'object') {
      var sp = cov.spreads_in_core;
      covPanels += '<div style="margin-top:12px">' + panel('Spreads in core', kvTable([
        ['expected', fmt(sp.expected)],
        ['present', fmt(sp.present)],
        ['missing', chips(sp.missing || [])]
      ])) + '</div>';
    }

    var safety = cov.safety_pairs_per_category_in_core;
    var safetyHtml = '';
    if (safety && typeof safety === 'object') {
      var sRows = Object.keys(safety).map(function (k) {
        return '<tr><td class="rpt-key">' + esc(k) + '</td><td>' +
          (num(safety[k]) > 0 ? fmt(safety[k]) : '<span class="faint">' + fmt(safety[k]) + '</span>') +
          '</td></tr>';
      });
      safetyHtml = '<div style="margin-top:16px">' + panel('Safety pairs per category (in core)', table(['category', 'pairs'], sRows)) + '</div>';
    }

    var meta = panel('Metadata', kvTable([
      ['hash', '<span class="mono">' + esc(d.hash || '—') + '</span>'],
      ['hash_file', esc(d.hash_file || '—')],
      ['schema', esc(d.schema || '—')]
    ]));

    return cards + cascadePanel +
      '<div style="margin-top:16px">' + accPanel + '</div>' + safetyHtml +
      '<div style="margin-top:16px">' + covPanels + '</div>' +
      '<div style="margin-top:16px">' + meta + '</div>';
  }

  function splitsBreakdown(splits) {
    if (!splits || typeof splits !== 'object') return '';
    var counts = splits.counts || {};
    var byTask = splits.by_task_type || {};
    var splitOrder = ['train', 'val', 'test'].filter(function (s) {
      return byTask[s] && typeof byTask[s] === 'object';
    });
    var taskTypes = [];
    splitOrder.forEach(function (s) {
      Object.keys(byTask[s]).forEach(function (t) {
        if (taskTypes.indexOf(t) === -1) taskTypes.push(t);
      });
    });
    var rows = splitOrder.map(function (s) {
      return '<tr><td class="rpt-key">' + esc(s) + '</td>' +
        taskTypes.map(function (t) { return '<td class="num">' + fmt(byTask[s][t]) + '</td>'; }).join('') +
        '<td class="num">' + fmt(counts[s]) + '</td></tr>';
    });
    var info = '<div class="small muted" style="margin-top:8px">unmatched: ' +
      fmt(splits.unmatched_rows) + ' · total rows: ' + fmt(splits.total_rows) + '</div>';
    return panel('Per-split breakdown (splits.json × task_type)',
      table(['split'].concat(taskTypes, ['total']), rows) + info);
  }

  function splitStats(d, splits) {
    var sizes = d.sizes || {};
    var sKeys = ['train', 'val', 'test'].filter(function (k) { return k in sizes; });
    var total = sKeys.reduce(function (s, k) { return s + num(sizes[k]); }, 0);
    var maxSize = sKeys.length ? Math.max.apply(null, sKeys.map(function (k) { return num(sizes[k]); })) : 0;
    var cards = statCards(sKeys.map(function (k) {
      return [k, fmt(sizes[k])];
    }).concat([['total', fmt(total)]]));
    var barsHtml = sKeys.map(function (k) { return countBar(k, sizes[k], maxSize || 1, ''); }).join('');
    var note = '<div class="small muted" style="margin-top:8px">Per-split breakdown computed from ' +
      esc(d.splits_path || 'datasets/splits.json') + ' (via /api/stats).</div>';
    return cards +
      '<div style="margin-top:16px">' + panel('Split sizes', barsHtml) + '</div>' +
      '<div style="margin-top:16px">' + panel('Split acceptance', acceptanceTable(d.acceptance || {}, false)) + '</div>' +
      '<div style="margin-top:16px">' + panel('Test task types', chips(Array.isArray(d.test_task_types) ? d.test_task_types : [])) + '</div>' +
      '<div style="margin-top:16px">' + (splitsBreakdown(splits) || panel('Per-split breakdown', '<div class="muted small">Unavailable — /api/stats did not provide a splits section.</div>')) + note + '</div>' +
      '<div style="margin-top:16px">' + panel('Metadata', kvTable([
        ['anchor_file', esc(d.anchor_file || '—')],
        ['schema', esc(d.schema || '—')]
      ])) + '</div>';
  }

  function ablationReport(d) {
    var a = d.ablation || {};
    var floor = num(a.floor);
    var cards = statCards([
      ['Candidates', fmt(a.n_candidates)],
      ['Kept (none)', fmt(a.kept_none)],
      ['Kept (memory)', fmt(a.kept_memory)],
      ['Kept (all-three)', fmt(a.kept_all_three)]
    ]);
    var caveat = d.caveat
      ? '<div class="banner banner-warn" style="margin-bottom:16px">' + esc(d.caveat) + '</div>' : '';

    var condRows = [
      ['none', a.kept_none, a.distinct2_none],
      ['memory-only', a.kept_memory, a.distinct2_memory_only],
      ['all-three', a.kept_all_three, a.distinct2_all_three]
    ].map(function (c) {
      return '<tr><td class="rpt-key">' + esc(c[0]) + '</td>' +
        '<td class="num">' + fmt(c[1]) + '</td>' +
        '<td class="num">' + f4(c[2]) + '</td>' +
        '<td>' + boolBadge(num(c[2]) >= floor, false) + '</td></tr>';
    });
    var condPanel = panel('Filter-layer conditions',
      table(['condition', 'kept', 'distinct-2', '≥ floor ' + f4(floor)], condRows));

    var accPanel = panel('Acceptance', acceptanceTable(d.acceptance || {}, false));

    var verdict = d.floor_verdict;
    var verdictHtml = '';
    if (verdict && typeof verdict === 'object') {
      var vrows = [];
      if ('verdict' in verdict) {
        vrows.push(['verdict', badge(verdict.verdict, verdict.verdict === 'PASS' ? 'ok' : 'fail')]);
      }
      if ('criterion' in verdict) {
        vrows.push(['criterion', '<span class="small">' + esc(verdict.criterion) + '</span>']);
      }
      ['baseline_distinct2_200_sample', 'baseline_per_100_max', 'corpus_distinct2_random_200',
       'corpus_per_100_min', 'corpus_ge_baseline_200_sample', 'corpus_ge_baseline_every_window']
        .forEach(function (k) {
          if (k in verdict) {
            vrows.push([k, typeof verdict[k] === 'boolean' ? boolBadge(verdict[k], false) : f4(verdict[k])]);
          }
        });
      verdictHtml = '<div style="margin-top:16px">' + panel('Floor verdict', kvTable(vrows)) + '</div>';
    }

    var baseline = d.base_model_baseline;
    var baseHtml = '';
    if (baseline && typeof baseline === 'object') {
      var brows = [
        ['model', esc(baseline.model || '—')],
        ['n', fmt(baseline.n)],
        ['temperature', f4(baseline.temperature)],
        ['distinct2_200_sample', f4(baseline.distinct2_200_sample)],
        ['distinct2_per_100_windows', chips(baseline.distinct2_per_100_windows || [])]
      ];
      if (baseline.note) brows.push(['note', '<span class="small">' + esc(baseline.note) + '</span>']);
      baseHtml = '<div style="margin-top:16px">' + panel('Base-model baseline (no anti-collapse stack)', kvTable(brows)) + '</div>';
    }

    var corpus = d.corpus_metrics;
    var corpusHtml = '';
    if (corpus && typeof corpus === 'object') {
      var crows = [];
      ['distinct2_random_200', 'distinct2_first_200', 'distinct2_per_100_min'].forEach(function (k) {
        if (k in corpus) crows.push([k, f4(corpus[k])]);
      });
      if (Array.isArray(corpus.distinct2_per_100_windows)) {
        crows.push(['distinct2_per_100_windows', chips(corpus.distinct2_per_100_windows)]);
      }
      if ('corpus_internal_replay_dedup_rate' in corpus) {
        crows.push(['corpus_internal_replay_dedup_rate',
          boolBadge(num(corpus.corpus_internal_replay_dedup_rate) === 0, false) + ' ' + f4(corpus.corpus_internal_replay_dedup_rate)]);
      }
      if ('blacklist_forbidden_phrases' in corpus) crows.push(['blacklist_forbidden_phrases', fmt(corpus.blacklist_forbidden_phrases)]);
      if ('main_loop_rows' in corpus) crows.push(['main_loop_rows', fmt(corpus.main_loop_rows)]);
      if ('memory_index_size' in corpus) crows.push(['memory_index_size', fmt(corpus.memory_index_size)]);
      if (corpus.corpus_internal_replay_dedup_note) {
        crows.push(['note', '<span class="small">' + esc(corpus.corpus_internal_replay_dedup_note) + '</span>']);
      }
      corpusHtml = '<div style="margin-top:16px">' + panel('Corpus metrics (distinct-2, replay dedup)', kvTable(crows)) + '</div>';
    }

    var metaRows = [];
    if (d.method) metaRows.push(['method', esc(d.method)]);
    if (d.schema) metaRows.push(['schema', esc(d.schema)]);
    var metaHtml = metaRows.length
      ? '<div style="margin-top:16px">' + panel('Metadata', kvTable(metaRows)) + '</div>' : '';

    return caveat + cards +
      '<div style="margin-top:16px">' + condPanel + '</div>' +
      '<div style="margin-top:16px">' + accPanel + '</div>' +
      verdictHtml + baseHtml + corpusHtml + metaHtml;
  }

  function jaccardHistogram(dist) {
    var pairs = (dist && Array.isArray(dist.pairs)) ? dist.pairs : [];
    if (!pairs.length) return '<div class="muted small">No pair distribution.</div>';
    var vals = pairs.map(function (p) { return num(p.jaccard); });
    var min = Math.min.apply(null, vals);
    var max = Math.max.apply(null, vals);
    var lo = Math.floor(min * 20) / 20;
    var hi = Math.ceil(max * 20) / 20;
    if (hi - lo < 0.05) hi = lo + 0.05;
    var nbins = Math.max(1, Math.round((hi - lo) / 0.05));
    var bins = [];
    for (var i = 0; i < nbins; i++) bins.push(0);
    vals.forEach(function (v) {
      var idx = Math.min(nbins - 1, Math.max(0, Math.floor((v - lo) / 0.05)));
      bins[idx]++;
    });
    var maxBin = Math.max.apply(null, bins);
    var labels = bins.map(function (_, i) {
      return (lo + i * 0.05).toFixed(2) + '–' + (lo + (i + 1) * 0.05).toFixed(2);
    });
    return labels.map(function (l, i) {
      return countBar(l, bins[i], maxBin || 1, '');
    }).join('');
  }

  function perCardTable(perCard) {
    if (!perCard || typeof perCard !== 'object') return '';
    var ids = Object.keys(perCard).sort(function (a, b) { return num(a) - num(b); });
    var rows = ids.map(function (id) {
      var c = perCard[id] || {};
      var kw = Array.isArray(c.keywords_vi) ? c.keywords_vi : [];
      var kwChips = kw.slice(0, 4).map(function (k) {
        return '<span class="rpt-chip">' + esc(k) + '</span>';
      }).join('');
      if (kw.length > 4) kwChips += '<span class="rpt-chip faint">+' + (kw.length - 4) + '</span>';
      return '<tr><td class="num">' + esc(id) + '</td><td>' + esc(c.name_en || '—') + '</td>' +
        '<td class="num">' + fmt(c.attempts_used) + '</td><td>' + esc(c.attribution || '—') + '</td>' +
        '<td>' + (kwChips || '<span class="faint">—</span>') + '</td></tr>';
    });
    return panel('Per-card synthesis (' + ids.length + ' cards)',
      table(['card', 'name_en', 'attempts', 'attribution', 'keywords_vi'], rows));
  }

  function w22Gate(d) {
    var agg = d.aggregate || {};
    var rate = num(d.negative_control_rejection_rate);
    var floor = num(d.negative_control_floor);
    var failedGateOk = num(agg.failed_gate) === 0;
    var rateOk = rate >= floor;
    var pass = failedGateOk && rateOk;
    var banner = pass
      ? '<div class="banner banner-ok">PASS — failed_gate = ' + fmt(agg.failed_gate) +
        ' and negative_control_rejection_rate = ' + f4(rate) + ' ≥ floor ' + f4(floor) + '.</div>'
      : '<div class="banner banner-error">FAIL — failed_gate = ' + fmt(agg.failed_gate) +
        ' (need 0) and negative_control_rejection_rate = ' + f4(rate) +
        ' (need ≥ ' + f4(floor) + ').</div>';
    var cards = statCards([
      ['Cards processed', fmt(agg.cards_processed)],
      ['Synthetic', fmt(agg.synthetic)],
      ['Failed gate', badge(fmt(agg.failed_gate), failedGateOk ? 'ok' : 'fail')],
      ['Negative-control rate', badge(f4(rate), rateOk ? 'ok' : 'fail')]
    ]);

    var nc = Array.isArray(d.negative_control) ? d.negative_control : [];
    var rejected = nc.filter(function (x) { return x && x.rejected; }).length;
    var ncSummary = panel('Negative control (' + nc.length + ' samples)', kvTable([
      ['rejected', fmt(rejected)],
      ['rejection rate', f4(rate)],
      ['floor', f4(floor)],
      ['met', badge(rateOk ? 'true' : 'false', rateOk ? 'ok' : 'fail')]
    ]));

    var dist = d.authentic_pair_distribution || {};
    var distPanel = panel('Authentic upright/reversed Jaccard distribution',
      '<div class="grid cols-4" style="margin-bottom:10px">' +
      statCard('count', fmt(dist.count)) +
      statCard('max', f4(dist.max)) +
      statCard('percentile', f4(dist.percentile)) +
      statCard('threshold (p90)', f4(dist.threshold)) +
      '</div>' + jaccardHistogram(dist));

    var g1 = d.g1_calibration || {};
    var g1Panel = panel('G1 Vietnamese-ness calibration', kvTable([
      ['authentic_profile_n', fmt(g1.authentic_profile_n)],
      ['max_profile_distance (p90)', f4(g1.max_profile_distance)],
      ['method', esc(g1.method || '—')]
    ]));

    var miscRows = [];
    if ('jaccard_threshold' in d) miscRows.push(['jaccard_threshold', f4(d.jaccard_threshold)]);
    if ('max_retries' in d) miscRows.push(['max_retries', fmt(d.max_retries)]);
    if ('model' in d) miscRows.push(['model', esc(d.model)]);
    if ('seed' in d) miscRows.push(['seed', fmt(d.seed)]);
    if (d.temperatures && typeof d.temperatures === 'object') {
      miscRows.push(['temperatures', kvTable(Object.keys(d.temperatures).map(function (k) {
        return [k, f4(d.temperatures[k])];
      }))]);
    }
    var miscHtml = miscRows.length
      ? '<div style="margin-top:16px">' + panel('Parameters', kvTable(miscRows)) + '</div>' : '';

    var perCard = perCardTable(d.per_card);
    return banner +
      '<div style="margin-top:16px">' + cards + '</div>' +
      '<div style="margin-top:16px">' + ncSummary + '</div>' +
      '<div style="margin-top:16px">' + distPanel + '</div>' +
      '<div style="margin-top:16px">' + g1Panel + '</div>' +
      (perCard ? '<div style="margin-top:16px">' + perCard + '</div>' : '') +
      miscHtml;
  }

  function spreadsDisc(d) {
    var cards = statCards([
      ['Spreads', fmt(d.n_spreads)],
      ['Positions pooled', fmt(d.n_positions_pooled)],
      ['Overall top-1 rate', pctRate(d.overall_top1_rate)],
      ['Chance rate', pctRate(d.chance_rate)]
    ]);

    var per = d.per_spread || {};
    var names = Object.keys(per);
    var maxRate = names.length ? Math.max.apply(null, names.map(function (n) { return num(per[n].top1_rate); })) : 0;
    var rows = names.map(function (n) {
      var s = per[n] || {};
      var rate = num(s.top1_rate);
      var w = rate === 0 ? 'width:0;min-width:0' : 'width:' + ((rate / (maxRate || 1)) * 100).toFixed(2) + '%';
      var kind = rate >= 0.8 ? 'ok' : (rate >= 0.5 ? 'amber' : 'fail');
      return '<tr><td>' + esc(n) + '</td><td class="num">' + fmt(s.n_positions) + '</td>' +
        '<td class="num">' + fmt(s.top1_correct) + '</td>' +
        '<td><span class="bar-track" style="display:inline-block;vertical-align:middle;width:64px">' +
        '<span class="bar-fill" data-kind="' + kind + '" style="' + w + '"></span></span>' +
        '<span class="bar-value" style="display:inline-block;min-width:44px;text-align:right">' +
        pctRate(rate) + '</span></td>' +
        '<td>' + boolBadge(s.above_chance, false) + '</td></tr>';
    });
    var summary = panel('Per-spread top-1 rates',
      table(['spread', 'positions', 'correct', 'top-1 rate', 'above chance'], rows));

    var above = panel('Above-chance summary', kvTable([
      ['spreads_above_chance', fmt(d.spreads_above_chance)],
      ['required', fmt(d.spreads_above_chance_required)],
      ['met', badge(d.spreads_above_chance >= d.spreads_above_chance_required ? 'true' : 'false',
        d.spreads_above_chance >= d.spreads_above_chance_required ? 'ok' : 'fail')]
    ]));

    var extra = '';
    if (Array.isArray(d.failing_spreads)) {
      extra = '<div style="margin-top:16px">' + panel('Failing spreads',
        d.failing_spreads.length ? chips(d.failing_spreads) : '<span class="ok">none</span>') + '</div>';
    }

    return cards +
      '<div style="margin-top:16px">' + summary + '</div>' +
      '<div style="margin-top:16px">' + above + '</div>' + extra;
  }

  function curatedFor(id, d, splits) {
    try {
      if (id === 'filter_report' && d && typeof d.layers === 'object') return filterReport(d);
      if (id === 'coverage_report' && d && typeof d.dedup_cascade === 'object') return coverageReport(d);
      if (id === 'split_stats' && d && typeof d.sizes === 'object') return splitStats(d, splits);
      if (id === 'ablation_report' && d && typeof d.ablation === 'object') return ablationReport(d);
      if (id === 'w2_2_gate_report' && d && typeof d.aggregate === 'object') return w22Gate(d);
      if (id === 'spreads_discrimination_report' && d && typeof d.per_spread === 'object') return spreadsDisc(d);
    } catch (err) {
      console.error('reports: curated panel "' + id + '" failed — falling back to generic renderer:', err);
    }
    return genericView(d);
  }

  /* ---------------------------------------------------------------- */
  /* view layout + data flow                                           */
  /* ---------------------------------------------------------------- */

  var styleEl = document.createElement('style');
  styleEl.textContent = [
    '.rpt-layout{display:grid;grid-template-columns:240px 1fr;gap:16px;align-items:start}',
    '.rpt-side{position:sticky;top:64px;display:flex;flex-direction:column;gap:6px;background:var(--bg-panel);border:1px solid var(--border);border-radius:var(--radius);padding:12px}',
    '.rpt-side-title{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-dim);margin:0 0 6px}',
    '.rpt-item{width:100%;text-align:left;background:var(--bg-inset);border:1px solid var(--border);border-radius:var(--radius-sm);padding:7px 10px;cursor:pointer;color:var(--text);display:flex;flex-direction:column;gap:2px;font-family:var(--sans)}',
    '.rpt-item:hover{border-color:var(--border-strong);background:var(--bg-raised)}',
    '.rpt-item.active{border-color:var(--accent);background:var(--bg-raised)}',
    '.rpt-item-name{font-size:13px;font-weight:600}',
    '.rpt-item.active .rpt-item-name{color:var(--accent)}',
    '.rpt-item-path{font-family:var(--mono);font-size:10px;color:var(--text-faint);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
    '.rpt-main{min-width:0}',
    '.rpt-title{font-size:18px;margin:0 0 12px}',
    '.rpt-key{font-family:var(--mono);font-size:12px;color:var(--text-dim);white-space:nowrap}',
    '.rpt-chips{display:inline-flex;flex-wrap:wrap;gap:4px}',
    '.rpt-chip{display:inline-block;padding:0 7px;border-radius:999px;border:1px solid var(--border-strong);background:var(--bg-raised);color:var(--text-dim);font-family:var(--mono);font-size:10px;line-height:1.7;white-space:nowrap}',
    '.rpt-num{font-variant-numeric:tabular-nums}',
    '.rpt-long{overflow-wrap:anywhere}',
    '.rpt-raw{margin-top:16px}',
    '.rpt-raw summary{cursor:pointer;font-size:12px;color:var(--text-dim)}',
    '.rpt-raw .json-block{margin-top:8px}',
    '@media(max-width:860px){.rpt-layout{grid-template-columns:1fr}.rpt-side{position:static}}'
  ].join('\n');
  document.head.appendChild(styleEl);

  var state = { token: 0 };

  function setActive(id) {
    var items = containerEl.querySelectorAll('.rpt-item');
    for (var i = 0; i < items.length; i++) {
      if (items[i].getAttribute('data-report') === id) items[i].classList.add('active');
      else items[i].classList.remove('active');
    }
  }

  /* GET JSON without the toast-on-error behaviour (best-effort companion
   * fetch — e.g. /api/stats for the split_stats breakdown). */
  function silentJSON(path) {
    return fetch(path, { headers: { Accept: 'application/json' } })
      .then(function (res) { return res.ok ? res.json() : null; })
      .catch(function () { return null; });
  }

  function loadReport(id) {
    state.token++;
    var myToken = state.token;
    var body = containerEl.querySelector('.rpt-body');
    var title = containerEl.querySelector('.rpt-title');
    for (var i = 0; i < state.reports.length; i++) {
      if (state.reports[i].id === id) title.textContent = state.reports[i].title || id;
    }
    app.spinner.show(body);

    var p = app.fetchJSON('/api/reports/' + encodeURIComponent(id));
    if (id === 'split_stats') {
      p = Promise.all([p, silentJSON('/api/stats')]).then(function (r) {
        return { report: r[0], stats: r[1] };
      });
    }
    p.then(function (res) {
      if (myToken !== state.token) return;
      var d = (res && res.report) ? res.report : res;
      var splits = (res && res.stats) ? (res.stats.splits || null) : null;
      app.spinner.hide(body);
      body.innerHTML = curatedFor(id, d, splits) + rawToggle(d);
    }).catch(function (err) {
      if (myToken !== state.token) return;
      app.spinner.hide(body);
      body.innerHTML = '<div class="view-error"><h2>Failed to load report</h2><p>' +
        esc(err && err.message ? err.message : String(err)) + '</p></div>';
    });
  }

  function renderSelector(list) {
    state.reports = list;
    var html = '<div class="rpt-layout">' +
      '<aside class="rpt-side">' +
      '<h3 class="rpt-side-title">Reports</h3>' +
      list.map(function (r) {
        return '<button type="button" class="rpt-item" data-report="' + esc(r.id) + '"' +
          ' title="' + esc(r.title || '') + '">' +
          '<span class="rpt-item-name">' + esc(SHORT[r.id] || r.id) + '</span>' +
          '<span class="rpt-item-path">' + esc(r.path || '') + '</span>' +
          '</button>';
      }).join('') +
      '</aside>' +
      '<section class="rpt-main">' +
      '<h2 class="rpt-title"></h2>' +
      '<div class="rpt-body"></div>' +
      '</section>' +
      '</div>';
    containerEl.innerHTML = html;
    var items = containerEl.querySelectorAll('.rpt-item');
    for (var i = 0; i < items.length; i++) {
      items[i].addEventListener('click', (function (id) {
        return function () { setActive(id); loadReport(id); };
      })(items[i].getAttribute('data-report')));
    }
  }

  app.spinner.show(containerEl);
  app.fetchJSON('/api/reports').then(function (data) {
    app.spinner.hide(containerEl);
    var list = (data && Array.isArray(data.reports)) ? data.reports : [];
    if (!list.length) {
      containerEl.innerHTML = '<div class="view-error"><h2>No reports available</h2>' +
        '<p>The /api/reports listing returned no entries.</p></div>';
      return;
    }
    renderSelector(list);
    var first = list[0].id;
    setActive(first);
    loadReport(first);
  }).catch(function (err) {
    app.spinner.hide(containerEl);
    containerEl.innerHTML = '<div class="view-error"><h2>Failed to load reports</h2><p>' +
      esc(err && err.message ? err.message : String(err)) + '</p></div>';
  });
};



