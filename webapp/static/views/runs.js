/* tfvn-tarot dataset viewer — Runs view (B.8)
 *
 * Safety-critical view: it triggers pipeline re-runs, some billed. Reads
 * /api/runs (whitelist + running + orphaned_pending) and /api/runs/history;
 * drives POST /api/runs/{id} (start, tiered confirm), POST /api/runs/{id}/kill,
 * POST /api/runs/{id}/ack (orphan acknowledgement) and
 * GET /api/runs/{id}/log?offset=N (polled every 800 ms while running).
 *
 * Tiered confirm contract:
 *   safe   -> single Run button (one click starts; preflight gates it)
 *   slow   -> Run enabled only after a confirmation checkbox
 *   billed -> confirmation checkbox + typed acknowledgement phrase; the POST
 *             body carries accept_cost=true (the server re-enforces this: a
 *             billed start without accept_cost -> 409, toasted)
 * Run buttons are disabled while ANY run is active (single-flight; a stale
 * page that races another session gets a 409 from the server, surfaced as a
 * "a run is in progress" toast).
 *
 * On completion the view toasts, refreshes itself and dispatches an
 * `app:data-changed` CustomEvent on window so other views (stats / catalog /
 * hashcheck) can re-read the inputs a run just rewrote.
 */
(function () {
  'use strict';

  window.Views = window.Views || {};
  window.Views.runs = function (containerEl, app) {
    containerEl.setAttribute('data-view', 'runs');
    containerEl.innerHTML = '';
    app.spinner.show(containerEl);

    /* ---------------------------------------------------------------- */
    /* view-specific styles (theme tokens only; styles.css untouched)    */
    /* ---------------------------------------------------------------- */
    var style = document.createElement('style');
    style.textContent = [
      '.runs-page-hdr { display:flex; align-items:baseline; gap:12px; justify-content:space-between; }',
      '.runs-group { margin-bottom:22px; }',
      '.runs-tier-hdr { display:flex; align-items:center; gap:10px; padding:5px 0 8px 10px; border-bottom:1px solid var(--border); margin-bottom:10px; border-left:3px solid var(--border-strong); }',
      '.runs-tier-safe { border-left-color: rgba(92,192,138,.55); }',
      '.runs-tier-slow { border-left-color: rgba(217,166,79,.55); }',
      '.runs-tier-billed { border-left-color: rgba(180,140,232,.55); }',
      '.runs-cards { display:grid; gap:12px; }',
      '.runs-card { border-left:3px solid var(--border-strong); }',
      '.runs-card-safe { border-left-color: rgba(92,192,138,.55); }',
      '.runs-card-slow { border-left-color: rgba(217,166,79,.55); }',
      '.runs-card-billed { border-left-color: rgba(180,140,232,.55); }',
      '.runs-card-hdr { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }',
      '.runs-card-hdr h3 { margin:0; }',
      '.runs-sub { font-size:11px; color:var(--text-faint); text-transform:uppercase; letter-spacing:.04em; margin:10px 0 4px; }',
      '.runs-argv { background:var(--bg-inset); border:1px solid var(--border); border-radius:var(--radius-sm); padding:6px 8px; font-size:12px; color:var(--text-dim); overflow-x:auto; white-space:pre; }',
      '.runs-preflight { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:8px; }',
      '.runs-pf-reason { width:100%; font-size:12px; color:var(--fail); }',
      '.runs-options { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-top:10px; }',
      '.runs-opt label { font-family:var(--mono); font-size:11px; }',
      '.runs-fresh { border:1px dashed var(--border-strong); border-radius:var(--radius-sm); padding:6px 8px; display:flex; flex-direction:column; gap:4px; }',
      '.runs-fresh-warn { font-size:12px; color:var(--amber); }',
      '.runs-confirm { display:flex; flex-direction:column; gap:8px; margin-top:12px; padding-top:10px; border-top:1px solid var(--border); }',
      '.runs-confirm-lbl { display:flex; align-items:center; gap:8px; cursor:pointer; font-size:13px; }',
      '.runs-cost { display:flex; align-items:center; gap:6px; }',
      '.runs-cost input { flex:1; max-width:340px; }',
      '.runs-modifies { margin-top:10px; font-size:12px; color:var(--amber); border:1px solid rgba(217,166,79,.35); background:rgba(217,166,79,.07); border-radius:var(--radius-sm); padding:6px 8px; }',
      '.runs-actions { display:flex; align-items:center; gap:10px; margin-top:12px; flex-wrap:wrap; }',
      '.runs-hint { font-size:12px; color:var(--amber); }',
      '.runs-live { border-left:3px solid var(--amber); }',
      '.runs-live-hdr { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }',
      '.runs-status { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:8px; font-size:12px; }',
      '.runs-log-wrap { margin-top:10px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--bg-inset); }',
      '.runs-log { max-height:320px; overflow-y:auto; padding:8px 10px; font-size:12px; line-height:1.5; }',
      '.runs-line { white-space:pre-wrap; word-break:break-word; color:var(--text-dim); }',
      '.runs-trunc { font-size:12px; color:var(--amber); padding:4px 10px; }',
      '.runs-live-actions { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }',
      '.runs-kill { border-color: var(--fail); color: var(--fail); background: rgba(224,101,92,.08); }',
      '.runs-orphan button { margin-left:auto; }'
    ].join('\n');
    containerEl.appendChild(style);

    /* ---------------------------------------------------------------- */
    /* constants + state                                                 */
    /* ---------------------------------------------------------------- */
    var COST_PHRASE = 'I understand this costs money';
    var POLL_MS = 800;
    var KILL_COPY = 'stops local processing; in-flight API calls may continue up to 120 s';

    var TIER_META = {
      safe: { badge: 'badge-safe', label: 'safe', title: 'offline \u2014 no API calls' },
      slow: { badge: 'badge-slow', label: 'slow', title: 'local compute \u2014 may take a while' },
      billed: { badge: 'billed', label: '$ billed', title: 'calls the paid LLM API \u2014 costs money' }
    };

    var STATUS_META = {
      running: { cls: 'amber', text: 'running' },
      completed: { cls: 'ok', text: 'completed' },
      failed: { cls: 'fail', text: 'failed' },
      killed: { cls: 'fail', text: 'killed' },
      timed_out: { cls: 'fail', text: 'timed out' },
      orphaned: { cls: 'amber', text: 'orphaned' }
    };

    var state = {
      scripts: [],
      byId: {},
      running: null,   // {run_id, script_id} | null
      orphaned: [],    // run_ids awaiting acknowledgement
      history: []
    };
    var live = null;   // live-pane handle
    var cards = [];    // per-card control handles

    var orphanHost, liveHost, cardsHost, historyHost;

    /* ---------------------------------------------------------------- */
    /* DOM + formatting helpers                                          */
    /* ---------------------------------------------------------------- */
    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text != null) node.textContent = String(text);
      return node;
    }

    function fmtDuration(s) {
      if (s == null || isNaN(s)) return '\u2014';
      if (s < 60) return Number(s).toFixed(1) + ' s';
      var m = Math.floor(s / 60);
      var r = Math.round(s % 60);
      return m + ' m ' + r + ' s';
    }

    function fmtTime(iso) {
      if (!iso) return '\u2014';
      var d = new Date(iso);
      if (isNaN(d.getTime())) return String(iso);
      return d.toLocaleString();
    }

    function runShort(id) {
      return id ? String(id).slice(0, 16) : '\u2014';
    }

    function shaShort(head) {
      return head && head !== 'unknown' ? String(head).slice(0, 8) : null;
    }

    function statusMeta(status) {
      return STATUS_META[status] || { cls: 'badge', text: String(status || 'unknown') };
    }

    /* POST helper — app.fetchJSON is GET-only; views own their mutating
     * calls. Same error contract: toast FastAPI detail + throw. */
    function postJSON(path, body) {
      var req;
      try {
        req = fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify(body || {})
        });
      } catch (err) {
        var netMsg = 'Network error POSTing ' + path + ': ' +
          (err && err.message ? err.message : String(err));
        app.toast(netMsg);
        var netErr = new Error(netMsg);
        netErr.status = 0;
        return Promise.reject(netErr);
      }
      return req.then(function (res) {
        if (!res.ok) {
          return res.json().then(function (b) {
            var detail = 'HTTP ' + res.status + ' ' + res.statusText;
            if (b && typeof b.detail !== 'undefined') detail = formatDetail(b.detail);
            else if (b && typeof b.message !== 'undefined') detail = String(b.message);
            var msg = path + ' \u2192 ' + detail;
            app.toast(msg);
            var err = new Error(msg);
            err.status = res.status;
            throw err;
          }, function () {
            var msg2 = path + ' \u2192 HTTP ' + res.status + ' ' + res.statusText;
            app.toast(msg2);
            var err2 = new Error(msg2);
            err2.status = res.status;
            throw err2;
          });
        }
        return res.json().catch(function () { return {}; });
      });
    }

    function formatDetail(detail) {
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        return detail.map(function (d) {
          if (!d || typeof d !== 'object') return String(d);
          var loc = (d.loc || []).filter(function (x) {
            return x !== 'query' && x !== 'path' && x !== 'body' && x !== 'header';
          }).join('.');
          return (loc ? loc + ': ' : '') + (d.msg || d.type || JSON.stringify(d));
        }).join('; ');
      }
      if (detail && typeof detail === 'object') {
        try { return JSON.stringify(detail); } catch (_) { return String(detail); }
      }
      return String(detail);
    }

    /* ---------------------------------------------------------------- */
    /* data fetch + page render                                          */
    /* ---------------------------------------------------------------- */
    function fetchAll() {
      var pending = 2;

      function done() {
        app.spinner.hide(containerEl);
        renderPage();
        if (state.running) {
          var spec = state.byId[state.running.script_id] || {};
          startLive(state.running.run_id, state.running.script_id,
            spec.label || state.running.script_id);
        }
      }

      app.fetchJSON('/api/runs').then(function (d) {
        state.scripts = (d && d.scripts) || [];
        state.byId = {};
        state.scripts.forEach(function (s) { state.byId[s.script_id] = s; });
        state.running = (d && d.running) || null;
        state.orphaned = (d && d.orphaned_pending) || [];
      }, function () { /* toasted by fetchJSON; keep stale slice */ })
        .then(function () { if (--pending === 0) done(); });

      app.fetchJSON('/api/runs/history').then(function (d) {
        state.history = (d && d.runs) || [];
      }, function () { /* toasted; keep stale history */ })
        .then(function () { if (--pending === 0) done(); });
    }

    function renderPage() {
      renderOrphan();
      renderCards();
      renderHistory();
    }

    /* ---------------------------------------------------------------- */
    /* orphaned-run banner                                               */
    /* ---------------------------------------------------------------- */
    function ackOrphan(runId) {
      postJSON('/api/runs/' + runId + '/ack', {}).then(function () {
        app.toast('Orphaned run ' + runId + ' acknowledged', 'ok');
        fetchAll();
      }, function () { /* toasted by postJSON */ });
    }

    function renderOrphan() {
      orphanHost.innerHTML = '';
      if (!state.orphaned.length) return;
      var banner = el('div', 'banner banner-error runs-orphan');
      banner.setAttribute('data-orphan-banner', '');
      banner.appendChild(el('strong', null, 'Orphaned run pending'));
      banner.appendChild(el('span', null,
        ' \u2014 ' + state.orphaned.join(', ') +
        ' was left over from a previous server session. New runs are blocked ' +
        'until it is acknowledged (it cannot be re-attached; this only clears ' +
        'the single-flight lock).'));
      var ack = el('button', 'btn', 'Acknowledge');
      ack.type = 'button';
      ack.setAttribute('data-orphan-ack', '');
      (function (rid) {
        ack.addEventListener('click', function () { ackOrphan(rid); });
      })(state.orphaned[0]);
      banner.appendChild(ack);
      orphanHost.appendChild(banner);
    }

    /* ---------------------------------------------------------------- */
    /* script cards grouped by tier                                      */
    /* ---------------------------------------------------------------- */
    function renderCards() {
      cardsHost.innerHTML = '';
      cards = [];
      ['safe', 'slow', 'billed'].forEach(function (tier) {
        var group = el('div', 'runs-group');
        var meta = TIER_META[tier];
        var hdr = el('div', 'runs-tier-hdr runs-tier-' + tier);
        hdr.appendChild(el('span', 'badge ' + meta.badge, meta.label));
        hdr.appendChild(el('span', 'muted small', meta.title));
        group.appendChild(hdr);
        var list = el('div', 'runs-cards');
        state.scripts.forEach(function (spec) {
          if (spec.tier === tier) list.appendChild(buildCard(spec));
        });
        group.appendChild(list);
        cardsHost.appendChild(group);
      });
      updateRunButtons();
    }

    function buildCard(spec) {
      var card = {
        spec: spec,
        confirmBox: null,
        costInput: null,
        freshBox: null,
        freshWarn: null,
        argvEl: null,
        modifiesWrap: null,
        runBtn: null,
        runHint: null,
        preflightFail: null
      };
      var meta = TIER_META[spec.tier] || TIER_META.safe;

      var section = el('section', 'panel runs-card runs-card-' + spec.tier);
      section.setAttribute('data-script', spec.script_id);
      section.setAttribute('data-tier', spec.tier);
      card.root = section;

      // header: label + tier badge (+ $ marker on billed)
      var hdr = el('div', 'runs-card-hdr');
      hdr.appendChild(el('h3', null, spec.label));
      hdr.appendChild(el('span', 'badge ' + meta.badge, meta.label));
      if (spec.tier === 'billed') hdr.appendChild(el('span', 'badge billed', '$'));
      section.appendChild(hdr);

      section.appendChild(el('p', 'muted small', spec.description));

      // effective argv (read-only, updates as options are typed)
      section.appendChild(el('div', 'runs-sub', 'effective argv'));
      card.argvEl = el('div', 'runs-argv mono');
      section.appendChild(card.argvEl);

      // preflight chips: ok chips green, failing chips red + visible reason
      var pf = spec.preflight || [];
      if (pf.length) {
        var pfWrap = el('div', 'runs-preflight');
        pf.forEach(function (r) {
          if (r.ok) {
            pfWrap.appendChild(el('span', 'badge ok', '\u2713 ' + r.label));
          } else {
            card.preflightFail = r.reason || r.label;
            var chip = el('span', 'badge fail', '\u2717 ' + r.label);
            chip.title = r.reason || r.label;
            pfWrap.appendChild(chip);
            pfWrap.appendChild(el('div', 'runs-pf-reason', (r.reason || 'preflight failed') + ' \u2014 run disabled'));
          }
        });
        section.appendChild(pfWrap);
      }

      // options (w32 requires an explicit --limit number; fresh_run for w32 only)
      var optKeys = Object.keys(spec.options || {});
      if (optKeys.length || spec.script_id === 'w32') {
        var optsWrap = el('div', 'runs-options');
        optKeys.forEach(function (key) {
          var o = spec.options[key];
          var fld = el('div', 'field runs-opt');
          var id = 'opt-' + spec.script_id + '-' + key;
          var lbl = el('label', null, '--' + key + (o.required ? ' (required)' : ''));
          lbl.setAttribute('for', id);
          fld.appendChild(lbl);

          if (o.type === 'bool') {
            var cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.id = id;
            if (o.default) cb.checked = true;
            fld.appendChild(cb);
            card['opt_' + key] = cb;
            cb.addEventListener('change', function () { renderArgv(card); updateRunButton(card); });
          } else if (o.type === 'int' || o.type === 'float') {
            var num = document.createElement('input');
            num.type = 'number';
            num.id = id;
            num.step = o.type === 'int' ? '1' : 'any';
            if (o.min != null) num.min = o.min;
            if (o.max != null) num.max = o.max;
            if (o.default != null) num.value = o.default;
            fld.appendChild(num);
            card['opt_' + key] = num;
            num.addEventListener('input', function () { renderArgv(card); updateRunButton(card); });
          } else {
            var tx = document.createElement('input');
            tx.type = 'text';
            tx.id = id;
            if (o.default != null) tx.value = o.default;
            fld.appendChild(tx);
            card['opt_' + key] = tx;
            tx.addEventListener('input', function () { renderArgv(card); updateRunButton(card); });
          }
          optsWrap.appendChild(fld);
        });

        if (spec.script_id === 'w32') {
          var fr = el('div', 'runs-fresh');
          var frcb = document.createElement('input');
          frcb.type = 'checkbox';
          frcb.id = 'fresh-w32';
          var frlbl = el('label', 'runs-fresh-lbl', 'fresh_run \u2014 move datasets/raw/generated.jsonl aside first');
          frlbl.setAttribute('for', frcb.id);
          fr.appendChild(frcb);
          fr.appendChild(frlbl);
          card.freshBox = frcb;
          card.freshWarn = el('div', 'runs-fresh-warn', '');
          fr.appendChild(card.freshWarn);
          frcb.addEventListener('change', function () {
            card.freshWarn.textContent = frcb.checked
              ? 'the old generated.jsonl will be renamed to generated.jsonl.<timestamp>.bak before the run'
              : '';
            updateRunButton(card);
          });
          optsWrap.appendChild(fr);
        }
        section.appendChild(optsWrap);
      }

      // "modifies these tracked files" warning
      card.modifiesWrap = el('div', 'runs-modifies', '');
      card.modifiesWrap.style.display = 'none';
      section.appendChild(card.modifiesWrap);

      function renderModifies(show) {
        var files = spec.modifies || [];
        if (!files.length || !show) {
          card.modifiesWrap.style.display = 'none';
          return;
        }
        card.modifiesWrap.innerHTML = '';
        card.modifiesWrap.appendChild(el('strong', null, 'modifies these tracked files: '));
        files.forEach(function (p, i) {
          card.modifiesWrap.appendChild(document.createTextNode((i ? ', ' : '') + p));
        });
        card.modifiesWrap.style.display = 'block';
      }
      if (spec.tier === 'safe') renderModifies(true); // one-click: warning IS the confirm

      // confirm UI per tier
      if (spec.tier === 'slow' || spec.tier === 'billed') {
        var conf = el('div', 'runs-confirm');
        var cid = 'conf-' + spec.script_id;
        var ccb = document.createElement('input');
        ccb.type = 'checkbox';
        ccb.id = cid;
        var clbl = el('label', 'runs-confirm-lbl',
          spec.tier === 'billed'
            ? 'I confirm I want to run this billed script'
            : 'I confirm \u2014 this script may take a while');
        clbl.setAttribute('for', cid);
        conf.appendChild(ccb);
        conf.appendChild(clbl);
        card.confirmBox = ccb;
        ccb.addEventListener('change', function () {
          renderModifies(ccb.checked);
          updateRunButton(card);
        });

        if (spec.tier === 'billed') {
          var t = el('div', 'runs-cost');
          t.appendChild(el('span', 'small faint', 'Type \u2014 '));
          var costInp = document.createElement('input');
          costInp.type = 'text';
          costInp.id = 'cost-' + spec.script_id;
          costInp.placeholder = COST_PHRASE;
          costInp.autocomplete = 'off';
          costInp.spellcheck = false;
          costInp.setAttribute('aria-label', 'Cost acknowledgement phrase');
          t.appendChild(costInp);
          card.costInput = costInp;
          costInp.addEventListener('input', function () { updateRunButton(card); });
          conf.appendChild(t);
        }
        section.appendChild(conf);
      }

      // Run button + reason hint
      var act = el('div', 'runs-actions');
      var btn = el('button', 'btn btn-primary runs-runbtn', 'Run');
      btn.type = 'button';
      btn.setAttribute('data-run-script', spec.script_id);
      (function (c) {
        btn.addEventListener('click', function () { startRun(c); });
      })(card);
      card.runBtn = btn;
      card.runHint = el('div', 'runs-hint', '');
      act.appendChild(btn);
      act.appendChild(card.runHint);
      section.appendChild(act);

      cards.push(card);
      renderArgv(card);
      updateRunButton(card);
      return section;
    }

    function renderArgv(card) {
      var spec = card.spec;
      // spec.argv is the server's DISPLAY argv — drop "<required>" placeholders
      // (and the flag preceding each) before re-appending live option values.
      var argv = [];
      for (var i = 0; i < spec.argv.length; i++) {
        if (spec.argv[i] === '<required>') {
          if (argv.length) argv.pop();
          continue;
        }
        argv.push(spec.argv[i]);
      }
      var opts = spec.options || {};
      Object.keys(opts).forEach(function (key) {
        var o = opts[key];
        var inp = card['opt_' + key];
        var flag = '--' + key; // API option specs omit `flag`; CLI flag is "--" + key
        if (o.type === 'bool') {
          if (inp ? inp.checked : !!o.default) argv.push(flag);
        } else if (inp && String(inp.value).trim() !== '') {
          argv.push(flag, String(inp.value));
        } else if (o.default != null) {
          argv.push(flag, String(o.default));
        } else if (o.required) {
          argv.push(flag, '<required>');
        }
      });
      card.argvEl.textContent = argv.join('  ');
    }

    /* ---------------------------------------------------------------- */
    /* Run button gating                                                 */
    /* ---------------------------------------------------------------- */
    function updateRunButton(card) {
      var spec = card.spec;
      var reason = null;

      if (state.orphaned.length) {
        reason = 'an orphaned run is pending acknowledgement';
      } else if (state.running) {
        reason = 'a run is in progress (' + state.running.script_id + ')';
      } else if (card.preflightFail) {
        reason = 'preflight: ' + card.preflightFail;
      } else if (spec.tier === 'slow' || spec.tier === 'billed') {
        if (!card.confirmBox || !card.confirmBox.checked) {
          reason = 'confirmation required \u2014 tick the checkbox above';
        } else if (spec.tier === 'billed' &&
                   (!card.costInput || card.costInput.value !== COST_PHRASE)) {
          reason = 'type the cost acknowledgement phrase to enable';
        }
      }
      if (!reason) {
        var opts = spec.options || {};
        for (var key in opts) {
          if (!opts[key].required) continue;
          var inp = card['opt_' + key];
          if (!inp || String(inp.value).trim() === '') {
            reason = '--' + key + ' is required';
            break;
          }
          if (opts[key].type === 'int') {
            var n = Number(inp.value);
            if (!isFinite(n) || Math.floor(n) !== n ||
                (opts[key].min != null && n < opts[key].min) ||
                (opts[key].max != null && n > opts[key].max)) {
              reason = '--' + key + ' must be an integer ' +
                (opts[key].min != null ? '\u2265 ' + opts[key].min + ' ' : '') +
                (opts[key].max != null ? '\u2264 ' + opts[key].max : '');
              break;
            }
          }
        }
      }
      card.runBtn.disabled = !!reason;
      card.runHint.textContent = reason || '';
    }

    function updateRunButtons() {
      cards.forEach(updateRunButton);
    }

    /* ---------------------------------------------------------------- */
    /* start a run                                                       */
    /* ---------------------------------------------------------------- */
    function startRun(card) {
      if (card.runBtn.disabled) return;
      var spec = card.spec;
      var body = {
        confirm: spec.tier === 'slow' || spec.tier === 'billed',
        accept_cost: spec.tier === 'billed',
        options: {},
        fresh_run: !!(card.freshBox && card.freshBox.checked)
      };
      var opts = spec.options || {};
      Object.keys(opts).forEach(function (key) {
        var inp = card['opt_' + key];
        if (!inp) return;
        var o = opts[key];
        if (o.type === 'bool') body.options[key] = inp.checked;
        else if (o.type === 'int') body.options[key] = parseInt(inp.value, 10);
        else if (o.type === 'float') body.options[key] = parseFloat(inp.value);
        else body.options[key] = inp.value;
      });

      app.spinner.show(containerEl);
      postJSON('/api/runs/' + spec.script_id, body).then(function (resp) {
        app.spinner.hide(containerEl);
        state.running = { run_id: resp.run_id, script_id: resp.script_id || spec.script_id };
        updateRunButtons();
        startLive(resp.run_id, resp.script_id || spec.script_id, spec.label);
        app.toast('Run "' + spec.label + '" started', 'ok');
      }, function () {
        // 409/422 already toasted by postJSON; resync with the server so a
        // concurrent run (another session) shows up in the live pane.
        app.spinner.hide(containerEl);
        fetchAll();
      });
    }

    /* ---------------------------------------------------------------- */
    /* live pane: status + streaming log + kill                          */
    /* ---------------------------------------------------------------- */
    function stopPolling() {
      if (live && live.timer) { clearTimeout(live.timer); live.timer = null; }
    }

    function ensurePolling() {
      if (!live || live.phase !== 'running' || live.timer) return;
      live.timer = setTimeout(pollLog, POLL_MS);
    }

    function startLive(runId, scriptId, label) {
      if (live && live.runId === runId) { ensurePolling(); return; }
      stopPolling();
      live = {
        runId: runId,
        scriptId: scriptId,
        label: label,
        offset: 0,
        phase: 'running',
        errs: 0,
        timer: null,
        badgeEl: null,
        statusEl: null,
        logEl: null,
        truncEl: null,
        killBtn: null
      };
      liveHost.innerHTML = '';
      liveHost.style.display = 'block';

      var panel = el('section', 'panel runs-live');
      panel.setAttribute('data-live-run', runId);

      var hdr = el('div', 'runs-live-hdr');
      live.badgeEl = el('span', 'badge amber', 'running');
      hdr.appendChild(live.badgeEl);
      hdr.appendChild(el('strong', null, label));
      hdr.appendChild(el('span', 'mono faint', runId));
      panel.appendChild(hdr);

      live.statusEl = el('div', 'runs-status');
      live.statusEl.appendChild(el('span', 'muted', 'starting \u2026'));
      panel.appendChild(live.statusEl);

      var logWrap = el('div', 'runs-log-wrap');
      live.logEl = el('div', 'runs-log mono');
      logWrap.appendChild(live.logEl);
      panel.appendChild(logWrap);

      live.truncEl = el('div', 'runs-trunc', '');
      live.truncEl.style.display = 'none';
      panel.appendChild(live.truncEl);

      var actions = el('div', 'runs-live-actions');
      live.killBtn = el('button', 'btn runs-kill', 'Kill');
      live.killBtn.type = 'button';
      live.killBtn.setAttribute('data-kill', '');
      live.killBtn.addEventListener('click', killRun);
      actions.appendChild(live.killBtn);
      actions.appendChild(el('span', 'small faint', KILL_COPY));
      panel.appendChild(actions);

      liveHost.appendChild(panel);
      ensurePolling();
    }

    function appendLines(lines) {
      if (!live || !live.logEl || !lines.length) return;
      var logEl = live.logEl;
      var atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 48;
      lines.forEach(function (ln) {
        logEl.appendChild(el('div', 'runs-line', ln));
      });
      if (atBottom) logEl.scrollTop = logEl.scrollHeight;
    }

    function setLiveBadge(status) {
      if (!live || !live.badgeEl) return;
      var m = statusMeta(status);
      live.badgeEl.className = 'badge ' + m.cls;
      live.badgeEl.textContent = m.text;
    }

    function pollLog() {
      if (!live) return;
      live.timer = null;
      if (!containerEl.isConnected) { stopPolling(); return; }
      app.fetchJSON('/api/runs/' + live.runId + '/log?offset=' + live.offset).then(
        function (data) {
          if (!live || live.runId !== data.run_id) return;
          live.errs = 0;
          appendLines(data.lines || []);
          if (data.offset != null) live.offset = data.offset;
          if (data.truncated) {
            live.truncEl.textContent = '\u2026 earlier lines truncated (server ring buffer)';
            live.truncEl.style.display = 'block';
          }
          if (data.status === 'running') {
            ensurePolling();
          } else {
            live.phase = data.status;
            setLiveBadge(data.status);
            completeRun(data.status);
          }
        },
        function () {
          // transient — retry briefly, then fall back to /api/runs truth
          if (!live) return;
          live.errs += 1;
          if (live.errs <= 3 && containerEl.isConnected) {
            live.timer = setTimeout(pollLog, 1500);
          } else {
            stopPolling();
            setLiveBadge('failed');
            if (live.statusEl) {
              live.statusEl.appendChild(el('span', 'small fail', 'log stream unavailable'));
            }
            fetchAll();
          }
        }
      );
    }

    function killRun() {
      if (!live || live.phase !== 'running' || !live.killBtn) return;
      live.killBtn.disabled = true;
      live.killBtn.textContent = 'killing \u2026';
      postJSON('/api/runs/' + live.runId + '/kill', {}).then(function () {
        app.toast('Kill signal sent to ' + live.runId, 'ok');
      }, function () {
        if (live && live.killBtn) {
          live.killBtn.disabled = false;
          live.killBtn.textContent = 'Kill';
        }
      });
      // polling continues — the server transitions the run to "killed".
    }

    /* ---------------------------------------------------------------- */
    /* completion: status line + toast + data-changed event              */
    /* ---------------------------------------------------------------- */
    function fetchHistoryRecord(runId, cb) {
      var attempts = 0;
      (function tryNow() {
        app.fetchJSON('/api/runs/history').then(function (h) {
          var rec = null;
          (h.runs || []).some(function (r) {
            if (r.run_id === runId) { rec = r; return true; }
            return false;
          });
          if (rec) cb(rec);
          else if (attempts++ < 6) setTimeout(tryNow, 400);
          else cb(null);
        }, function () {
          if (attempts++ < 4) setTimeout(tryNow, 400);
          else cb(null);
        });
      })();
    }

    function completeRun(status) {
      var runId = live.runId;
      var scriptId = live.scriptId;
      stopPolling();
      fetchHistoryRecord(runId, function (rec) {
        if (!rec) rec = { run_id: runId, script_id: scriptId, status: status };
        showCompletion(rec);
        toastCompletion(rec);
        window.dispatchEvent(new CustomEvent('app:data-changed', {
          detail: {
            source: 'runs',
            runId: runId,
            scriptId: rec.script_id || scriptId,
            status: rec.status || status
          }
        }));
        fetchAll();
      });
    }

    function showCompletion(rec) {
      if (!live) return;
      live.phase = 'done';
      setLiveBadge(rec.status || 'completed');

      var s = live.statusEl;
      s.innerHTML = '';
      s.appendChild(el('span', 'badge ' + statusMeta(rec.status).cls, rec.status || 'completed'));
      if (rec.exit_code != null) s.appendChild(el('span', 'mono', 'exit ' + rec.exit_code));
      if (rec.gates_passed != null) {
        s.appendChild(el('span', 'badge ' + (rec.gates_passed ? 'ok' : 'fail'),
          rec.gates_passed ? 'gates passed' : 'gates NOT passed'));
        if (rec.gate_detail) s.appendChild(el('span', 'small faint', rec.gate_detail));
      }
      if (rec.duration_s != null) s.appendChild(el('span', 'mono', fmtDuration(rec.duration_s)));
      var head = shaShort(rec.git_head);
      if (head) {
        s.appendChild(el('span', 'mono faint', 'git ' + head));
        if (rec.git_dirty) s.appendChild(el('span', 'badge amber', 'dirty worktree'));
      }
      if (rec.killed) s.appendChild(el('span', 'badge fail', 'killed'));
      if (rec.timed_out) s.appendChild(el('span', 'badge fail', 'timed out'));
      // w34 rewrites the tiers but not DATASET_HASH.txt — w35 recomputes it.
      if (rec.script_id === 'w34-skip-l4' || rec.script_id === 'w34-full') {
        s.appendChild(el('span', 'badge amber', 'DATASET_HASH pending w35'));
      }
      if (live.killBtn) {
        live.killBtn.disabled = true;
        live.killBtn.textContent = 'stopped';
      }
    }

    function toastCompletion(rec) {
      var label = rec.label || rec.script_id || (live ? live.label : 'run');
      if (rec.status === 'completed' && rec.gates_passed) {
        app.toast('Run "' + label + '" completed \u2014 exit ' + rec.exit_code + ', gates passed', 'ok');
      } else if (rec.status === 'completed') {
        app.toast('Run "' + label + '" completed but gates NOT passed', 'warn');
      } else if (rec.status === 'killed') {
        app.toast('Run "' + label + '" killed', 'warn');
      } else if (rec.status === 'timed_out') {
        app.toast('Run "' + label + '" timed out', 'error');
      } else if (rec.status === 'failed') {
        app.toast('Run "' + label + '" failed (exit ' +
          (rec.exit_code == null ? '?' : rec.exit_code) + ')', 'error');
      }
    }

    /* ---------------------------------------------------------------- */
    /* history table                                                     */
    /* ---------------------------------------------------------------- */
    function renderHistory() {
      historyHost.innerHTML = '';
      var panel = el('section', 'panel');
      panel.appendChild(el('h2', 'panel-title', 'Run history'));
      var runs = state.history || [];
      if (!runs.length) {
        panel.appendChild(el('p', 'muted small',
          'No runs yet \u2014 the server records every run here (file-backed, survives restarts).'));
        historyHost.appendChild(panel);
        return;
      }
      var wrap = el('div', 'table-wrap');
      var table = el('table', 'data');
      var thead = document.createElement('thead');
      var headRow = document.createElement('tr');
      ['Run', 'Script', 'Tier', 'Status', 'Exit', 'Gates', 'Duration', 'Started', 'Killed']
        .forEach(function (h) { headRow.appendChild(el('th', null, h)); });
      thead.appendChild(headRow);
      table.appendChild(thead);

      var tbody = document.createElement('tbody');
      runs.forEach(function (r) {
        var tr = document.createElement('tr');
        tr.setAttribute('data-history-run', r.run_id || '');
        tr.appendChild(el('td', 'mono', runShort(r.run_id)));
        var scriptCell = el('td');
        scriptCell.appendChild(el('span', null, r.label || r.script_id || '?'));
        if (r.script_id && r.script_id !== (r.label || '')) {
          scriptCell.appendChild(el('div', 'faint small', r.script_id));
        }
        tr.appendChild(scriptCell);
        var tierCell = el('td');
        tierCell.appendChild(el('span',
          'badge ' + (TIER_META[r.tier] ? TIER_META[r.tier].badge : 'badge'), r.tier || '?'));
        tr.appendChild(tierCell);
        var st = statusMeta(r.status);
        var statusCell = el('td');
        statusCell.appendChild(el('span', 'badge ' + st.cls, st.text));
        tr.appendChild(statusCell);
        tr.appendChild(el('td', 'num', r.exit_code == null ? '\u2014' : String(r.exit_code)));
        if (r.gates_passed == null) {
          tr.appendChild(el('td', null, '\u2014'));
        } else {
          var gatesCell = el('td');
          gatesCell.appendChild(el('span', 'badge ' + (r.gates_passed ? 'ok' : 'fail'),
            r.gates_passed ? 'pass' : 'fail'));
          tr.appendChild(gatesCell);
        }
        tr.appendChild(el('td', 'num', fmtDuration(r.duration_s)));
        tr.appendChild(el('td', null, fmtTime(r.started_at)));
        var killedCell = el('td');
        killedCell.appendChild(r.killed ? el('span', 'badge fail', 'yes') : el('span', null, '\u2014'));
        tr.appendChild(killedCell);
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      panel.appendChild(wrap);
      historyHost.appendChild(panel);
    }

    /* ---------------------------------------------------------------- */
    /* skeleton + boot                                                   */
    /* ---------------------------------------------------------------- */
    var hdr = el('div', 'runs-page-hdr');
    hdr.appendChild(el('h1', null, 'Pipeline runs'));
    var refresh = el('button', 'btn', 'Refresh');
    refresh.type = 'button';
    refresh.addEventListener('click', fetchAll);
    hdr.appendChild(refresh);
    containerEl.appendChild(hdr);
    containerEl.appendChild(el('p', 'muted small',
      'Whitelisted runner \u2014 single-flight. Safe scripts are offline; slow ' +
      'scripts need local compute; billed scripts call the paid LLM API and ' +
      'require confirmation plus a typed cost acknowledgement.'));

    orphanHost = el('div');
    containerEl.appendChild(orphanHost);
    liveHost = el('div');
    liveHost.style.display = 'none';
    containerEl.appendChild(liveHost);
    cardsHost = el('div');
    containerEl.appendChild(cardsHost);
    historyHost = el('div');
    containerEl.appendChild(historyHost);

    fetchAll();
  };
})();
