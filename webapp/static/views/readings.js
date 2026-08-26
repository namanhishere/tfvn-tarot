/* tfvn-tarot dataset viewer — Readings view (multi-turn tarot chat)
 *
 * Talks to the /api/readings router:
 *   GET  /api/readings/backend          health probe (banner + dot)
 *   POST /api/readings/session          create session {n_cards, seed}
 *   DEL  /api/readings/{sid}            drop session ("New reading")
 *   POST /api/readings/{sid}/turn       SSE stream of one conversation turn
 *
 * SSE frame types (tfvn.reading_stream.stream_turn):
 *   step   — pipeline stage trace (crisis_gate, draw, context)
 *   stop   — turn ended early (crisis | clarification | empty_question |
 *            backend_error); message_vi carries the user-facing text
 *   tokens — streamed content delta
 *   validate — validator verdict for one generation attempt
 *   regen  — constrained regeneration starting (text replaced live)
 *   done   — committed turn (validation_warning flag inside)
 *
 * Reconnect discipline: an interrupted stream never commits server-side,
 * so the failed turn is retried verbatim via a Retry affordance on the
 * interrupted bubble. One turn in flight at a time.
 *
 * Transcript survives page reloads: messages are mirrored into
 * sessionStorage under the session id; the server keeps the authoritative
 * draw/history while this tab is open.
 */
(function () {
  'use strict';

  window.Views = window.Views || {};
  window.Views.readings = function (containerEl, app) {
    containerEl.setAttribute('data-view', 'readings');
    containerEl.innerHTML = '';

    var style = document.createElement('style');
    style.textContent = [
      '.readings-grid { display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 16px; }',
      '@media (max-width: 900px) { .readings-grid { grid-template-columns: 1fr; } .readings-aside { order: 2; } }',
      '.readings-main { min-width: 0; display: flex; flex-direction: column; gap: 12px; }',
      '.banner-down { border: 1px solid rgba(224,101,92,.55); background: rgba(224,101,92,.12); color: var(--fail); border-radius: var(--radius); padding: 10px 14px; font-weight: 600; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }',
      '.banner-down code { font-family: var(--mono); color: var(--text); }',
      '.session-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }',
      '.session-bar select, .session-bar input { background: var(--bg-inset); border: 1px solid var(--border-strong); border-radius: var(--radius-sm); color: var(--text); padding: 6px 8px; font-size: 13px; }',
      '.session-bar input { width: 110px; }',
      '.session-label { color: var(--text-dim); font-size: 13px; }',
      '.transcript { border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-panel); padding: 14px; height: 46vh; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }',
      '.rdg-empty { color: var(--text-faint); margin: auto; text-align: center; max-width: 420px; }',
      '.msg { max-width: 82%; border-radius: var(--radius); padding: 9px 12px; white-space: pre-wrap; overflow-wrap: anywhere; }',
      '.msg-user { align-self: flex-end; background: var(--bg-raised); border: 1px solid var(--border-strong); }',
      '.msg-assistant { align-self: flex-start; background: var(--bg-inset); border: 1px solid var(--border); }',
      '.msg-crisis { align-self: stretch; max-width: 100%; border-color: rgba(224,101,92,.55); background: rgba(224,101,92,.12); color: var(--text); }',
      '.msg-clarify { align-self: flex-start; border-color: rgba(217,166,79,.55); background: rgba(217,166,79,.10); }',
      '.msg-meta { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }',
      '.chip { display: inline-block; border: 1px solid var(--border-strong); border-radius: 999px; padding: 1px 9px; font-size: 12px; background: var(--bg-raised); }',
      '.badge-up { color: var(--ok); font-weight: 600; }',
      '.badge-rev { color: var(--amber); font-weight: 600; }',
      '.warn-flag { color: var(--fail); font-weight: 600; font-size: 12px; border: 1px solid rgba(224,101,92,.5); border-radius: var(--radius-sm); padding: 0 6px; }',
      '.interrupted-note { color: var(--amber); font-size: 12px; margin-top: 6px; }',
      '.composer { display: flex; gap: 8px; }',
      '.composer textarea { flex: 1; resize: vertical; min-height: 44px; max-height: 160px; background: var(--bg-inset); border: 1px solid var(--border-strong); border-radius: var(--radius-sm); color: var(--text); padding: 9px 11px; font-family: var(--sans); font-size: 14px; }',
      '.composer textarea:focus { outline: 1px solid var(--accent); }',
      '.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--text-faint); margin-right: 6px; }',
      '.dot-ok { background: var(--ok); } .dot-fail { background: var(--fail); }',
      '.trace-list { list-style: none; margin: 0; padding: 0; font-family: var(--mono); font-size: 12px; color: var(--text-dim); max-height: 52vh; overflow-y: auto; }',
      '.trace-list li { padding: 3px 0; border-bottom: 1px dashed var(--border); overflow-wrap: anywhere; }',
      '.trace-stage { color: var(--accent); }',
      '.trace-warn { color: var(--fail); }',
      '.trace-time { color: var(--text-faint); margin-right: 6px; }'
    ].join('\n');
    containerEl.appendChild(style);

    /* ---------------- state ---------------- */
    var state = {
      sessionId: null,
      nCards: 3,
      busy: false,          // one SSE turn in flight
      backendOk: null,      // null unknown, true/false probed
      pollTimer: null,
      msgs: []              // [{role:'user'|'assistant'|'crisis'|'clarify', text, chips?, warn?, interrupted?}]
    };

    /* ---------------- layout ---------------- */
    var wrap = document.createElement('div');
    wrap.className = 'readings-grid';
    containerEl.appendChild(wrap);

    var main = document.createElement('div');
    main.className = 'readings-main';
    wrap.appendChild(main);

    var aside = document.createElement('div');
    aside.className = 'panel readings-aside';
    var asideTitle = document.createElement('div');
    asideTitle.className = 'panel-title';
    asideTitle.textContent = 'Pipeline trace';
    var dotLine = document.createElement('div');
    dotLine.style.cssText = 'font-size:13px;margin-bottom:8px;';
    var dotEl = document.createElement('span');
    dotEl.className = 'dot';
    var dotText = document.createElement('span');
    dotText.textContent = 'backend: checking\u2026';
    dotLine.appendChild(dotEl);
    dotLine.appendChild(dotText);
    var traceList = document.createElement('ul');
    traceList.className = 'trace-list';
    aside.appendChild(asideTitle);
    aside.appendChild(dotLine);
    aside.appendChild(traceList);
    wrap.appendChild(aside);

    // banner slot (top of main column)
    var bannerSlot = document.createElement('div');
    main.appendChild(bannerSlot);

    // header + session controls
    var h1 = document.createElement('h1');
    h1.textContent = 'Readings';
    main.appendChild(h1);

    var bar = document.createElement('div');
    bar.className = 'session-bar';
    var lblSpread = document.createElement('span');
    lblSpread.className = 'session-label';
    lblSpread.textContent = 'Spread:';
    var selCards = document.createElement('select');
    [['1', 1], ['3', 3], ['10', 10]].forEach(function (pair) {
      var opt = document.createElement('option');
      opt.value = String(pair[1]);
      opt.textContent = pair[0] + ' l\u00e1';
      if (pair[1] === 3) opt.selected = true;
      selCards.appendChild(opt);
    });
    var lblSeed = document.createElement('span');
    lblSeed.className = 'session-label';
    lblSeed.textContent = 'Seed (tu\u1ef3 ch\u1ecdn):';
    var inpSeed = document.createElement('input');
    inpSeed.type = 'number';
    inpSeed.placeholder = 'random';
    var btnNew = document.createElement('button');
    btnNew.className = 'btn';
    btnNew.textContent = 'B\u00e0i m\u1edbi';
    btnNew.title = 'Drop this conversation and draw fresh cards';
    bar.appendChild(lblSpread);
    bar.appendChild(selCards);
    bar.appendChild(lblSeed);
    bar.appendChild(inpSeed);
    bar.appendChild(btnNew);
    main.appendChild(bar);

    // transcript
    var transcript = document.createElement('div');
    transcript.className = 'transcript';
    main.appendChild(transcript);
    var emptyHint = document.createElement('div');
    emptyHint.className = 'rdg-empty';
    emptyHint.textContent =
      'Nh\u1eadp c\u00e2u h\u1ecfi b\u1eb1ng ti\u1ebfng Vi\u1ec7t \u2014 c\u00e0i \u0111\u1eb7t s\u1ebd r\u00fat b\u00e0i, ' +
      'd\u1ecbch danh t\u1eeb l\u00e1 ti\u1ebfng Anh gi\u1eef nguy\u00ean, ki\u1ec3m tra b\u1ea5t bi\u1ebfn tr\u01b0\u1edbc khi hi\u1ec3n th\u1ecb.';
    transcript.appendChild(emptyHint);

    // composer
    var composer = document.createElement('div');
    composer.className = 'composer';
    var input = document.createElement('textarea');
    input.rows = 2;
    input.placeholder = 'C\u00e2u h\u1ecfi c\u1ee7a b\u1ea1n\u2026 (Enter g\u1eedi, Shift+Enter xu\u1ed1ng d\u00f2ng)';
    var btnSend = document.createElement('button');
    btnSend.className = 'btn btn-primary';
    btnSend.textContent = 'G\u1eedi';
    composer.appendChild(input);
    composer.appendChild(btnSend);
    main.appendChild(composer);

    /* ---------------- helpers ---------------- */
    function el(tag, className, text) {
      var e = document.createElement(tag);
      if (className) e.className = className;
      if (text != null) e.textContent = text;
      return e;
    }

    function storageKey() { return 'readings.' + state.sessionId; }
    var CURRENT_KEY = 'readings.current';

    function setCurrent(sid) {
      state.sessionId = sid;
      try {
        if (sid) sessionStorage.setItem(CURRENT_KEY, sid);
        else sessionStorage.removeItem(CURRENT_KEY);
      } catch (_) { /* best-effort */ }
    }

    function persist() {
      try {
        if (state.sessionId) {
          sessionStorage.setItem(storageKey(), JSON.stringify(state.msgs));
        }
      } catch (_) { /* private mode etc. — transcript is best-effort */ }
    }

    function restore(sessionId) {
      try {
        var raw = sessionStorage.getItem('readings.' + sessionId);
        return raw ? JSON.parse(raw) : [];
      } catch (_) { return []; }
    }

    function recoverSession() {
      /* Reload support: reattach to the live server-side session if it
       * still exists; otherwise start clean (server restart case). */
      var prev = null;
      try { prev = sessionStorage.getItem(CURRENT_KEY); } catch (_) {}
      if (!prev) return Promise.resolve();
      return fetch('/api/readings/' + prev).then(function (resp) {
        if (!resp.ok) { setCurrent(null); return null; }
        return resp.json().then(function (st) {
          setCurrent(prev);
          state.msgs = restore(prev);
          state.nCards = st.n_cards;
          selCards.value = String(st.n_cards);
          rerender();
          trace('recovered', 'session ' + prev + ' \u00b7 ' +
            st.turns + ' turn(s)');
          return prev;
        });
      }, function () { setCurrent(null); });
    }

    function renderMessage(m, opts) {
      opts = opts || {};
      var div;
      if (m.role === 'user') {
        div = el('div', 'msg msg-user', m.text);
      } else if (m.role === 'crisis') {
        div = el('div', 'msg msg-crisis', m.text);
      } else if (m.role === 'clarify') {
        div = el('div', 'msg msg-clarify', m.text);
      } else {
        div = el('div', 'msg msg-assistant');
        if (m.chips && m.chips.length) {
          var meta = el('div', 'msg-meta');
          m.chips.forEach(function (c) {
            var chip = el('span', 'chip');
            var badge = el('span', c.orientation === 'reversed'
              ? 'badge-rev' : 'badge-up',
              c.orientation === 'reversed' ? 'NG\u01af\u1ee2' : 'XU\u00d4I');
            chip.appendChild(document.createTextNode(c.name_en + ' '));
            chip.appendChild(badge);
            meta.appendChild(chip);
          });
          div.appendChild(meta);
        }
        var body = el('div', null, m.text);
        div.appendChild(body);
        if (m.warn) div.appendChild(el('span', 'warn-flag',
          'validation_warning'));
        if (opts.live) m._bodyEl = body;
      }
      if (m.interrupted) {
        var note = el('div', 'interrupted-note');
        note.textContent =
          'M\u1ea5t k\u1ebft n\u1ed1i gi\u1eefa chuy\u1ec3n \u2014 ch\u01b0a l\u01b0u. ';
        var retry = el('button', 'btn-link', 'Th\u1eed l\u1ea1i');
        retry.addEventListener('click', function () {
          m.interrupted = false;
          persist();
          sendTurn(m.text);
        });
        note.appendChild(retry);
        div.appendChild(note);
      }
      transcript.appendChild(div);
      transcript.scrollTop = transcript.scrollHeight;
      return div;
    }

    function rerender() {
      transcript.innerHTML = '';
      if (!state.msgs.length) transcript.appendChild(emptyHint);
      state.msgs.forEach(function (m) { renderMessage(m); });
    }

    function trace(stage, detail, warn) {
      var li = el('li');
      li.appendChild(el('span', 'trace-time',
        new Date().toLocaleTimeString()));
      li.appendChild(el('span', warn ? 'trace-stage trace-warn'
        : 'trace-stage', stage));
      if (detail) li.appendChild(document.createTextNode(' ' + detail));
      traceList.appendChild(li);
      traceList.scrollTop = traceList.scrollHeight;
    }

    function setBackend(ok, err) {
      state.backendOk = ok;
      dotEl.className = 'dot ' + (ok ? 'dot-ok' : 'dot-fail');
      dotText.textContent = ok ? 'llama-server: OK'
        : ('llama-server: DOWN' + (err ? ' (' + err + ')' : ''));
      renderBanner();
      updateSendEnabled();
    }

    function renderBanner() {
      bannerSlot.innerHTML = '';
      if (state.backendOk !== false) return;
      var b = el('div', 'banner-down');
      b.appendChild(el('span', null,
        'llama-server kh\u00f4ng ch\u1ea1y \u2014 kh\u00f4ng sinh \u0111\u01b0\u1ee3c b\u00e0i \u0111\u1ecdc.'));
      var hint = el('code', null,
        'TAROT_MODEL=<gguf> scripts/serve.sh');
      b.appendChild(hint);
      var retryBtn = el('button', 'btn-link', 'Ki\u1ec3m tra l\u1ea1i');
      retryBtn.addEventListener('click', function () { probe(); });
      b.appendChild(retryBtn);
      bannerSlot.appendChild(b);
    }

    function updateSendEnabled() {
      btnSend.disabled = state.busy || state.backendOk === false;
    }

    function probe() {
      app.fetchJSON('/api/readings/backend').then(function (r) {
        setBackend(!!r.ok, r.error);
      }).catch(function () { setBackend(false, 'probe failed'); });
    }

    /* ---------------- session management ---------------- */
    function createSession() {
      // raw fetch: app.fetchJSON is GET-only by shell contract
      var payload = { n_cards: parseInt(selCards.value, 10) };
      if (inpSeed.value !== '') payload.seed = parseInt(inpSeed.value, 10);
      return fetch('/api/readings/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(function (resp) {
        if (!resp.ok) {
          return resp.json().catch(function () { return {}; })
            .then(function (b) {
              throw new Error('HTTP ' + resp.status +
                (b && b.detail ? ': ' + b.detail : ''));
            });
        }
        return resp.json();
      });
    }

    btnNew.addEventListener('click', function () {
      var finish = function () {
        setCurrent(null);
        state.msgs = [];
        rerender();
        traceList.innerHTML = '';
        beginSession();
      };
      if (state.sessionId) {
        fetch('/api/readings/' + state.sessionId, { method: 'DELETE' })
          .catch(function () {})
          .then(finish, finish);
      } else {
        finish();
      }
    });

    function beginSession() {
      if (state.sessionId) return Promise.resolve(state.sessionId);
      return createSession().then(function (r) {
        setCurrent(r.session_id);
        state.nCards = r.n_cards;
        state.msgs = restore(r.session_id);
        rerender();
        return r.session_id;
      }, function (err) {
        app.toast('Kh\u00f4ng t\u1ea1o \u0111\u01b0\u1ee3c phi\u00ean: ' +
          (err && err.message ? err.message : err));
        throw err;
      });
    }

    /* ---------------- SSE turn ---------------- */
    function parseFrames(buffer) {
      // returns {events: [...], rest: incomplete-frame-tail}
      var events = [];
      var parts = buffer.split('\n\n');
      var rest = parts.pop();
      parts.forEach(function (frame) {
        var lines = frame.split('\n');
        for (var i = 0; i < lines.length; i++) {
          if (lines[i].indexOf('data: ') === 0) {
            try { events.push(JSON.parse(lines[i].slice(6))); }
            catch (_) { /* malformed frame — skip */ }
          }
        }
      });
      return { events: events, rest: rest };
    }

    function sendTurn(content) {
      content = (content || '').trim();
      if (!content || state.busy) return;

      var am = null;            // hoisted: the catch below may run early
      var finished = function (ok) {
        state.busy = false;
        updateSendEnabled();
        if (!ok && am) {
          am.interrupted = true;   // server never committed -> retry is safe
          rerender();
        }
        persist();
      };

      state.busy = true;
      updateSendEnabled();

      // Session FIRST: beginSession may replace state.msgs (restore) and
      // rerender — bubbles must be created after that, never before.
      beginSession().then(function (sid) {
        var um = { role: 'user', text: content };
        state.msgs.push(um);
        renderMessage(um);
        persist();

        var am = { role: 'assistant', text: '', chips: [] };
        state.msgs.push(am);
        var amEl = renderMessage(am, { live: true });

        var controller = new AbortController();
        am._abort = controller;
        return fetch('/api/readings/' + sid + '/turn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: content }),
          signal: controller.signal
        }).then(function (resp) {
          if (!resp.ok || !resp.body) {
            return resp.json().catch(function () { return {}; })
              .then(function (b) {
                throw new Error('HTTP ' + resp.status +
                  (b && b.detail ? ': ' + b.detail : ''));
              });
          }
          var reader = resp.body.getReader();
          var decoder = new TextDecoder();
          var buffer = '';
          var sawDone = false;

          function pump() {
            return reader.read().then(function (chunk) {
              if (chunk.done) {
                finished(sawDone);
                return;
              }
              buffer += decoder.decode(chunk.value, { stream: true });
              var parsed = parseFrames(buffer);
              buffer = parsed.rest;
              parsed.events.forEach(function (ev) {
                handleEvent(ev, am, amEl);
                if (ev.type === 'done' || ev.type === 'stop') {
                  sawDone = true;   // terminal event -> turn completed
                }
              });
              if (am._bodyEl) {
                am._bodyEl.textContent = am.text;
                transcript.scrollTop = transcript.scrollHeight;
              }
              return pump();
            });
          }
          return pump();
        });
      }).catch(function (err) {
        if (err && err.name === 'AbortError') return;
        app.toast((err && err.message) ? err.message : String(err));
        if (am) am.interrupted = true;
        finished(false);
      });

      function handleEvent(ev, target, bubbleEl) {
        switch (ev.type) {
          case 'step':
            if (ev.stage === 'draw') {
              ev.cards.forEach(function (c) {
                target.chips.push({ name_en: c.name_en,
                  orientation: c.orientation });
              });
              if (target._bodyEl && bubbleEl) {
                rerenderChips(bubbleEl, target);
              }
              trace('draw', ev.cards.map(function (c) {
                return c.name_en + '/' +
                  (c.orientation === 'reversed' ? '\u0111\u1ea3o' : 'xu\u00f4i');
              }).join(', ') + ' (seed ' + ev.seed + ')');
            } else if (ev.stage === 'crisis_gate') {
              trace('crisis_gate', ev.routed ? 'ROUTED' : 'pass');
            } else if (ev.stage === 'context') {
              trace('context', 'history=' + ev.history_messages +
                ' dropped=' + ev.dropped);
            } else if (ev.stage === 'clarification') {
              trace('clarification', '');
            }
            break;
          case 'tokens':
            target.text += ev.text;
            break;
          case 'validate':
            trace('validate#' + ev.attempt,
              ev.ok ? 'ok' : 'FAIL ' + JSON.stringify(ev.failures),
              !ev.ok);
            if (ev.tok_s_approx != null && ev.attempt === 0) {
              trace('timing', ev.elapsed_ms + ' ms \u00b7 ~' +
                ev.tok_s_approx + ' tok/s');
            }
            break;
          case 'regen':
            target.text = '';
            trace('regen', 'constrained retry', true);
            break;
          case 'stop':
            if (ev.reason === 'crisis') {
              // replace the empty assistant bubble with the crisis card
              state.msgs.pop();
              if (am._abort) am._abort.abort();
              var cm = { role: 'crisis', text: ev.message_vi };
              state.msgs.push(cm);
              rerender();
              trace('stop', 'crisis routing', true);
              persist();
              state.busy = false;
              updateSendEnabled();
            } else if (ev.reason === 'clarification') {
              state.msgs.pop();
              var qm = { role: 'clarify', text: ev.message_vi };
              state.msgs.push(qm);
              rerender();
              trace('stop', 'clarification asked');
              persist();
              state.busy = false;
              updateSendEnabled();
            } else {
              target.text += (target.text ? '\n' : '') + ev.message_vi;
              if (target._bodyEl) target._bodyEl.textContent = target.text;
              trace('stop', ev.reason, ev.reason === 'backend_error');
              persist();
              state.busy = false;
              updateSendEnabled();
            }
            break;
          case 'done':
            target.warn = !!ev.validation_warning;
            if (target._bodyEl) target._bodyEl.textContent = target.text;
            if (target.warn && bubbleEl) {
              bubbleEl.appendChild(el('span', 'warn-flag',
                'validation_warning'));
            }
            trace('done', 'turn ' + ev.turn + ' \u00b7 ' +
              ev.elapsed_ms + ' ms' +
              (target.warn ? ' \u00b7 WARNING' : ''), target.warn);
            persist();
            break;
          default:
            break;
        }
      }

      function rerenderChips(bubbleEl, m) {
        var existing = bubbleEl.querySelector('.msg-meta');
        if (existing) existing.remove();
        if (!m.chips.length) return;
        var meta = el('div', 'msg-meta');
        m.chips.forEach(function (c) {
          var chip = el('span', 'chip');
          var badge = el('span', c.orientation === 'reversed'
            ? 'badge-rev' : 'badge-up',
            c.orientation === 'reversed' ? 'NG\u01af\u1ee2' : 'XU\u00d4I');
          chip.appendChild(document.createTextNode(c.name_en + ' '));
          chip.appendChild(badge);
          meta.appendChild(chip);
        });
        bubbleEl.insertBefore(meta, bubbleEl.firstChild);
      }
    }

    function submit() {
      var text = input.value.trim();
      if (!text) return;
      if (state.backendOk === false) {
        app.toast('llama-server \u0111ang t\u1eaft \u2014 kh\u1edfi \u0111\u1ed9ng scripts/serve.sh tr\u01b0\u1edbc.');
        return;
      }
      input.value = '';
      sendTurn(text);
    }

    btnSend.addEventListener('click', submit);
    input.addEventListener('keydown', function (ke) {
      if (ke.key === 'Enter' && !ke.shiftKey) {
        ke.preventDefault();
        submit();
      }
    });

    /* ---------------- boot ---------------- */
    probe();
    state.pollTimer = setInterval(probe, 15000);
    recoverSession();
    containerEl.addEventListener('removed', function () {
      clearInterval(state.pollTimer);
    });
    // hash-router view swaps replace #view content without events; also
    // stop polling when our nodes detach (cheap guard).
    var pollGuard = setInterval(function () {
      if (!document.body.contains(containerEl)) {
        clearInterval(state.pollTimer);
        clearInterval(pollGuard);
      }
    }, 5000);
  };
})();
