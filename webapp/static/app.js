/* tfvn-tarot dataset viewer — SPA shell (B.1)
 *
 * Zero-toolchain vanilla JS: hash router + fetch wrapper + view registry.
 *
 * VIEW CONTRACT (B.2–B.8 must follow this exactly):
 *   - Each view is a plain file webapp/static/views/<name>.js that defines
 *       window.Views.<name> = function (containerEl, app) { ... }
 *   - containerEl is the #view element; the view owns its content there.
 *   - app is the shell object exposing:
 *       app.fetchJSON(path)            -> Promise<object>; GET JSON. On
 *                                        non-2xx it surfaces the FastAPI
 *                                        `detail` (422 messages included),
 *                                        shows an error toast and THROWS.
 *       app.toast(message, kind?)      kind: 'error' (default) | 'ok' | 'warn'
 *       app.spinner.show(containerEl?) / app.spinner.hide(containerEl?)
 *                                        overlay loader inside the container
 *       app.esc(value)                 HTML-escape interpolated data
 *   - The router injects <script src="views/<name>.js"> lazily (once), then
 *     calls window.Views[name](containerEl, app). Missing view file renders a
 *     "not implemented" placeholder so the shell works standalone.
 *   - window.app is exposed for debugging.
 */
(function () {
  'use strict';

  var ROUTES = ['dashboard', 'readings', 'cards', 'dataset', 'raw', 'stats', 'reports', 'runs'];
  var viewEl = document.getElementById('view');
  var loadedViews = {};

  /* ---------------------------------------------------------------- */
  /* esc() — HTML-escape any interpolated data (XSS hygiene).          */
  /* ---------------------------------------------------------------- */
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /* ---------------------------------------------------------------- */
  /* toast() — transient status message, top-right.                    */
  /* ---------------------------------------------------------------- */
  function toast(message, kind) {
    var root = document.getElementById('toast-root');
    if (!root) return;
    var el = document.createElement('div');
    el.className = 'toast ' + (kind === 'ok' || kind === 'warn' ? kind : 'error');
    el.setAttribute('role', 'status');
    el.textContent = message;

    var close = document.createElement('button');
    close.className = 'toast-close';
    close.setAttribute('aria-label', 'Dismiss');
    close.textContent = '\u00d7';
    close.addEventListener('click', function () { el.remove(); });
    el.appendChild(close);

    root.appendChild(el);
    setTimeout(function () { el.classList.add('toast-leave'); }, 4500);
    setTimeout(function () { el.remove(); }, 4900);
  }

  /* ---------------------------------------------------------------- */
  /* spinner() — overlay loader inside a container (defaults #view).   */
  /* ---------------------------------------------------------------- */
  function spinnerTarget(container) { return container || viewEl; }

  function showSpinner(container) {
    var target = spinnerTarget(container);
    if (!target || target.querySelector('.spinner-overlay')) return;
    var overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    var ring = document.createElement('div');
    ring.className = 'spinner';
    ring.setAttribute('role', 'status');
    ring.setAttribute('aria-label', 'Loading');
    overlay.appendChild(ring);
    target.appendChild(overlay);
  }

  function hideSpinner(container) {
    var target = spinnerTarget(container);
    if (!target) return;
    var overlay = target.querySelector('.spinner-overlay');
    if (overlay) overlay.remove();
  }

  /* ---------------------------------------------------------------- */
  /* fetchJSON() — GET JSON; toast + throw on any failure.             */
  /* ---------------------------------------------------------------- */
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

  function fetchJSON(path) {
    var req;
    try {
      req = fetch(path, { headers: { Accept: 'application/json' } });
    } catch (err) {
      var netMsg = 'Network error fetching ' + path + ': ' + (err && err.message ? err.message : String(err));
      toast(netMsg);
      var netErr = new Error(netMsg);
      netErr.status = 0;
      return Promise.reject(netErr);
    }
    return req.then(function (res) {
      if (!res.ok) {
        return res.json().then(function (body) {
          var detail = 'HTTP ' + res.status + ' ' + res.statusText;
          if (body && typeof body.detail !== 'undefined') detail = formatDetail(body.detail);
          else if (body && typeof body.message !== 'undefined') detail = String(body.message);
          var msg = path + ' \u2192 ' + detail;
          toast(msg);
          var err = new Error(msg);
          err.status = res.status;
          throw err;
        }, function () {
          // non-JSON error body
          var msg2 = path + ' \u2192 HTTP ' + res.status + ' ' + res.statusText;
          toast(msg2);
          var err2 = new Error(msg2);
          err2.status = res.status;
          throw err2;
        });
      }
      return res.json().catch(function (err) {
        var msg3 = 'Invalid JSON from ' + path + ': ' + (err && err.message ? err.message : String(err));
        toast(msg3);
        throw new Error(msg3);
      });
    });
  }

  /* ---------------------------------------------------------------- */
  /* router                                                            */
  /* ---------------------------------------------------------------- */
  function currentRouteName() {
    var raw = location.hash.replace(/^#\/?/, '');
    var seg = raw.split('?')[0].split('/')[0];
    return ROUTES.indexOf(seg) !== -1 ? seg : 'dashboard';
  }

  /* Unknown hash -> rewrite to #/dashboard (no history entry, no loop). */
  function normalizeHash() {
    var raw = location.hash.replace(/^#\/?/, '');
    if (raw === '') return; // empty hash === dashboard, leave URL alone
    var seg = raw.split('?')[0].split('/')[0];
    if (ROUTES.indexOf(seg) === -1) {
      history.replaceState(null, '', '#/dashboard');
    }
  }

  function updateNav(name) {
    var links = document.querySelectorAll('nav a[data-route]');
    for (var i = 0; i < links.length; i++) {
      if (links[i].getAttribute('data-route') === name) {
        links[i].setAttribute('aria-current', 'page');
      } else {
        links[i].removeAttribute('aria-current');
      }
    }
    document.title = 'tfvn-tarot \u2014 ' + name;
  }

  function errorPanel(message) {
    var div = document.createElement('div');
    div.className = 'view-error';
    var h = document.createElement('h2');
    h.textContent = 'Something went wrong';
    var p = document.createElement('p');
    p.textContent = message;
    div.appendChild(h);
    div.appendChild(p);
    return div;
  }

  function renderNotImplemented(name) {
    viewEl.setAttribute('data-view', name);
    viewEl.innerHTML = '';
    var wrap = document.createElement('div');
    wrap.className = 'placeholder';
    var h = document.createElement('h2');
    h.textContent = 'View "' + name + '" is not implemented yet';
    var p = document.createElement('p');
    p.innerHTML =
      'This shell (B.1) ships the router, fetch wrapper and styles; views are ' +
      'added one per task (B.2\u2013B.8) as <code>views/' + esc(name) + '.js</code>.';
    wrap.appendChild(h);
    wrap.appendChild(p);
    viewEl.appendChild(wrap);
  }

  function runView(name) {
    viewEl.setAttribute('data-view', name);
    try {
      window.Views[name](viewEl, app);
    } catch (err) {
      var msg = 'View "' + name + '" failed: ' + (err && err.message ? err.message : String(err));
      console.error(msg, err);
      toast(msg);
      viewEl.innerHTML = '';
      viewEl.appendChild(errorPanel(msg));
    }
  }

  function loadView(name) {
    if (window.Views && typeof window.Views[name] === 'function') {
      runView(name);
      return;
    }
    if (loadedViews[name]) {
      renderNotImplemented(name);
      return;
    }
    loadedViews[name] = true;
    var script = document.createElement('script');
    script.src = 'views/' + name + '.js';
    script.onload = function () {
      if (window.Views && typeof window.Views[name] === 'function') runView(name);
      else renderNotImplemented(name);
    };
    script.onerror = function () {
      renderNotImplemented(name);
    };
    document.head.appendChild(script);
  }

  function render() {
    normalizeHash();
    var name = currentRouteName();
    updateNav(name);
    window.scrollTo(0, 0);
    loadView(name);
  }

  /* ---------------------------------------------------------------- */
  /* shell object passed to every view                                 */
  /* ---------------------------------------------------------------- */
  var app = {
    fetchJSON: fetchJSON,
    toast: toast,
    spinner: { show: showSpinner, hide: hideSpinner },
    esc: esc
  };

  window.Views = window.Views || {};
  window.app = app;
  window.addEventListener('hashchange', render);
  render();
})();
