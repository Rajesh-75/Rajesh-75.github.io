/*!
 * quiz.js - shared timed-quiz animation for all tutorial pages.
 *
 * One function: startQuiz(config). Pages supply only the questions.
 *
 *   <div id="quiz"></div>
 *   <script src="quiz.js"></script>
 *   <script>
 *     startQuiz({
 *       mount: '#quiz',                       // CSS selector or DOM element
 *       questions: [
 *         { q: 'Question text (HTML ok)',
 *           a: 'The correct answer',
 *           w: ['Wrong 1', 'Wrong 2', 'Wrong 3'] }
 *       ],
 *       timeLimit: 5,                         // optional, seconds per question (default 5)
 *       feedbackMs: 1600                      // optional, pause showing the right answer (default 1600)
 *     });
 *   </script>
 *
 * Options are shuffled on every attempt. Unanswered questions count as wrong.
 * Styles are injected automatically, so no separate CSS file is needed.
 */
function startQuiz(config) {
    'use strict';

    var QUESTIONS   = config.questions || [];
    var TIME_LIMIT  = config.timeLimit  || 5;
    var FEEDBACK_MS = config.feedbackMs || 1600;
    var LETTERS     = ['A', 'B', 'C', 'D', 'E', 'F'];

    var box = typeof config.mount === 'string' ? document.querySelector(config.mount) : config.mount;
    if (!box) { console.error('startQuiz: mount element not found'); return; }
    if (!QUESTIONS.length) { console.error('startQuiz: no questions supplied'); return; }

    /* ---- styles (injected once per page, shared by every quiz on it) ---- */
    if (!document.getElementById('tq-style')) {
        var st = document.createElement('style');
        st.id = 'tq-style';
        st.textContent =
            '.tq-box{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:22px;box-shadow:0 1px 3px rgba(0,0,0,.05);max-width:750px;font-family:system-ui,-apple-system,sans-serif;color:#1e293b}' +
            '.tq-box .tq-meta{display:flex;justify-content:space-between;font-size:.9rem;color:#64748b;margin-bottom:8px}' +
            '.tq-box .tq-bar{height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin-bottom:18px}' +
            '.tq-box .tq-bar>div{height:100%;width:100%;background:#2563eb}' +
            '.tq-box .tq-bar>div.tq-low{background:#dc2626}' +
            '.tq-box .tq-q{font-size:1.12rem;font-weight:600;color:#0f172a;margin:0 0 14px 0;line-height:1.45}' +
            '.tq-box .tq-opt{display:block;width:100%;text-align:left;padding:11px 14px;margin:0 0 10px 0;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;font:inherit;font-size:.98rem;color:#1e293b;cursor:pointer;line-height:1.4}' +
            '.tq-box .tq-opt:hover:not(:disabled){background:#eff6ff;border-color:#2563eb}' +
            '.tq-box .tq-opt:disabled{cursor:default}' +
            '.tq-box .tq-opt b{display:inline-block;width:1.6em;color:#2563eb}' +
            '.tq-box .tq-opt.tq-right{background:#dcfce7;border-color:#16a34a;color:#14532d}' +
            '.tq-box .tq-opt.tq-wrong{background:#fee2e2;border-color:#dc2626;color:#7f1d1d}' +
            '.tq-box .tq-msg{min-height:1.4em;font-weight:600;margin-top:4px}' +
            '.tq-box .tq-btn{background:#1565c0;color:#fff;border:0;border-radius:6px;padding:10px 22px;font:inherit;font-weight:600;cursor:pointer}' +
            '.tq-box .tq-btn:hover{background:#0d47a1}' +
            '.tq-box .tq-score{display:flex;gap:14px;margin:16px 0;flex-wrap:wrap}' +
            '.tq-box .tq-score>div{flex:1;min-width:140px;text-align:center;padding:16px;border-radius:8px}' +
            '.tq-box .tq-score .n{font-size:2.2rem;font-weight:700;line-height:1.1}' +
            '.tq-box .tq-review{margin:10px 0 18px 0;padding-left:20px;font-size:.92rem;color:#334155}' +
            '.tq-box .tq-review li{margin-bottom:8px}';
        document.head.appendChild(st);
    }

    /* ---- state ---- */
    var state = null, timer = null, fillEl = null, secsEl = null, msgEl = null;
    box.className = (box.className ? box.className + ' ' : '') + 'tq-box';
    box.setAttribute('aria-live', 'polite');

    function shuffle(arr) {
        var a = arr.slice(), i, j, t;
        for (i = a.length - 1; i > 0; i--) { j = Math.floor(Math.random() * (i + 1)); t = a[i]; a[i] = a[j]; a[j] = t; }
        return a;
    }

    function showStart() {
        clearInterval(timer);
        fillEl = null;
        box.innerHTML =
            '<p style="margin-top:0;">' + QUESTIONS.length + ' multiple-choice questions, <strong>' + TIME_LIMIT +
            ' seconds</strong> each. Click an answer (or press <kbd>1</kbd>&ndash;<kbd>4</kbd>) before the bar runs out. ' +
            'Unanswered questions count as wrong.</p>' +
            '<button class="tq-btn" data-act="start">Start Quiz</button>';
    }

    function begin() {
        state = { i: 0, right: 0, wrong: 0, timedOut: 0, missed: [], locked: false, opts: [], deadline: 0 };
        showQuestion();
    }

    function showQuestion() {
        var item = QUESTIONS[state.i], html, k;
        state.locked = false;
        state.opts = shuffle([{ t: item.a, ok: true }].concat(item.w.map(function (x) { return { t: x, ok: false }; })));
        html = '<div class="tq-meta"><span>Question ' + (state.i + 1) + ' of ' + QUESTIONS.length + '</span>' +
               '<span class="tq-secs">' + TIME_LIMIT + 's</span></div>' +
               '<div class="tq-bar"><div class="tq-fill"></div></div>' +
               '<p class="tq-q">' + item.q + '</p>';
        for (k = 0; k < state.opts.length; k++) {
            html += '<button class="tq-opt" data-k="' + k + '"><b>' + LETTERS[k] + '.</b>' + state.opts[k].t + '</button>';
        }
        html += '<div class="tq-msg"></div>';
        box.innerHTML = html;
        fillEl = box.querySelector('.tq-fill');
        secsEl = box.querySelector('.tq-secs');
        msgEl  = box.querySelector('.tq-msg');

        state.deadline = Date.now() + TIME_LIMIT * 1000;
        clearInterval(timer);
        timer = setInterval(tick, 50);
    }

    function tick() {
        var left = state.deadline - Date.now();
        if (left <= 0) { answer(-1); return; }
        fillEl.style.width = (left / (TIME_LIMIT * 1000) * 100) + '%';
        fillEl.className = 'tq-fill' + (left < 1500 ? ' tq-low' : '');
        secsEl.textContent = Math.ceil(left / 1000) + 's';
    }

    function answer(k) {                       // k = -1 means time ran out
        if (state.locked) return;
        state.locked = true;
        clearInterval(timer);
        var btns = box.querySelectorAll('.tq-opt'), j, good = (k >= 0 && state.opts[k].ok);
        for (j = 0; j < btns.length; j++) {
            btns[j].disabled = true;
            if (state.opts[j].ok) btns[j].className += ' tq-right';
        }
        if (good) {
            state.right++;
            msgEl.style.color = '#15803d'; msgEl.textContent = 'Correct!';
        } else {
            state.wrong++;
            if (k >= 0) { btns[k].className += ' tq-wrong'; msgEl.textContent = 'Not quite \u2014 the correct answer is highlighted.'; }
            else        { state.timedOut++; msgEl.textContent = 'Time\u2019s up! The correct answer is highlighted.'; }
            msgEl.style.color = '#b91c1c';
            state.missed.push({ q: QUESTIONS[state.i].q, a: QUESTIONS[state.i].a });
        }
        fillEl.style.width = '0%';
        setTimeout(function () {
            state.i++;
            if (state.i < QUESTIONS.length) showQuestion(); else showResult();
        }, FEEDBACK_MS);
    }

    function showResult() {
        var total = QUESTIONS.length, pct = Math.round(state.right / total * 100), html, k;
        fillEl = null;
        html = '<h3 style="margin:0 0 4px 0;color:#0f172a;">Quiz complete</h3>' +
               '<p style="margin:0;color:#475569;">You scored ' + state.right + ' / ' + total + ' (' + pct + '%).</p>' +
               '<div class="tq-score">' +
                 '<div style="background:#dcfce7;color:#14532d;"><div class="n">' + state.right + '</div>Correct</div>' +
                 '<div style="background:#fee2e2;color:#7f1d1d;"><div class="n">' + state.wrong + '</div>Wrong' +
                    (state.timedOut ? '<br><small>(' + state.timedOut + ' timed out)</small>' : '') + '</div>' +
               '</div>';
        if (state.missed.length) {
            html += '<p style="margin:0 0 4px 0;font-weight:600;">Review the ones you missed:</p><ol class="tq-review">';
            for (k = 0; k < state.missed.length; k++) {
                html += '<li>' + state.missed[k].q + '<br><span style="color:#15803d;">&#10003; ' + state.missed[k].a + '</span></li>';
            }
            html += '</ol>';
        } else {
            html += '<p style="color:#15803d;font-weight:600;">Perfect score &mdash; nothing to review!</p>';
        }
        html += '<button class="tq-btn" data-act="start">Try Again</button>';
        box.innerHTML = html;
    }

    /* ---- events ---- */
    box.addEventListener('click', function (e) {
        var el = e.target.closest('button');
        if (!el || !box.contains(el)) return;
        if (el.getAttribute('data-act') === 'start') begin();
        else if (el.hasAttribute('data-k')) answer(parseInt(el.getAttribute('data-k'), 10));
    });

    document.addEventListener('keydown', function (e) {      // 1-4 pick an answer while this quiz has a live question
        if (!state || state.locked || !fillEl || !document.body.contains(fillEl)) return;
        var n = parseInt(e.key, 10);
        if (n >= 1 && n <= state.opts.length) answer(n - 1);
    });

    showStart();
}