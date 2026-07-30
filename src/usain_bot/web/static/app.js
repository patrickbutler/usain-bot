(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let body = null;
    try { body = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) {
      const detail = body && body.detail ? body.detail : res.statusText;
      throw new Error(detail);
    }
    return body;
  }

  // ---------- tabs ----------
  function initTabs() {
    $$(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        $$(".tab").forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
        $$(".panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        const panel = $(`#panel-${tab.dataset.tab}`);
        panel.classList.add("active");
        if (tab.dataset.tab === "upcoming") loadUpcoming();
        if (tab.dataset.tab === "history") loadHistory();
      });
    });
  }

  // ---------- status pill ----------
  async function loadStatus() {
    const pill = $("#status-pill");
    try {
      const s = await api("/api/status");
      if (!s.has_plan) {
        pill.textContent = "no plan yet — run `usain-bot init`";
        return;
      }
      const synced = s.last_sync ? timeAgo(new Date(s.last_sync)) : "never";
      pill.textContent = `plan v${s.plan_version} · synced ${synced}`;
    } catch (e) {
      pill.textContent = "offline";
    }
  }

  function timeAgo(date) {
    const secs = Math.floor((Date.now() - date.getTime()) / 1000);
    if (secs < 60) return "just now";
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  }

  // ---------- markdown ----------
  // Deliberately self-contained: no CDN, no bundler, no third-party parser.
  // Everything is HTML-escaped up front, so the only tags that survive are
  // the ones this function emits itself.
  function renderInline(text) {
    return text
      .replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, `<a href="$2" target="_blank" rel="noopener">$1</a>`);
  }

  function renderMarkdown(src) {
    const lines = escapeHtml(String(src == null ? "" : src)).split("\n");
    const out = [];
    let list = null;      // "ul" | "ol" | null
    let inCode = false;
    let codeBuf = [];
    let paraBuf = [];

    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    const flushPara = () => {
      if (paraBuf.length) { out.push(`<p>${renderInline(paraBuf.join(" "))}</p>`); paraBuf = []; }
    };
    const openList = (kind) => { if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; } };

    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");

      if (/^\s*```/.test(line)) {
        if (inCode) { out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`); codeBuf = []; inCode = false; }
        else { flushPara(); closeList(); inCode = true; }
        continue;
      }
      if (inCode) { codeBuf.push(raw); continue; }

      if (!line.trim()) { flushPara(); closeList(); continue; }

      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flushPara(); closeList();
        const level = Math.min(heading[1].length + 2, 6);   // # -> h3, keeps h1/h2 for the app chrome
        out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }

      if (/^(\*\*\*|---|___)\s*$/.test(line.trim())) { flushPara(); closeList(); out.push("<hr>"); continue; }

      const quote = line.match(/^&gt;\s?(.*)$/);
      if (quote) { flushPara(); closeList(); out.push(`<blockquote>${renderInline(quote[1])}</blockquote>`); continue; }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) { flushPara(); openList("ul"); out.push(`<li>${renderInline(ul[1])}</li>`); continue; }

      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ol) { flushPara(); openList("ol"); out.push(`<li>${renderInline(ol[1])}</li>`); continue; }

      // Pipe table: header row, separator, then body rows.
      if (/^\s*\|.*\|\s*$/.test(line)) {
        flushPara(); closeList();
        const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const last = out[out.length - 1] || "";
        if (/^[\s|:-]+$/.test(line) && last.startsWith("<table>")) continue;   // separator row
        if (last.startsWith("<table>")) {
          out[out.length - 1] = last.replace("</table>", `<tr>${cells(line).map((c) => `<td>${renderInline(c)}</td>`).join("")}</tr></table>`);
        } else {
          out.push(`<table><tr>${cells(line).map((c) => `<th>${renderInline(c)}</th>`).join("")}</tr></table>`);
        }
        continue;
      }

      paraBuf.push(line.trim());
    }
    if (inCode && codeBuf.length) out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`);
    flushPara();
    closeList();
    return out.join("");
  }

  // ---------- chat ----------
  function appendMessage(role, text, { markdown = false } = {}) {
    const log = $("#chat-log");
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const bubble = document.createElement("div");
    if (markdown) {
      bubble.className = "bubble md";
      bubble.innerHTML = renderMarkdown(text);
    } else {
      bubble.className = "bubble plain";
      bubble.textContent = text;
    }
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return wrap;
  }

  function appendToolNote(toolCalls) {
    if (!toolCalls || !toolCalls.length) return;
    const log = $("#chat-log");
    const wrap = document.createElement("div");
    wrap.className = "msg tool-note";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = `used: ${toolCalls.join(", ")}`;
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  // Tools that can change what the other tabs show. When the coach runs one,
  // the Upcoming tab is re-fetched immediately, so a plan change made in
  // conversation is visible in the plan without a manual reload.
  const MUTATING_TOOLS = new Set([
    "publish_draft_plan", "ease_upcoming_week", "shift_marathon_date", "push_milestone",
    "set_long_run_day_preference", "set_health_flag", "clear_health_flag",
    "record_run_feeling", "trigger_garmin_sync",
  ]);

  async function refreshViewsAfterChat(toolCalls) {
    if (!toolCalls || !toolCalls.some((t) => MUTATING_TOOLS.has(t))) return;
    await loadUpcoming();
    if ($("#panel-history").classList.contains("active")) loadHistory();
  }

  function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }

  async function sendChat(message) {
    const input = $("#chat-input");
    const sendBtn = $("#chat-send");
    const note = $("#chat-note");
    appendMessage("user", message);
    input.value = "";
    autoGrow(input);
    sendBtn.disabled = true;
    note.textContent = "thinking…";
    try {
      const result = await api("/api/chat", { method: "POST", body: JSON.stringify({ message }) });
      appendMessage("agent", result.reply, { markdown: true });
      appendToolNote(result.tool_calls);
      note.textContent = "";
      await refreshViewsAfterChat(result.tool_calls);
    } catch (e) {
      note.textContent = String(e.message || e);
      appendMessage("agent", `Couldn't get a reply: ${e.message || e}`);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function submitChat() {
    const input = $("#chat-input");
    const text = input.value.trim();
    if (text) sendChat(text);
  }

  function initChat() {
    const input = $("#chat-input");

    $("#chat-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      submitChat();
    });

    // Enter inserts a newline (the textarea's own default); sending is
    // deliberate — the Send button or Cmd/Ctrl+Enter.
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
        ev.preventDefault();
        submitChat();
      }
    });
    input.addEventListener("input", () => autoGrow(input));
    autoGrow(input);

    $$("#quick-actions button").forEach((btn) => {
      btn.addEventListener("click", () => sendChat(btn.dataset.quick));
    });
  }

  // ---------- upcoming ----------
  function zoneClass(acwr) {
    if (acwr === null || acwr === undefined) return "";
    if (acwr > 1.5) return "zone-red";
    if (acwr > 1.3) return "zone-yellow";
    return "zone-green";
  }

  let activeFlag = null;

  function renderFlagButtons() {
    ["hip", "back", "fatigue"].forEach((f) => {
      const btn = $(`#flag-${f}`);
      const isActive = activeFlag === f;
      btn.classList.toggle("active", isActive);
      btn.title = isActive ? `${f} flag active — click to clear it` : `Toggle ${f} flag`;
    });
  }

  function renderToday(payload) {
    const rec = payload.recommendation;
    const a = payload.anchors;
    const body = $("#today-body");
    activeFlag = payload.active_health_flag || null;
    renderFlagButtons();
    const dist = rec.target_distance_mi !== null ? `${rec.target_distance_mi.toFixed(1)} mi` : "—";
    body.innerHTML = `
      <div class="rec-headline">
        <span class="rec-distance">${dist}</span>
        <span class="rec-type">${rec.run_type}</span>
      </div>
      <div class="rec-line">${escapeHtml(rec.effort_guidance)}</div>
      <div class="binding-chip">${escapeHtml(rec.binding_constraint)}</div>
      <div class="rec-line" style="margin-top:10px"><span class="label">Anchors:</span>
        acute ${a.acute_load_mi} mi · chronic ${a.chronic_load_mi} mi/wk · long-run anchor ${a.long_run_anchor_mi} mi ·
        ACWR <span class="${zoneClass(a.acwr)}">${a.acwr ?? "n/a"}</span> · gap ${a.gap_days}d (${a.gap_severity})
      </div>
      <ul class="reasoning-list">${rec.reasoning.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      <div class="rec-line" style="margin-top:8px"><span class="label">Unlocks more:</span> ${escapeHtml(rec.unlock_next_time)}</div>
      ${activeFlag ? `<div class="rec-line zone-yellow" style="margin-top:8px">⚑ <strong>${escapeHtml(activeFlag)}</strong> flag is active and capping today. Click the <strong>${escapeHtml(activeFlag)}</strong> button above to clear it.</div>` : ""}
      ${payload.runs_per_week_actual ? `<div class="rec-line"><span class="label">Actual running frequency:</span> ${payload.runs_per_week_actual} day(s)/week (plan adapts to this)</div>` : ""}
      ${!payload.sync.live ? `<div class="rec-line zone-yellow" style="margin-top:8px">⚠ ${escapeHtml(payload.sync.message)}</div>` : ""}
    `;

    renderUnratedRuns(payload.unrated_recent_runs || []);

    const next7 = $("#next7-body");
    next7.innerHTML = payload.next_7_days.map((d) => `
      <div class="day-row">
        <span class="day-date">${d.weekday.slice(0, 3)} ${d.date.slice(5)}</span>
        <span class="day-type">${d.run_type.replace(/_/g, " ")}</span>
        <span class="day-dist">${d.distance_mi !== null ? d.distance_mi.toFixed(1) + " mi" : ""}</span>
      </div>
    `).join("");
  }

  const SCORE_LABELS = { 1: "Awful", 2: "Rough", 3: "Okay", 4: "Good", 5: "Great" };

  function renderUnratedRuns(runs) {
    const el = $("#feelings-body");
    const prompt = runs.length
      ? `<p class="rec-line">Recent runs you haven't rated yet — how did they feel?</p>
         ${runs.slice(0, 5).map((r) => `
           <div class="feeling-row" data-date="${r.date}">
             <span class="day-date">${r.date.slice(5)}</span>
             <span class="day-type"><span class="cap">${r.run_class}</span> · ${r.distance_mi.toFixed(1)} mi</span>
             <span class="score-buttons">
               ${[1, 2, 3, 4, 5].map((s) => `<button class="score-btn" data-date="${r.date}" data-score="${s}" title="${SCORE_LABELS[s]}">${s}</button>`).join("")}
             </span>
           </div>`).join("")}`
      : `<p class="hint-text">All recent runs rated — nice. Log more any time via chat.</p>`;
    el.innerHTML = prompt + `<div id="feelings-history"></div>`;

    $$(".score-btn", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          await api("/api/feelings", {
            method: "POST",
            body: JSON.stringify({ score: Number(btn.dataset.score), activity_date: btn.dataset.date }),
          });
        } finally {
          loadUpcoming();
        }
      });
    });
    loadFeelingsHistory();
  }

  async function loadFeelingsHistory() {
    const el = $("#feelings-history");
    if (!el) return;
    try {
      const data = await api("/api/feelings?days=21");
      if (!data.count) { el.innerHTML = ""; return; }
      el.innerHTML = `
        <p class="rec-line" style="margin-top:12px"><span class="label">Last 21 days:</span>
          mean ${data.mean_score}/5 across ${data.count} check-in(s)</p>
        ${data.entries.slice(0, 6).map((e) => `
          <div class="day-row">
            <span class="day-date">${e.activity_date ? e.activity_date.slice(5) : "unlinked"}</span>
            <span class="day-type">${SCORE_LABELS[e.score] || e.score} (${e.score}/5)</span>
            <span class="day-dist">${escapeHtml(e.comment || "")}</span>
          </div>`).join("")}`;
    } catch (e) {
      el.innerHTML = "";
    }
  }

  function renderValidation(validation) {
    const el = $("#validation-body");
    if (!validation || validation.error) {
      el.innerHTML = `<p class="loading">${escapeHtml((validation && validation.error) || "unavailable")}</p>`;
      return;
    }
    const badge = validation.valid
      ? `<span class="valid-badge ok">PASSES</span>`
      : `<span class="valid-badge bad">${validation.error_count} ERROR(S)</span>`;
    const issues = (validation.issues || []).map((i) => `
      <div class="issue-row ${i.severity}">
        <span class="issue-sev">${i.severity}</span>
        <span class="issue-msg">${escapeHtml(i.message)}</span>
      </div>`).join("");
    el.innerHTML = `
      <p class="rec-line">${badge} plan v${validation.plan_version} —
        ${validation.error_count} error(s), ${validation.warning_count} warning(s)</p>
      ${issues || `<p class="hint-text">All milestone and build-rate checks passed.</p>`}`;
  }

  const STAGE_BLURB = {
    increase_mileage: "climbing toward the peak long run",
    maintain_mileage: "peak reached — volume stays high while the long run oscillates, so peak weeks are never back-to-back",
    taper: "strategic reduction into a milestone",
    reduce_mileage: "below 15 mi, no milestone being chased",
  };

  // Phases come from the server (stages.stage_summary) rather than being
  // regrouped here, so the legend, the chat and the API can't disagree
  // about the shape of the plan.
  function renderStageLegend(phases) {
    if (!phases || phases.length < 2) return "";
    return `<div class="stage-legend">${phases.map((p) => `
      <span class="stage-tag stage-${p.stage || "unknown"}" title="${escapeHtml(STAGE_BLURB[p.stage] || "")}">
        ${escapeHtml(p.label || "")} · ${p.weeks}w
      </span>`).join("")}</div>`;
  }

  function renderPlan(plan) {
    const body = $("#plan-body");
    if (plan.error) {
      body.innerHTML = `<p class="loading">${escapeHtml(plan.error)}</p>`;
      return;
    }
    renderValidation(plan.validation);
    const rows = plan.weeks.map((w) => `
      <tr class="${w.is_backoff ? "backoff" : ""}">
        <td>${w.week_number}</td>
        <td>${w.start_date}</td>
        <td><span class="week-tag">${w.block.replace(/_/g, " ")}</span></td>
        <td><span class="stage-tag stage-${w.stage || "unknown"}">${(w.stage || "").replace(/_/g, " ")}</span></td>
        <td>${w.target_volume_mi.toFixed(1)} mi</td>
        <td>${w.long_run_mi.toFixed(1)} mi</td>
        <td>${w.quality_sessions}</td>
      </tr>
    `).join("");
    body.innerHTML = `
      ${renderStageLegend(plan.stages)}
      <table>
        <thead><tr><th>Wk</th><th>Start</th><th>Block</th><th>Stage</th><th>Volume</th><th>Long run</th><th>Quality</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${plan.half_marathon_capability_week ? `<p class="rec-line" style="margin-top:10px">Half-marathon capability projected: week ${plan.half_marathon_capability_week.week_number} (${plan.half_marathon_capability_week.start_date})</p>` : ""}
    `;
  }

  function fieldLabel(field) {
    return { target_volume_mi: "volume", long_run_mi: "long run", block: "block", is_backoff: "back-off" }[field] || field;
  }

  function formatWeekValue(field, week) {
    if (!week) return "";
    if (field === "target_volume_mi") return `${week.target_volume_mi.toFixed(1)} mi`;
    if (field === "long_run_mi") return `${week.long_run_mi.toFixed(1)} mi`;
    if (field === "block") return week.block.replace(/_/g, " ");
    if (field === "is_backoff") return week.is_backoff ? "back-off" : "normal";
    return "";
  }

  function renderVersionDiffTable(weekDiffs) {
    if (!weekDiffs.length) return `<p class="hint-text">No week-level changes recorded.</p>`;
    const rows = weekDiffs.map((d) => {
      if (d.change_type === "added") {
        return `<tr><td>wk ${d.week_number}</td><td class="diff-added">+ added</td>
          <td colspan="2">${escapeHtml(formatWeekValue("block", d.new))} · ${formatWeekValue("target_volume_mi", d.new)} · LR ${formatWeekValue("long_run_mi", d.new)}</td></tr>`;
      }
      if (d.change_type === "removed") {
        return `<tr><td>wk ${d.week_number}</td><td class="diff-removed">removed</td><td colspan="2">was ${escapeHtml(formatWeekValue("block", d.old))}</td></tr>`;
      }
      const fields = d.changed_fields.map((f) => `
        <div><span class="hint-text">${fieldLabel(f)}:</span>
          <span class="diff-old">${escapeHtml(formatWeekValue(f, d.old))}</span>
          <span class="diff-new">${escapeHtml(formatWeekValue(f, d.new))}</span>
        </div>
      `).join("");
      return `<tr><td>wk ${d.week_number}</td><td>changed</td><td colspan="2">${fields}</td></tr>`;
    }).join("");
    return `<table><tbody>${rows}</tbody></table>`;
  }

  function renderPlanHistory(payload) {
    const body = $("#plan-history-body");
    const versions = payload.versions || [];
    if (!versions.length) {
      body.innerHTML = `<p class="loading">No plan history yet.</p>`;
      return;
    }
    body.innerHTML = `<div class="version-list">${versions.map((v) => `
      <div class="version-row" data-version="${v.version}">
        <div class="version-row-head">
          <span class="expand-icon">▸</span>
          <span class="version-badge">v${v.version}</span>
          <span class="trigger-badge trigger-${v.trigger}">${v.trigger.replace(/_/g, " ")}</span>
          <span class="version-date">${new Date(v.created_at).toLocaleString()}</span>
          <span class="version-changed">${v.weeks_changed_count} week${v.weeks_changed_count === 1 ? "" : "s"} changed</span>
        </div>
        <div class="version-rationale">${escapeHtml(v.rationale)}</div>
        <div class="version-diff-detail">${renderVersionDiffTable(v.week_diffs)}</div>
      </div>
    `).join("")}</div>`;

    $$(".version-row-head", body).forEach((head) => {
      head.addEventListener("click", () => head.parentElement.classList.toggle("expanded"));
    });
  }

  async function loadUpcoming() {
    try {
      const [today, plan, history] = await Promise.all([api("/api/today"), api("/api/plan"), api("/api/plan/history")]);
      renderToday(today);
      renderPlan(plan);
      renderPlanHistory(history);
    } catch (e) {
      $("#today-body").innerHTML = `<p class="loading">Couldn't load: ${escapeHtml(e.message || String(e))}</p>`;
    }
    loadStatus();
  }

  function initUpcoming() {
    $("#refresh-today").addEventListener("click", async () => {
      $("#today-body").innerHTML = `<p class="loading">Refreshing…</p>`;
      await api("/api/today/refresh", { method: "POST" });
      loadUpcoming();
    });
    // Flags toggle: clicking an already-active flag clears it, so a
    // mis-tap is undoable without leaving the UI.
    ["hip", "back", "fatigue"].forEach((flag) => {
      $(`#flag-${flag}`).addEventListener("click", async () => {
        const clearing = activeFlag === flag;
        $("#today-body").innerHTML = `<p class="loading">${clearing ? `Clearing ${flag} flag…` : `Applying ${flag} flag…`}</p>`;
        try {
          if (clearing) {
            await api("/api/health-flag", { method: "DELETE" });
          } else {
            await api("/api/health-flag", { method: "POST", body: JSON.stringify({ flag }) });
          }
        } finally {
          loadUpcoming();
        }
      });
    });
  }

  // ---------- history ----------
  function classClass(runClass) {
    if (runClass === "long") return "class-long";
    if (runClass === "quality") return "class-quality";
    if (runClass === "recovery") return "class-recovery";
    return "";
  }

  function renderHistoryChart(weekly) {
    const el = $("#history-chart");
    if (!weekly.length) { el.innerHTML = `<p class="loading">No history yet.</p>`; return; }
    const max = Math.max(...weekly.map((w) => w.total_mi), 1);
    el.innerHTML = `<div class="bars">${weekly.map((w) => `
      <div class="bar-wrap" title="${w.week_start}: ${w.total_mi} mi">
        <div class="bar" style="height:${Math.max((w.total_mi / max) * 130, 2)}px"></div>
        <span class="bar-label">${w.week_start.slice(5)}</span>
      </div>
    `).join("")}</div>`;
  }

  function renderHistoryTable(activities) {
    const el = $("#history-table");
    if (!activities.length) { el.innerHTML = `<p class="loading">No activities yet — try syncing.</p>`; return; }
    const rows = activities.slice(0, 200).map((a) => `
      <tr>
        <td>${a.date}</td>
        <td class="${classClass(a.run_class)}">${a.run_class}</td>
        <td>${a.type}</td>
        <td>${a.distance_mi.toFixed(2)} mi</td>
        <td>${a.avg_hr ?? ""}</td>
        <td>${escapeHtml(a.name || "")}</td>
      </tr>
    `).join("");
    el.innerHTML = `
      <table>
        <thead><tr><th>Date</th><th>Class</th><th>Type</th><th>Distance</th><th>Avg HR</th><th>Name</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function renderRunFeelings(data) {
    const el = $("#history-feelings");
    const runs = data.runs || [];
    if (!runs.length) {
      el.innerHTML = `<p class="loading">No runs in this window yet.</p>`;
      return;
    }
    const summary = data.mean_score !== null && data.mean_score !== undefined
      ? `<p class="rec-line"><span class="label">Average feel:</span> ${data.mean_score}/5
           across ${data.rated_count} rated run(s) · ${data.unrated_count} not scored yet</p>`
      : `<p class="rec-line">None of these runs are scored yet — tap a number, or just tell the coach in chat.</p>`;

    // A scored run shows its score and is never asked about again; only
    // unscored runs get the 1-5 buttons.
    const rows = runs.map((r) => `
      <div class="feeling-row ${r.score ? "rated" : ""}">
        <span class="day-date">${r.date.slice(5)}</span>
        <span class="day-type"><span class="cap">${r.run_class}</span> · ${r.distance_mi.toFixed(1)} mi</span>
        ${r.score
          ? `<span class="day-dist">${escapeHtml(r.comment || "")}</span>
             <span class="score-chip score-${r.score}">${r.score}/5 ${SCORE_LABELS[r.score] || ""}</span>`
          : `<span class="score-buttons">${[1, 2, 3, 4, 5]
              .map((s) => `<button class="score-btn" data-date="${r.date}" data-score="${s}" title="${SCORE_LABELS[s]}">${s}</button>`)
              .join("")}</span>`}
      </div>`).join("");

    // Newest first in a bounded scroller — a 90-day window is 35+ runs and
    // would otherwise push the rest of the tab off the screen.
    el.innerHTML = summary + `<div class="feelings-scroll">${rows}</div>`;

    $$(".score-btn", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        $$(".score-btn", el).forEach((b) => { b.disabled = true; });
        try {
          await api("/api/feelings", {
            method: "POST",
            body: JSON.stringify({ score: Number(btn.dataset.score), activity_date: btn.dataset.date }),
          });
        } finally {
          loadRunFeelings();
        }
      });
    });
  }

  async function loadRunFeelings() {
    const el = $("#history-feelings");
    const days = $("#history-window").value;
    try {
      renderRunFeelings(await api(`/api/runs/feelings?days=${days}`));
    } catch (e) {
      el.innerHTML = `<p class="loading">Couldn't load: ${escapeHtml(e.message || String(e))}</p>`;
    }
  }

  async function loadHistory() {
    const days = $("#history-window").value;
    $("#history-chart").innerHTML = `<p class="loading">Loading…</p>`;
    $("#history-table").innerHTML = `<p class="loading">Loading…</p>`;
    $("#history-feelings").innerHTML = `<p class="loading">Loading…</p>`;
    loadRunFeelings();
    try {
      const data = await api(`/api/history?days=${days}`);
      renderHistoryChart(data.weekly_volume);
      renderHistoryTable(data.activities);
    } catch (e) {
      $("#history-chart").innerHTML = `<p class="loading">Couldn't load: ${escapeHtml(e.message || String(e))}</p>`;
    }
  }

  function initHistory() {
    $("#history-window").addEventListener("change", loadHistory);
    $("#sync-now").addEventListener("click", async () => {
      $("#sync-now").disabled = true;
      $("#sync-now").textContent = "Syncing…";
      try {
        await api("/api/sync", { method: "POST" });
      } finally {
        $("#sync-now").disabled = false;
        $("#sync-now").textContent = "Sync Garmin";
        loadHistory();
        loadStatus();
      }
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initChat();
    initUpcoming();
    initHistory();
    loadStatus();
    loadUpcoming();
  });
})();
