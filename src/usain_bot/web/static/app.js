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

  // ---------- chat ----------
  function appendMessage(role, text) {
    const log = $("#chat-log");
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
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

  async function sendChat(message) {
    const input = $("#chat-input");
    const sendBtn = $("#chat-send");
    const note = $("#chat-note");
    appendMessage("user", message);
    input.value = "";
    sendBtn.disabled = true;
    note.textContent = "thinking…";
    try {
      const result = await api("/api/chat", { method: "POST", body: JSON.stringify({ message }) });
      appendMessage("agent", result.reply);
      appendToolNote(result.tool_calls);
      note.textContent = "";
    } catch (e) {
      note.textContent = String(e.message || e);
      appendMessage("agent", `Couldn't get a reply: ${e.message || e}`);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function initChat() {
    $("#chat-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const input = $("#chat-input");
      const text = input.value.trim();
      if (text) sendChat(text);
    });
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

  function renderToday(payload) {
    const rec = payload.recommendation;
    const a = payload.anchors;
    const body = $("#today-body");
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
      ${!payload.sync.live ? `<div class="rec-line zone-yellow" style="margin-top:8px">⚠ ${escapeHtml(payload.sync.message)}</div>` : ""}
    `;

    const next7 = $("#next7-body");
    next7.innerHTML = payload.next_7_days.map((d) => `
      <div class="day-row">
        <span class="day-date">${d.weekday.slice(0, 3)} ${d.date.slice(5)}</span>
        <span class="day-type">${d.run_type.replace(/_/g, " ")}</span>
        <span class="day-dist">${d.distance_mi !== null ? d.distance_mi.toFixed(1) + " mi" : ""}</span>
      </div>
    `).join("");
  }

  function renderPlan(plan) {
    const body = $("#plan-body");
    if (plan.error) {
      body.innerHTML = `<p class="loading">${escapeHtml(plan.error)}</p>`;
      return;
    }
    const rows = plan.weeks.map((w) => `
      <tr class="${w.is_backoff ? "backoff" : ""}">
        <td>${w.week_number}</td>
        <td>${w.start_date}</td>
        <td><span class="week-tag">${w.block.replace(/_/g, " ")}</span></td>
        <td>${w.target_volume_mi.toFixed(1)} mi</td>
        <td>${w.long_run_mi.toFixed(1)} mi</td>
        <td>${w.quality_sessions}</td>
      </tr>
    `).join("");
    body.innerHTML = `
      <table>
        <thead><tr><th>Wk</th><th>Start</th><th>Block</th><th>Volume</th><th>Long run</th><th>Quality</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${plan.half_marathon_capability_week ? `<p class="rec-line" style="margin-top:10px">Half-marathon capability projected: week ${plan.half_marathon_capability_week.week_number} (${plan.half_marathon_capability_week.start_date})</p>` : ""}
    `;
  }

  async function loadUpcoming() {
    try {
      const [today, plan] = await Promise.all([api("/api/today"), api("/api/plan")]);
      renderToday(today);
      renderPlan(plan);
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
    ["hip", "back", "fatigue"].forEach((flag) => {
      $(`#flag-${flag}`).addEventListener("click", async () => {
        $(`#flag-${flag}`).classList.add("active");
        $("#today-body").innerHTML = `<p class="loading">Applying ${flag} flag…</p>`;
        try {
          await api("/api/health-flag", { method: "POST", body: JSON.stringify({ flag }) });
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

  async function loadHistory() {
    const days = $("#history-window").value;
    $("#history-chart").innerHTML = `<p class="loading">Loading…</p>`;
    $("#history-table").innerHTML = `<p class="loading">Loading…</p>`;
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
