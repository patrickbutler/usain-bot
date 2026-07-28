(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  async function api(path, options = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error((body && body.detail) || res.statusText);
    return body;
  }

  function render(settings) {
    const toggle = $("#jamaican-toggle");
    toggle.classList.toggle("on", settings.jamaican_mode);
    toggle.setAttribute("aria-checked", String(settings.jamaican_mode));
    const provider = $("#provider-info");
    provider.classList.remove("loading");
    provider.textContent = `${settings.chat_provider} · ${settings.chat_model}`;
  }

  async function load() {
    try {
      render(await api("/api/settings"));
    } catch (e) {
      $("#settings-note").textContent = `Couldn't load settings: ${e.message}`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    load();
    $("#jamaican-toggle").addEventListener("click", async () => {
      const turningOn = !$("#jamaican-toggle").classList.contains("on");
      $("#settings-note").textContent = "Saving…";
      try {
        render(await api("/api/settings", {
          method: "POST",
          body: JSON.stringify({ jamaican_mode: turningOn }),
        }));
        $("#settings-note").textContent = turningOn
          ? "Jamaican mode ON — the coach speaks with the accent."
          : "Jamaican mode OFF — professional voice for demos.";
      } catch (e) {
        $("#settings-note").textContent = `Couldn't save: ${e.message}`;
      }
    });
  });
})();
