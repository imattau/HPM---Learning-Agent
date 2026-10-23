"use strict";
// HPM study assistant frontend. All text from sources and replies is inserted
// with textContent; nothing here uses innerHTML.

const $ = (id) => document.getElementById(id);
const state = { config: null, sources: [], convs: [], current: null, busy: false, sourceCache: {} };
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined || method === "POST") {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body ?? {});
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || `Request failed (${res.status})`);
  return data;
}

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid) node.append(kid);
  return node;
}

// ---- rendering assistant messages ---------------------------------------
function chips(cites, citations) {
  const frag = document.createDocumentFragment();
  for (const n of cites) {
    const c = citations[n - 1];
    frag.append(el("button", {
      class: "chip", type: "button", text: String(n),
      "aria-label": `Open source ${c.title}, characters ${c.start} to ${c.end}`,
      onclick: () => openSource(c),
    }));
  }
  return frag;
}

function itemNode(item, citations, tag = "span") {
  const node = el(tag, { class: "item" });
  node.append(el("span", { class: "t", text: item.text }), chips(item.cites || [], citations));
  return node;
}

function renderBlock(block, msg) {
  const wrap = el("div", { class: "block" });
  if (block.title) wrap.append(el("h2", { text: block.title }));
  const cit = msg.citations;
  switch (block.type) {
    case "paragraph":
      block.items.forEach((i) => wrap.append(el("p", {}, itemNode(i, cit))));
      break;
    case "numbered_list":
    case "bullet_list": {
      const list = el(block.type === "numbered_list" ? "ol" : "ul");
      block.items.forEach((i) => list.append(el("li", {}, itemNode(i, cit))));
      wrap.append(list);
      break;
    }
    case "flashcards": {
      const cards = el("div", { class: "cards" });
      block.items.forEach((i, idx) => cards.append(el("div", { class: "card" },
        el("div", { class: "front", text: `Card ${idx + 1} · ${i.front}` }), itemNode(i, cit, "div"))));
      wrap.append(cards);
      break;
    }
    case "compare": {
      const grid = el("div", { class: "compare" });
      block.items.forEach((i) => grid.append(el("div", { class: "card" }, el("h3", { text: i.label }), itemNode(i, cit, "div"))));
      wrap.append(grid);
      break;
    }
    case "clarify": {
      const box = el("div", { class: "notice clarify" }, el("div", { text: block.text }));
      const opts = el("div", { class: "options" });
      for (const o of block.options) {
        opts.append(el("button", {
          class: "btn", type: "button", text: o.label,
          onclick: () => send(msg.why.effective_question, o.source_id, o.label),
        }));
      }
      box.append(opts);
      wrap.append(box);
      break;
    }
    case "abstain": {
      const box = el("div", { class: "notice" }, el("div", { text: block.text }));
      box.append(el("small", { text: `Sources searched: ${state.sources.length}. Try rephrasing, or ask about one of them.` }));
      wrap.append(box);
      break;
    }
    default:
      wrap.append(el("div", { class: "notice error", text: block.text || "This reply could not be displayed." }));
  }
  return wrap;
}

function metaLine(msg) {
  const meta = el("div", { class: "meta" });
  meta.append(el("span", { class: `badge ${msg.act}`, text: msg.act }));
  meta.append(el("span", { text: msg.task_family.replace("_", " ") }));
  meta.append(el("span", { text: `confidence ${msg.confidence.toFixed(2)}`, title: "Calibrated against validation episodes, not a probability of being right." }));
  meta.append(el("span", { class: `badge ${msg.verified ? "" : "bad"}`, text: msg.verified ? "verified" : "unverified" }));
  const why = el("div", { class: "why", hidden: "" });
  why.hidden = true;
  why.append(
    el("div", { text: `reason: ${msg.why.reason || "n/a"}` }),
    el("div", { text: `procedure: ${msg.why.procedure_id || "none"}` }),
    el("div", { text: `question used: ${msg.why.effective_question}` }),
  );
  meta.append(el("button", { class: "link", type: "button", text: "why?", onclick: () => { why.hidden = !why.hidden; } }));
  meta.append(el("button", { class: "link", type: "button", text: "copy", onclick: (e) => copyText(msg.text, e.target) }));
  return [meta, why];
}

function copyText(text, button) {
  const done = () => { const old = button.textContent; button.textContent = "copied"; setTimeout(() => { button.textContent = old; }, 1200); };
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
}

function assistantNode(msg) {
  const node = el("div", { class: "msg assistant" }, el("div", { class: "who", text: "HPM study assistant" }));
  const body = el("div", { class: "body" });
  msg.blocks.forEach((b) => body.append(renderBlock(b, msg)));
  node.append(body);
  const [meta, why] = metaLine(msg);
  node.append(meta, why);
  return node;
}

function userNode(msg) {
  const node = el("div", { class: "msg user" }, el("div", { class: "bubble", text: msg.text }));
  if (msg.scope_source_id) {
    const s = state.sources.find((x) => x.source_id === msg.scope_source_id);
    node.append(el("div", { class: "scope", text: `Using ${s ? s.title : msg.scope_source_id}` }));
  }
  return node;
}

// Cosmetic reveal: text appears word by word. The reply was already computed.
function animate(node, done) {
  const targets = [...node.querySelectorAll(".item > .t")];
  const full = targets.map((t) => t.textContent);
  const words = full.map((s) => s.split(/(\s+)/));
  const total = words.reduce((n, w) => n + w.length, 0);
  const perTick = Math.max(1, Math.ceil(total / 40));
  node.classList.add("hide-cites");
  targets.forEach((t) => { t.textContent = ""; });
  let ti = 0, wi = 0;
  const timer = setInterval(() => {
    for (let k = 0; k < perTick && ti < targets.length; k++) {
      targets[ti].textContent += words[ti][wi++];
      if (wi >= words[ti].length) { ti++; wi = 0; }
    }
    scrollDown();
    if (ti >= targets.length) { clearInterval(timer); targets.forEach((t, i) => { t.textContent = full[i]; }); node.classList.remove("hide-cites"); done(); }
  }, 25);
}

function scrollDown() { const t = $("thread"); t.scrollTop = t.scrollHeight; }

// ---- source panel -----------------------------------------------------------
async function openSource(c) {
  let source = state.sourceCache[c.source_id];
  if (!source) {
    try { source = state.sourceCache[c.source_id] = await api("GET", `/api/sources/${encodeURIComponent(c.source_id)}`); }
    catch (err) { return showError(err.message); }
  }
  $("panel-title").textContent = source.title;
  const body = $("panel-body");
  body.textContent = "";
  const mark = el("mark", { text: source.text.slice(c.start, c.end) });
  body.append(source.text.slice(0, c.start), mark, source.text.slice(c.end));
  $("panel").hidden = false;
  $("app").classList.add("panel-open");
  $("scrim").hidden = false;
  mark.scrollIntoView({ block: "center" });
}
function closePanel() { $("panel").hidden = true; $("app").classList.remove("panel-open"); $("scrim").hidden = true; }

// ---- thread / conversations -----------------------------------------------
function showError(message) {
  const thread = $("thread");
  thread.append(el("div", { class: "msg assistant" }, el("div", { class: "notice error", text: message })));
  scrollDown();
}

function renderWelcome() {
  const thread = $("thread");
  thread.textContent = "";
  const box = el("div", { class: "welcome" },
    el("h2", { text: "What would you like to study?" }),
    el("p", { text: "Answers come only from your loaded sources, with citations. If they don't cover it, I'll say so." }));
  const grid = el("div", { class: "suggestions" });
  const prompts = [];
  for (const s of state.sources.slice(0, 2)) {
    prompts.push(`Summarize ${s.title}`, `Make flashcards for ${s.title}`);
  }
  if (state.sources[0]) prompts.push(`Outline ${state.sources[0].title}`);
  prompts.slice(0, 4).forEach((p) => grid.append(el("button", { class: "suggestion", type: "button", text: p, onclick: () => send(p) })));
  box.append(grid);
  thread.append(box);
}

function renderThread(conv) {
  const thread = $("thread");
  thread.textContent = "";
  if (!conv || !conv.messages.length) return renderWelcome();
  conv.messages.forEach((m) => thread.append(m.role === "user" ? userNode(m) : assistantNode(m)));
  scrollDown();
}

function renderList() {
  const list = $("conv-list");
  list.textContent = "";
  if (!state.convs.length) list.append(el("div", { class: "empty-list", text: "No conversations yet." }));
  for (const c of state.convs) {
    const row = el("div", { class: "conv" + (state.current && state.current.id === c.id ? " active" : "") },
      el("a", { href: "#", text: c.title, onclick: (e) => { e.preventDefault(); selectConv(c.id); } }),
      el("span", { class: "acts" },
        el("button", { class: "btn icon", type: "button", "aria-label": "Rename", text: "✎", onclick: () => renameConv(c) }),
        el("button", { class: "btn icon", type: "button", "aria-label": "Delete", text: "🗑", onclick: () => deleteConv(c) })));
    list.append(row);
  }
}

async function refreshList() { state.convs = await api("GET", "/api/conversations"); renderList(); }

async function selectConv(id) {
  closeSidebar();
  state.current = await api("GET", `/api/conversations/${id}`);
  $("title").textContent = state.current.title;
  renderThread(state.current);
  renderList();
}

function newChat() {
  state.current = null;
  $("title").textContent = "New chat";
  renderWelcome();
  renderList();
  closeSidebar();
  $("input").focus();
}

async function renameConv(c) {
  const title = window.prompt("Rename conversation", c.title);
  if (!title || !title.trim()) return;
  try { await api("PATCH", `/api/conversations/${c.id}`, { title }); } catch (e) { return showError(e.message); }
  if (state.current && state.current.id === c.id) { state.current.title = title.trim(); $("title").textContent = title.trim(); }
  refreshList();
}

async function deleteConv(c) {
  if (!window.confirm(`Delete "${c.title}"? The usage log keeps its record of the turns.`)) return;
  try { await api("DELETE", `/api/conversations/${c.id}`); } catch (e) { return showError(e.message); }
  if (state.current && state.current.id === c.id) newChat();
  refreshList();
}

async function send(text, scope = null, scopeLabel = null) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  const thread = $("thread");
  if (!state.current) {
    try {
      const created = await api("POST", "/api/conversations", {});
      state.current = { ...created, messages: [] };
      thread.textContent = "";
    } catch (e) { state.busy = false; $("send").disabled = false; return showError(e.message); }
  } else if (!state.current.messages.length) thread.textContent = "";
  const userMsg = { role: "user", text, scope_source_id: scope };
  thread.append(userNode(userMsg));
  const typing = el("div", { class: "msg assistant" }, el("div", { class: "typing", text: "Searching sources" }));
  thread.append(typing);
  scrollDown();
  $("input").value = ""; autosize();
  try {
    const reply = await api("POST", `/api/conversations/${state.current.id}/messages`, { text, scope_source_id: scope });
    typing.remove();
    const node = assistantNode(reply);
    thread.append(node);
    state.current.messages.push(userMsg, reply);
    const finish = () => { state.busy = false; $("send").disabled = false; $("input").focus(); };
    if ($("animate").checked && !matchMedia("(prefers-reduced-motion: reduce)").matches) animate(node, finish);
    else { finish(); scrollDown(); }
    await refreshList();
    const row = state.convs.find((c) => c.id === state.current.id);
    if (row) { state.current.title = row.title; $("title").textContent = row.title; }
  } catch (err) {
    typing.remove();
    showError(err.message);
    state.busy = false; $("send").disabled = false;
  }
}

// ---- chrome ------------------------------------------------------------------
function autosize() {
  const i = $("input");
  i.style.height = "auto";
  i.style.height = Math.min(i.scrollHeight, 180) + "px";
  i.style.overflowY = i.scrollHeight > 180 ? "auto" : "hidden";
}
function openSidebar() { $("app").classList.add("sidebar-open"); $("scrim").hidden = false; }
function closeSidebar() { $("app").classList.remove("sidebar-open"); if ($("panel").hidden) $("scrim").hidden = true; }

function renderConfig() {
  const c = state.config;
  const parts = [`${c.source_count} source${c.source_count === 1 ? "" : "s"}`, c.representation, c.realizer, `user ${c.user_id}`];
  $("config").textContent = parts.filter(Boolean).join(" · ");
}

async function init() {
  $("animate").checked = store.get("animate", "1") === "1";
  $("animate").addEventListener("change", (e) => store.set("animate", e.target.checked ? "1" : "0"));
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send($("input").value); });
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($("input").value); } });
  $("input").addEventListener("input", autosize);
  $("new-chat").addEventListener("click", newChat);
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("close-panel").addEventListener("click", closePanel);
  $("scrim").addEventListener("click", () => { closePanel(); closeSidebar(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePanel(); closeSidebar(); } });
  try {
    [state.config, state.sources] = await Promise.all([api("GET", "/api/config"), api("GET", "/api/sources")]);
    renderConfig();
    await refreshList();
    if (state.convs.length) await selectConv(state.convs[0].id); else renderWelcome();
  } catch (err) { showError(err.message); }
  $("input").focus();
}
init();
