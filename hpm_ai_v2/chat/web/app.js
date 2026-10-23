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

// POST that streams real stage progress (server-sent events) and resolves to the reply.
async function postStream(path, body, onStage) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.message || `Request failed (${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", reply = null;
  const handle = (chunk) => {
    let name = "message", data = "";
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (name === "stage") onStage(payload);
    else if (name === "reply") reply = payload;
    else if (name === "error") throw new Error(payload.message || "The turn failed.");
  };
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) { handle(buffer.slice(0, idx)); buffer = buffer.slice(idx + 2); }
    if (done) break;
  }
  if (buffer.trim()) handle(buffer);
  if (!reply) throw new Error("The connection closed before a reply arrived.");
  return reply;
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
  const [controls, note] = feedbackControls(msg);
  controls.forEach((c) => meta.append(c));
  return [meta, why, note];
}

// ---- feedback: thumbs and "correct this" ------------------------------------
function feedbackControls(msg) {
  if (msg.act === "clarify" || !state.current) return [[], null];
  const done = msg.feedback || [];
  const note = el("div", { class: "fb-note" });
  note.hidden = true;
  const show = (text) => { note.textContent = text; note.hidden = false; };
  const mark = (signal) => { msg.feedback = (msg.feedback || []).concat(signal); };
  const thumbs = ["up", "down"].map((signal) => {
    const chosen = done.includes(signal);
    const button = el("button", {
      class: "link thumb" + (chosen ? " chosen" : ""), type: "button",
      "aria-label": signal === "up" ? "Good answer" : "Bad answer", "aria-pressed": String(chosen),
      text: signal === "up" ? "👍" : "👎",
    });
    button.disabled = done.includes("up") || done.includes("down");
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const r = await api("POST", `/api/conversations/${state.current.id}/messages/${msg.turn_id}/feedback`, { signal });
        mark(signal);
        thumbs.forEach((b) => { b.disabled = true; });
        button.classList.add("chosen"); button.setAttribute("aria-pressed", "true");
        show(r.message);
      } catch (e) { button.disabled = false; show(e.message); }
    });
    return button;
  });
  const correct = el("button", { class: "link", type: "button", text: done.includes("correct") ? "corrected" : "correct this" });
  correct.disabled = done.includes("correct");
  correct.addEventListener("click", () => openPicker(msg, (r) => { mark("correct"); correct.textContent = "corrected"; correct.disabled = true; show(r.message); }));
  return [[...thumbs, correct], note];
}

// The picker reuses the source panel: the user chooses the exact source sentence that answers the question.
async function openPicker(msg, onDone) {
  let rows;
  try { rows = await api("GET", `/api/conversations/${state.current.id}/messages/${msg.turn_id}/candidates`); }
  catch (err) { return showError(err.message); }
  $("panel-title").textContent = "Which sentence answers this?";
  const body = $("panel-body");
  body.textContent = "";
  body.append(el("p", { class: "picker-help", text: "Pick the source sentence that should have been used. Your correction is saved for you only and applies to close rewordings of the question." }));
  let lastSource = null;
  for (const row of rows) {
    if (row.source_id !== lastSource) { body.append(el("h3", { class: "picker-source", text: row.title })); lastSource = row.source_id; }
    const option = el("button", { class: "pick" + (row.current ? " current" : ""), type: "button" },
      el("span", { text: row.text }), row.current ? el("small", { text: " (what I cited)" }) : null);
    option.addEventListener("click", async () => {
      body.querySelectorAll(".pick").forEach((b) => { b.disabled = true; });
      try {
        const r = await api("POST", `/api/conversations/${state.current.id}/messages/${msg.turn_id}/feedback`,
          { signal: "correct", correction: { source_id: row.source_id, start: row.start, end: row.end } });
        body.textContent = "";
        body.append(el("p", { text: r.message }));
        onDone(r);
      } catch (e) { body.append(el("p", { class: "picker-error", text: e.message })); body.querySelectorAll(".pick").forEach((b) => { b.disabled = false; }); }
    });
    body.append(option);
  }
  $("panel").hidden = false;
  $("app").classList.add("panel-open");
  $("scrim").hidden = false;
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
  const [meta, why, note] = metaLine(msg);
  node.append(meta, why);
  if (note) node.append(note);
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
  updateEditLink();
}

// Only the last user message can be edited, and not one that answered a clarification.
function updateEditLink() {
  document.querySelectorAll(".edit-link").forEach((n) => n.remove());
  const msgs = state.current ? state.current.messages : [];
  const last = [...msgs].reverse().find((m) => m.role === "user");
  if (state.busy || !last || last.scope_source_id || !last.turn_id) return;
  const nodes = $("thread").querySelectorAll(".msg.user");
  const node = nodes[nodes.length - 1];
  if (node) node.append(el("button", { class: "link edit-link", type: "button", text: "edit", onclick: () => startEdit(node, last) }));
}

function startEdit(node, msg) {
  if (state.busy) return;
  node.textContent = "";
  const box = el("textarea", { class: "edit-box", rows: "2", "aria-label": "Edit your message", maxlength: "4000" });
  box.value = msg.text;
  const save = el("button", { class: "btn primary", type: "button", text: "Save and resend", onclick: () => submitEdit(msg, box.value) });
  const cancel = el("button", { class: "btn", type: "button", text: "Cancel", onclick: () => { node.textContent = ""; node.append(el("div", { class: "bubble", text: msg.text })); updateEditLink(); } });
  box.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitEdit(msg, box.value); } if (e.key === "Escape") cancel.click(); });
  node.append(box, el("div", { class: "edit-actions" }, cancel, save));
  box.focus();
}

async function submitEdit(msg, text) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  const messages = state.current.messages;
  const at = messages.findIndex((m) => m.turn_id === msg.turn_id && m.role === "user");
  const thread = $("thread");
  const nodes = [...thread.children];
  const userIdx = nodes.map((n) => n.classList.contains("user")).lastIndexOf(true);
  nodes.slice(userIdx).forEach((n) => n.remove());
  const userMsg = { role: "user", text, scope_source_id: null };
  thread.append(userNode(userMsg));
  await runTurn(`/api/conversations/${state.current.id}/messages/${msg.turn_id}/edit`, { text }, userMsg, (reply) => {
    state.current.messages = messages.slice(0, at).concat([userMsg, reply]);
  });
}

function renderList() {
  const list = $("conv-list");
  list.textContent = "";
  if (!state.convs.length) list.append(el("div", { class: "empty-list", text: searchTerm ? "No matches." : "No conversations yet." }));
  for (const c of state.convs) {
    const row = el("div", { class: "conv" + (state.current && state.current.id === c.id ? " active" : "") },
      el("a", { href: "#", text: c.title, onclick: (e) => { e.preventDefault(); selectConv(c.id); } }),
      el("span", { class: "acts" },
        el("button", { class: "btn icon", type: "button", "aria-label": "Rename", text: "✎", onclick: () => renameConv(c) }),
        el("button", { class: "btn icon", type: "button", "aria-label": "Delete", text: "🗑", onclick: () => deleteConv(c) })));
    list.append(row);
  }
}

let searchTerm = "";
async function refreshList() {
  const q = searchTerm ? `?q=${encodeURIComponent(searchTerm)}` : "";
  state.convs = await api("GET", `/api/conversations${q}`);
  renderList();
}

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
  document.querySelectorAll(".edit-link").forEach((n) => n.remove());
  const userMsg = { role: "user", text, scope_source_id: scope };
  thread.append(userNode(userMsg));
  $("input").value = ""; autosize();
  await runTurn(`/api/conversations/${state.current.id}/messages`, { text, scope_source_id: scope }, userMsg,
    (reply) => { state.current.messages.push(userMsg, reply); });
}

// Shared by send and edit: show real stage progress, then the reply.
async function runTurn(path, body, userMsg, commit) {
  const thread = $("thread");
  const label = el("div", { class: "typing", text: "Working" });
  const typing = el("div", { class: "msg assistant" }, label);
  thread.append(typing);
  scrollDown();
  const finish = () => { state.busy = false; $("send").disabled = false; updateEditLink(); $("input").focus(); };
  try {
    const reply = await postStream(path, body, (s) => { label.textContent = s.label; });
    typing.remove();
    userMsg.turn_id = reply.turn_id;
    const node = assistantNode(reply);
    thread.append(node);
    commit(reply);
    if ($("animate").checked && !matchMedia("(prefers-reduced-motion: reduce)").matches) animate(node, finish);
    else { finish(); scrollDown(); }
    await refreshList();
    const row = state.convs.find((c) => c.id === state.current.id);
    if (row) { state.current.title = row.title; $("title").textContent = row.title; }
  } catch (err) {
    typing.remove();
    showError(err.message);
    finish();
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

async function initPreferences() {
  const select = $("pref-summary");
  let prefs;
  try { prefs = (await api("GET", "/api/preferences")).summary_format; } catch { return; }
  select.textContent = "";
  if (!prefs.value) select.append(el("option", { value: "", text: "Not set (numbered)", disabled: "", selected: "" }));
  prefs.options.forEach((o) => select.append(el("option", { value: o, text: o })));
  if (prefs.value) select.value = prefs.value;
  select.addEventListener("change", async () => {
    $("pref-note").textContent = "";
    try {
      await api("POST", "/api/preferences", { key: "summary_format", value: select.value });
      $("pref-note").textContent = "Saved for you.";
    } catch (e) { $("pref-note").textContent = e.message; }
  });
}

function renderConfig() {
  const c = state.config;
  const parts = [`${c.source_count} source${c.source_count === 1 ? "" : "s"}`, c.representation, c.realizer, c.abstention, `user ${c.user_id}`];
  $("config").textContent = parts.filter(Boolean).join(" · ");
}

async function init() {
  $("animate").checked = store.get("animate", "1") === "1";
  $("animate").addEventListener("change", (e) => store.set("animate", e.target.checked ? "1" : "0"));
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send($("input").value); });
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($("input").value); } });
  $("input").addEventListener("input", autosize);
  $("new-chat").addEventListener("click", newChat);
  let searchTimer;
  $("search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { searchTerm = e.target.value.trim(); refreshList().catch((err) => showError(err.message)); }, 200);
  });
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("close-panel").addEventListener("click", closePanel);
  $("scrim").addEventListener("click", () => { closePanel(); closeSidebar(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePanel(); closeSidebar(); } });
  try {
    [state.config, state.sources] = await Promise.all([api("GET", "/api/config"), api("GET", "/api/sources")]);
    renderConfig();
    initPreferences();
    await refreshList();
    if (state.convs.length) await selectConv(state.convs[0].id); else renderWelcome();
  } catch (err) { showError(err.message); }
  $("input").focus();
}
init();
