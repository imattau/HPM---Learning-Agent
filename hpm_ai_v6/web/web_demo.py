#!/usr/bin/env python3
"""
Simple Flask demo for the V6 HPM multi-agent reader.

Run with:
    python -m hpm_ai_v6.web.web_demo
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

from flask import Flask, jsonify, render_template_string, request


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent
from hpm_ai_v6.agents.web_agent import WebAgent


app = Flask(__name__)
reader: Optional[MultiAgentReader] = None
web_agent: Optional[WebAgent] = None
dataset_agent: Optional[DatasetTrainingAgent] = None
gutenberg_cycle_thread: Optional[threading.Thread] = None
gutenberg_cycle_stop_event = threading.Event()
gutenberg_cycle_state = {
    "active": False,
    "message": "Idle.",
    "book_id": None,
    "chapter": None,
    "phase": "idle",
    "books_processed": 0,
    "chapter_added": 0,
}
CORPUS_LABEL = "alice_mini.txt"


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HPM v6</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0d0f14;
      --surface: #131720;
      --surface2: #1a2030;
      --border: #1e2435;
      --border2: #263046;
      --accent: #00e5a0;
      --accent-dim: rgba(0,229,160,0.12);
      --accent-glow: rgba(0,229,160,0.25);
      --text: #e2e8f0;
      --muted: #6b7a99;
      --danger: #ff5c5c;
      --danger-dim: rgba(255,92,92,0.12);
      --success-dim: rgba(0,229,160,0.10);
      --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
      --sans: 'Inter', system-ui, sans-serif;
    }
    html { font-size: 15px; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
      line-height: 1.6;
    }

    /* ── Header ── */
    .header {
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      height: 52px;
      background: rgba(13,15,20,0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
    }
    .header-logo {
      font-family: var(--mono);
      font-size: 1rem;
      font-weight: 600;
      color: var(--accent);
      letter-spacing: 0.05em;
    }
    .header-meta {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      font-family: var(--mono);
      font-size: 0.75rem;
      padding: 3px 10px;
      border-radius: 999px;
      border: 1px solid var(--border2);
      color: var(--muted);
      background: var(--surface);
    }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 6px var(--accent-glow);
    }

    /* ── Status ribbon ── */
    .ribbon {
      display: none;
      align-items: center;
      gap: 10px;
      padding: 8px 24px;
      background: var(--accent-dim);
      border-bottom: 1px solid rgba(0,229,160,0.15);
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--accent);
    }
    .ribbon.visible { display: flex; }
    .ribbon-dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 1.4s ease-in-out infinite;
      flex-shrink: 0;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }

    /* ── Main layout ── */
    .main {
      max-width: 820px;
      margin: 0 auto;
      padding: 32px 20px 80px;
    }
    .page-title {
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 28px;
    }

    /* ── Accordion ── */
    .accordion { display: flex; flex-direction: column; gap: 10px; }
    .section {
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: var(--surface);
    }
    .section-header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 16px 20px;
      cursor: pointer;
      user-select: none;
      transition: background 0.15s;
    }
    .section-header:hover { background: var(--surface2); }
    .step-num {
      font-family: var(--mono);
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--accent);
      letter-spacing: 0.08em;
      min-width: 28px;
    }
    .section-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      flex: 1;
    }
    .section-badge {
      font-family: var(--mono);
      font-size: 0.68rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border2);
      color: var(--muted);
    }
    .chevron {
      color: var(--muted);
      font-size: 0.8rem;
      transition: transform 0.2s;
    }
    .section.open .chevron { transform: rotate(180deg); }
    .section-body {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease;
    }
    .section.open .section-body { max-height: 2000px; }
    .section-inner { padding: 0 20px 24px; }

    /* ── Sub-tabs (pills) ── */
    .pill-tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
    }
    .pill {
      font-family: var(--mono);
      font-size: 0.75rem;
      padding: 5px 14px;
      border-radius: 999px;
      border: 1px solid var(--border2);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      transition: all 0.15s;
    }
    .pill:hover { border-color: var(--accent); color: var(--accent); }
    .pill.active {
      background: var(--accent-dim);
      border-color: var(--accent);
      color: var(--accent);
    }
    .sub-panel { display: none; }
    .sub-panel.active { display: block; }

    /* ── Forms ── */
    .field { margin-bottom: 18px; }
    label {
      display: block;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    input[type="text"], input[type="number"], textarea {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border2);
      border-radius: 8px;
      padding: 10px 13px;
      font-family: var(--mono);
      font-size: 0.88rem;
      color: var(--text);
      outline: none;
      transition: border-color 0.15s;
    }
    input[type="text"]:focus, input[type="number"]:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    textarea { min-height: 110px; resize: vertical; }
    .slider-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .slider-val {
      font-family: var(--mono);
      font-size: 0.85rem;
      color: var(--accent);
      min-width: 32px;
      text-align: right;
    }
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 560px) { .form-row { grid-template-columns: 1fr; } }

    /* ── Buttons ── */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-family: var(--mono);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-primary {
      background: var(--accent);
      color: #0d0f14;
    }
    .btn-primary:hover { filter: brightness(1.1); box-shadow: 0 0 16px var(--accent-glow); }
    .btn-outline {
      background: transparent;
      border: 1px solid var(--border2);
      color: var(--muted);
    }
    .btn-outline:hover { border-color: var(--accent); color: var(--accent); }
    .btn-danger {
      background: transparent;
      border: 1px solid var(--danger);
      color: var(--danger);
    }
    .btn-danger:hover { background: var(--danger-dim); }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
    .spinner {
      display: none;
      width: 14px; height: 14px;
      border: 2px solid rgba(13,15,20,0.3);
      border-top-color: #0d0f14;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Output cards ── */
    .output-card {
      margin-top: 18px;
      background: var(--bg);
      border: 1px solid var(--border2);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      padding: 16px;
    }
    .output-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .output-label {
      font-family: var(--mono);
      font-size: 0.68rem;
      color: var(--accent);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .copy-btn {
      font-family: var(--mono);
      font-size: 0.68rem;
      padding: 3px 9px;
      border-radius: 5px;
      border: 1px solid var(--border2);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      transition: all 0.15s;
    }
    .copy-btn:hover { border-color: var(--accent); color: var(--accent); }
    .output-text {
      font-family: var(--mono);
      font-size: 0.88rem;
      color: var(--text);
      white-space: pre-wrap;
      line-height: 1.7;
    }

    /* ── Inline result cards ── */
    .result-card {
      margin-top: 18px;
      border-radius: 8px;
      padding: 12px 16px;
      font-family: var(--mono);
      font-size: 0.82rem;
      display: flex;
      align-items: flex-start;
      gap: 10px;
    }
    .result-card.success { background: var(--success-dim); border: 1px solid rgba(0,229,160,0.2); color: var(--accent); }
    .result-card.error { background: var(--danger-dim); border: 1px solid rgba(255,92,92,0.25); color: var(--danger); }
    .result-card-close {
      margin-left: auto;
      background: none;
      border: none;
      color: inherit;
      cursor: pointer;
      font-size: 1rem;
      opacity: 0.6;
      padding: 0;
    }
    .result-card-close:hover { opacity: 1; }

    /* ── Gutenberg cycle sub-section ── */
    .cycle-panel {
      margin-top: 20px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
    }
    .cycle-title {
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .cycle-desc {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 14px;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header class="header">
    <div class="header-logo">HPM v6</div>
    <div class="header-meta">
      <span class="badge">{{ corpus }}</span>
      <div class="status-dot" title="Reader ready"></div>
    </div>
  </header>

  <!-- Status ribbon (Gutenberg cycle) -->
  <div class="ribbon" id="status-ribbon">
    <div class="ribbon-dot"></div>
    <span id="ribbon-text">Idle.</span>
  </div>

  <!-- Main -->
  <main class="main">
    <div class="page-title">Multi-agent hierarchical pattern reader</div>

    <div class="accordion">

      <!-- 01 TRAIN -->
      <div class="section open" id="sec-train">
        <div class="section-header" onclick="toggleSection('sec-train')">
          <span class="step-num">01</span>
          <span class="section-title">Train</span>
          <span class="section-badge">web links &amp; gutenberg</span>
          <span class="chevron">&#9660;</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="pill-tabs">
              <button type="button" class="pill active" onclick="showSub('train', 'links', this)">Web Links</button>
              <button type="button" class="pill" onclick="showSub('train', 'gutenberg', this)">Gutenberg</button>
            </div>

            <!-- Web Links sub-panel -->
            <div class="sub-panel active" id="train-links">
              <form method="post" onsubmit="showFormResult(this)">
                <input type="hidden" name="action" value="train_links">
                <div class="field">
                  <label for="urls">URLs (one per line)</label>
                  <textarea id="urls" name="urls" placeholder="https://example.com/article">{{ urls }}</textarea>
                </div>
                <div class="field">
                  <label for="rss_url">RSS feed URL (optional)</label>
                  <input type="text" id="rss_url" name="rss_url" value="{{ rss_url }}" placeholder="https://example.com/feed.xml">
                </div>
                <div class="form-row">
                  <div class="field">
                    <label for="top_k">Top sentences</label>
                    <input type="number" id="top_k" name="top_k" value="{{ top_k }}" min="1" max="500">
                  </div>
                  <div class="field">
                    <label for="min_score">Min entropy score</label>
                    <input type="number" id="min_score" name="min_score" value="{{ min_score }}" min="0" step="0.05">
                  </div>
                </div>
                <div class="btn-row">
                  <button type="submit" class="btn btn-primary">
                    <span class="spinner" id="links-spinner"></span>
                    Train from links
                  </button>
                </div>
                {% if training_message %}
                <div class="result-card success" id="train-links-result">
                  <span>{{ training_message }}</span>
                  <button type="button" class="result-card-close" onclick="this.parentElement.remove()">&#10005;</button>
                </div>
                {% endif %}
                {% if error %}
                <div class="result-card error">
                  <span>{{ error }}</span>
                  <button type="button" class="result-card-close" onclick="this.parentElement.remove()">&#10005;</button>
                </div>
                {% endif %}
              </form>
            </div>

            <!-- Gutenberg sub-panel -->
            <div class="sub-panel" id="train-gutenberg">
              <form method="post" onsubmit="showFormResult(this)">
                <input type="hidden" name="action" value="train_gutenberg">
                <div class="field">
                  <label for="gutenberg_ids">Book IDs (comma-separated)</label>
                  <input type="text" id="gutenberg_ids" name="gutenberg_ids" value="{{ gutenberg_ids }}" placeholder="11, 84, 1342">
                </div>
                <div class="form-row">
                  <div class="field">
                    <label for="gutenberg_top_k">Top paragraphs</label>
                    <input type="number" id="gutenberg_top_k" name="gutenberg_top_k" value="{{ gutenberg_top_k }}" min="1" max="500">
                  </div>
                  <div class="field">
                    <label for="gutenberg_min_score">Min entropy score</label>
                    <input type="number" id="gutenberg_min_score" name="gutenberg_min_score" value="{{ gutenberg_min_score }}" min="0" step="0.05">
                  </div>
                </div>
                <div class="btn-row">
                  <button type="submit" class="btn btn-primary">
                    <span class="spinner"></span>
                    Train from Gutenberg
                  </button>
                </div>
              </form>

              <div class="cycle-panel">
                <div class="cycle-title">Curated cycle</div>
                <div class="cycle-desc">Runs the curated 11-book rotation continuously in the background.</div>
                <div class="btn-row">
                  <button type="button" class="btn btn-outline" onclick="submitAction('start_gutenberg_cycle', this)">
                    <span class="spinner"></span>
                    &#9654; Run curated cycle
                  </button>
                  <button type="button" class="btn btn-danger" onclick="submitAction('stop_gutenberg_cycle', this)">
                    <span class="spinner"></span>
                    &#9632; Stop
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 02 GENERATE -->
      <div class="section open" id="sec-generate">
        <div class="section-header" onclick="toggleSection('sec-generate')">
          <span class="step-num">02</span>
          <span class="section-title">Generate</span>
          <span class="section-badge">text continuation</span>
          <span class="chevron">&#9660;</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="field">
              <label for="seed">Seed phrase</label>
              <input type="text" id="seed" name="seed" value="{{ seed }}">
            </div>
            <div class="field">
              <label for="max_len_range">Max tokens &mdash; <span id="max-len-display">{{ max_len }}</span></label>
              <div class="slider-row">
                <input type="range" id="max_len_range" min="1" max="200" value="{{ max_len }}"
                  oninput="document.getElementById('max-len-display').textContent=this.value">
              </div>
            </div>
            <div class="btn-row">
              <button type="button" class="btn btn-primary" id="generate-btn" onclick="runGenerate()">
                <span class="spinner" id="gen-spinner"></span>
                Generate
              </button>
            </div>
            <div id="generate-output">
              {% if generated %}
              <div class="output-card">
                <div class="output-card-header">
                  <span class="output-label">Output</span>
                  <button class="copy-btn" onclick="copyText('gen-text')">copy</button>
                </div>
                <div class="output-text" id="gen-text">{{ generated }}</div>
              </div>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

      <!-- 03 REASON -->
      <div class="section open" id="sec-reason">
        <div class="section-header" onclick="toggleSection('sec-reason')">
          <span class="step-num">03</span>
          <span class="section-title">Reason</span>
          <span class="section-badge">question answering</span>
          <span class="chevron">&#9660;</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="field">
              <label for="question">Question</label>
              <textarea id="question" name="question" placeholder="Why did Alice follow the rabbit?">{{ question }}</textarea>
            </div>
            <div class="btn-row">
              <button type="button" class="btn btn-primary" id="reason-btn" onclick="runReason()">
                <span class="spinner" id="reason-spinner"></span>
                Reason
              </button>
            </div>
            <div id="reason-output">
              {% if reasoning %}
              <div class="output-card">
                <div class="output-card-header">
                  <span class="output-label">Reasoning</span>
                  <button class="copy-btn" onclick="copyText('reason-text')">copy</button>
                </div>
                <div class="output-text" id="reason-text">{{ reasoning }}</div>
              </div>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

    </div><!-- /accordion -->
  </main>

  <script>
    // ── Accordion ──
    function toggleSection(id) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle('open');
      try { localStorage.setItem('hpm_sec_' + id, el.classList.contains('open') ? '1' : '0'); } catch(e){}
    }
    ['sec-train','sec-generate','sec-reason'].forEach(function(id) {
      try {
        const v = localStorage.getItem('hpm_sec_' + id);
        const el = document.getElementById(id);
        if (v === '0' && el) el.classList.remove('open');
      } catch(e){}
    });

    // ── Sub-tabs ──
    function showSub(group, name, pill) {
      const parent = pill.closest('.section-inner');
      parent.querySelectorAll('.sub-panel').forEach(function(p){ p.classList.remove('active'); });
      document.getElementById(group + '-' + name).classList.add('active');
      pill.closest('.pill-tabs').querySelectorAll('.pill').forEach(function(p){ p.classList.remove('active'); });
      pill.classList.add('active');
    }

    // ── Form status ──
    function showFormResult(form) {
      const btn = form.querySelector('button[type="submit"]');
      const spinner = form.querySelector('.spinner');
      if (btn) btn.disabled = true;
      if (spinner) spinner.style.display = 'inline-block';
    }

    // ── AJAX: Generate ──
    async function runGenerate() {
      const seed = document.getElementById('seed').value;
      const max_len = parseInt(document.getElementById('max_len_range').value);
      const btn = document.getElementById('generate-btn');
      const spinner = document.getElementById('gen-spinner');
      btn.disabled = true;
      spinner.style.display = 'inline-block';
      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({seed: seed, max_len: max_len, temperature: 0})
        });
        const data = await res.json();
        const container = document.getElementById('generate-output');
        if (data.success) {
          container.innerHTML = `<div class="output-card"><div class="output-card-header"><span class="output-label">Output</span><button class="copy-btn" onclick="copyText('gen-text-ajax')">copy</button></div><div class="output-text" id="gen-text-ajax"></div></div>`;
          document.getElementById('gen-text-ajax').textContent = data.generated;
        } else {
          container.innerHTML = `<div class="result-card error"><span class="err-msg"></span><button class="result-card-close" onclick="this.parentElement.remove()">&#10005;</button></div>`;
          container.querySelector('.err-msg').textContent = data.error || 'Unknown error.';
        }
      } catch(err) {
        document.getElementById('generate-output').innerHTML = `<div class="result-card error"><span>Network error.</span></div>`;
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    // ── AJAX: Reason ──
    async function runReason() {
      const question = document.getElementById('question').value;
      const btn = document.getElementById('reason-btn');
      const spinner = document.getElementById('reason-spinner');
      btn.disabled = true;
      spinner.style.display = 'inline-block';
      try {
        const res = await fetch('/api/reason', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: question})
        });
        const data = await res.json();
        const container = document.getElementById('reason-output');
        if (data.success) {
          container.innerHTML = `<div class="output-card"><div class="output-card-header"><span class="output-label">Reasoning</span><button class="copy-btn" onclick="copyText('reason-text-ajax')">copy</button></div><div class="output-text" id="reason-text-ajax"></div></div>`;
          document.getElementById('reason-text-ajax').textContent = data.reasoning;
        } else {
          container.innerHTML = `<div class="result-card error"><span></span><button class="result-card-close" onclick="this.parentElement.remove()">&#10005;</button></div>`;
          container.querySelector('span').textContent = data.error || 'Unknown error.';
        }
      } catch(err) {
        document.getElementById('reason-output').innerHTML = `<div class="result-card error"><span>Network error.</span></div>`;
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    // ── Copy to clipboard ──
    function copyText(id) {
      const el = document.getElementById(id);
      if (!el) return;
      navigator.clipboard.writeText(el.textContent).catch(function(){});
    }

    // ── Submit hidden-action form ──
    function submitAction(action, btn) {
      if (btn) {
          btn.disabled = true;
          const spinner = btn.querySelector('.spinner');
          if (spinner) spinner.style.display = 'inline-block';
      }
      const form = document.createElement('form');
      form.method = 'post';
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = 'action';
      input.value = action;
      form.appendChild(input);
      document.body.appendChild(form);
      form.submit();
    }

    // ── Gutenberg cycle status ribbon ──
    async function refreshCycleStatus() {
      try {
        const res = await fetch('/api/gutenberg_cycle_status');
        const data = await res.json();
        const ribbon = document.getElementById('status-ribbon');
        const ribbonText = document.getElementById('ribbon-text');
        if (data.active) {
          ribbon.classList.add('visible');
          const parts = ['RUNNING'];
          if (data.book_id != null) parts.push('book ' + data.book_id);
          if (data.chapter != null) parts.push('ch ' + data.chapter);
          if (data.phase) parts.push(data.phase);
          if (data.books_processed != null) parts.push(data.books_processed + ' books');
          if (data.chapter_added != null) parts.push(data.chapter_added + ' added');
          ribbonText.textContent = parts.join(' │ ');
        } else {
          ribbon.classList.remove('visible');
        }
      } catch(e) {}
    }
    refreshCycleStatus();
    setInterval(refreshCycleStatus, 4000);
  </script>
</body>
</html>
"""


def _corpus_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "corpus", "alice_mini.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}")
    return path


def _build_reader() -> MultiAgentReader:
    local_reader = MultiAgentReader(_corpus_path(), warm_start=True)
    local_reader.train(
        episodes=1,
        max_chunks=2,
        max_words_per_chunk=32,
        enable_pruning=False,
        enable_causal=False,
    )
    return local_reader


def _build_web_agent(local_reader: MultiAgentReader) -> WebAgent:
    return WebAgent(local_reader, corpus_path=_corpus_path(), min_sentence_len=20)


def _build_dataset_agent(local_reader: MultiAgentReader) -> DatasetTrainingAgent:
    return DatasetTrainingAgent(local_reader, corpus_path=_corpus_path(), min_sentence_len=20)


def _start_gutenberg_cycle() -> str:
    global gutenberg_cycle_thread
    if dataset_agent is None:
        return "Dataset agent is not ready."
    if gutenberg_cycle_state["active"]:
        return "Curated Gutenberg cycle is already running."

    gutenberg_cycle_stop_event.clear()
    gutenberg_cycle_state["active"] = True
    gutenberg_cycle_state["message"] = "Running curated Gutenberg cycle in the background."

    def _worker() -> None:
        try:
            reports = dataset_agent.train_gutenberg_cycle(
                book_ids=None,
                top_k_per_chapter=8,
                min_score=0.05,
                retrain_epochs=1,
                split_on_paragraphs=False,
                stop_event=gutenberg_cycle_stop_event,
                repeat_books=True,
                report_progress=True,
                progress_callback=_update_gutenberg_cycle_state,
                maintenance_callback=_update_gutenberg_cycle_state,
            )
            final_books = len(reports)
            gutenberg_cycle_state["message"] = (
                f"Curated Gutenberg cycle stopped after {final_books} books."
            )
        except Exception as exc:  # pragma: no cover - defensive demo path
            gutenberg_cycle_state["message"] = f"Curated Gutenberg cycle failed: {exc}"
        finally:
            gutenberg_cycle_state["active"] = False

    gutenberg_cycle_thread = threading.Thread(target=_worker, daemon=True)
    gutenberg_cycle_thread.start()
    return gutenberg_cycle_state["message"]


def _stop_gutenberg_cycle() -> str:
    if not gutenberg_cycle_state["active"]:
        return "Curated Gutenberg cycle is not running."
    gutenberg_cycle_stop_event.set()
    gutenberg_cycle_state["phase"] = "stopping"
    return "Stopping curated Gutenberg cycle."


def _update_gutenberg_cycle_state(event: dict) -> None:
    event_type = event.get("type", "unknown")
    if event_type == "book_start":
        gutenberg_cycle_state["book_id"] = event.get("book_id")
        gutenberg_cycle_state["chapter"] = None
        gutenberg_cycle_state["phase"] = "book_start"
        gutenberg_cycle_state["books_processed"] = event.get("books_processed", gutenberg_cycle_state["books_processed"])
    elif event_type == "chapter_start":
        gutenberg_cycle_state["book_id"] = event.get("book_id")
        gutenberg_cycle_state["chapter"] = event.get("chapter")
        gutenberg_cycle_state["phase"] = "chapter_start"
    elif event_type == "chapter_skip":
        gutenberg_cycle_state["phase"] = "chapter_skip"
    elif event_type == "chapter_done":
        gutenberg_cycle_state["book_id"] = event.get("book_id", gutenberg_cycle_state["book_id"])
        gutenberg_cycle_state["chapter"] = event.get("chapter", gutenberg_cycle_state["chapter"])
        gutenberg_cycle_state["phase"] = "chapter_done"
        gutenberg_cycle_state["chapter_added"] = event.get("added", 0)
    elif event_type == "book_done":
        gutenberg_cycle_state["book_id"] = event.get("book_id", gutenberg_cycle_state["book_id"])
        gutenberg_cycle_state["chapter"] = None
        gutenberg_cycle_state["phase"] = "book_done"
        gutenberg_cycle_state["books_processed"] = event.get("books_processed", gutenberg_cycle_state["books_processed"])
        gutenberg_cycle_state["chapter_added"] = 0
        maintenance_report = event.get("report", {})
        summary = maintenance_report.get("_summary", {}) if isinstance(maintenance_report, dict) else {}
        improved = ", ".join(summary.get("improved_agents", [])) or "none"
        loaded = summary.get("loaded", 0)
        if loaded:
            gutenberg_cycle_state["message"] = (
                f"Processed book {gutenberg_cycle_state['book_id']} with maintenance loaded {loaded} patterns; "
                f"improved agents: {improved}."
            )
    elif event_type == "maintenance_done":
        gutenberg_cycle_state["book_id"] = event.get("book_id", gutenberg_cycle_state["book_id"])
        gutenberg_cycle_state["phase"] = "maintenance_done"
        maintenance_report = event.get("report", {})
        summary = maintenance_report.get("_summary", {}) if isinstance(maintenance_report, dict) else {}
        improved = ", ".join(summary.get("improved_agents", [])) or "none"
        loaded = summary.get("loaded", 0)
        gutenberg_cycle_state["message"] = (
            f"Processed book {gutenberg_cycle_state['book_id']} with maintenance loaded {loaded} patterns; "
            f"improved agents: {improved}."
        )
    elif event_type == "cycle_done":
        gutenberg_cycle_state["phase"] = "cycle_done"
        gutenberg_cycle_state["books_processed"] = event.get("books_processed", gutenberg_cycle_state["books_processed"])
        gutenberg_cycle_state["chapter_added"] = 0


@app.route("/", methods=["GET", "POST"])
def index():
    seed = "Alice was"
    max_len = 50
    urls = ""
    rss_url = ""
    top_k = 20
    min_score = 0.5
    gutenberg_ids = ", ".join(str(book_id) for book_id in DatasetTrainingAgent.curated_gutenberg_book_ids())
    gutenberg_top_k = 12
    gutenberg_min_score = 0.05
    generated = None
    error = None
    training_message = None
    question = "Why did Alice follow the rabbit?"
    reasoning = None

    if request.method == "POST":
        action = request.form.get("action", "generate")
        seed = request.form.get("seed", seed)
        max_len = int(request.form.get("max_len", max_len))
        urls = request.form.get("urls", urls)
        rss_url = request.form.get("rss_url", rss_url)
        top_k = int(request.form.get("top_k", top_k))
        min_score = float(request.form.get("min_score", min_score))
        gutenberg_ids = request.form.get("gutenberg_ids", gutenberg_ids)
        gutenberg_top_k = int(request.form.get("gutenberg_top_k", gutenberg_top_k))
        gutenberg_min_score = float(request.form.get("gutenberg_min_score", gutenberg_min_score))
        question = request.form.get("question", question)

        try:
            if action == "train_links":
                added = 0
                parsed_urls = [line.strip() for line in urls.splitlines() if line.strip()]
                if parsed_urls:
                    added += web_agent.add_informative_sentences(parsed_urls, top_k=top_k, min_score=min_score) if web_agent is not None else 0
                if rss_url.strip():
                    added += web_agent.process_rss_feed(rss_url.strip(), top_k=top_k, min_score=min_score) if web_agent is not None else 0
                training_message = f"Added {added} informative sentences and retrained the reader."
            elif action == "train_gutenberg":
                added = 0
                parsed_ids = [int(part.strip()) for part in gutenberg_ids.split(",") if part.strip()]
                if parsed_ids and dataset_agent is not None:
                    added = dataset_agent.add_from_gutenberg(
                        book_ids=parsed_ids,
                        top_k=gutenberg_top_k,
                        min_score=gutenberg_min_score,
                    )
                training_message = f"Added {added} Gutenberg paragraphs and retrained the reader."
            elif action == "start_gutenberg_cycle":
                training_message = _start_gutenberg_cycle()
            elif action == "stop_gutenberg_cycle":
                training_message = _stop_gutenberg_cycle()
            elif action == "reason":
                reasoning = reader.reason(question) if reader is not None else "Reader not ready."
            else:
                generated = reader.generate(seed, max_length=max_len) if reader is not None else None
        except Exception as exc:  # pragma: no cover - defensive demo path
            error = str(exc)

    cycle_message = gutenberg_cycle_state["message"]

    return render_template_string(
        HTML_TEMPLATE,
        seed=seed,
        max_len=max_len,
        urls=urls,
        rss_url=rss_url,
        top_k=top_k,
        min_score=min_score,
        gutenberg_ids=gutenberg_ids,
        gutenberg_top_k=gutenberg_top_k,
        gutenberg_min_score=gutenberg_min_score,
        generated=generated,
        error=error,
        training_message=training_message,
        question=question,
        reasoning=reasoning,
        cycle_message=cycle_message,
        corpus=CORPUS_LABEL,
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    seed = payload.get("seed", "Alice was")
    max_len = int(payload.get("max_len", 50))
    temperature = float(payload.get("temperature", 0.0))

    if reader is None:
        return jsonify({"success": False, "error": "Reader not ready — start the server via main()."}), 503
    try:
        generated = reader.generate(seed, max_length=max_len, temperature=temperature)
        return jsonify({"success": True, "generated": generated})
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(silent=True) or {}
    urls = payload.get("urls", [])
    rss_url = payload.get("rss_url", "")
    top_k = int(payload.get("top_k", 20))
    min_score = float(payload.get("min_score", 0.5))
    gutenberg_ids = payload.get("gutenberg_ids", [])
    gutenberg_top_k = int(payload.get("gutenberg_top_k", top_k))
    gutenberg_min_score = float(payload.get("gutenberg_min_score", min_score))
    cycle = bool(payload.get("gutenberg_cycle", False))

    try:
        added = 0
        if urls:
            added += web_agent.add_informative_sentences(urls, top_k=top_k, min_score=min_score) if web_agent is not None else 0
        if rss_url:
            added += web_agent.process_rss_feed(rss_url, top_k=top_k, min_score=min_score) if web_agent is not None else 0
        if gutenberg_ids and dataset_agent is not None:
            parsed_ids = [int(book_id) for book_id in gutenberg_ids]
            if cycle:
                added += len(dataset_agent.train_gutenberg_cycle(
                    book_ids=parsed_ids,
                    top_k_per_chapter=gutenberg_top_k,
                    min_score=gutenberg_min_score,
                    repeat_books=False,
                    report_progress=True,
                    max_books=len(parsed_ids),
                ))
            else:
                added += dataset_agent.add_from_gutenberg(
                    book_ids=parsed_ids,
                    top_k=gutenberg_top_k,
                    min_score=gutenberg_min_score,
                )
        return jsonify({"success": True, "added": added})
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reason", methods=["POST"])
def api_reason():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "Why did Alice follow the rabbit?")
    if reader is None:
        return jsonify({"success": False, "error": "Reader not ready — start the server via main()."}), 503
    try:
        answer = reader.reason(question)
        return jsonify({"success": True, "reasoning": answer})
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/gutenberg_cycle_status", methods=["GET"])
def api_gutenberg_cycle_status():
    return jsonify({
        "success": True,
        "active": gutenberg_cycle_state["active"],
        "message": gutenberg_cycle_state["message"],
        "book_id": gutenberg_cycle_state["book_id"],
        "chapter": gutenberg_cycle_state["chapter"],
        "phase": gutenberg_cycle_state["phase"],
        "books_processed": gutenberg_cycle_state["books_processed"],
        "chapter_added": gutenberg_cycle_state["chapter_added"],
    })


def main():
    global reader, web_agent, dataset_agent
    print("Loading MultiAgentReader...")
    reader = _build_reader()
    web_agent = _build_web_agent(reader)
    dataset_agent = _build_dataset_agent(reader)
    print("Reader ready.")
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
