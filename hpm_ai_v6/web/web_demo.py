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

from flask import Flask, jsonify, render_template_string, request, send_file


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent
from hpm_ai_v6.agents.web_agent import WebAgent
from hpm_ai_v6.agents.quiz_agent import QuizAgent as _QuizAgent


app = Flask(__name__)
reader: Optional[MultiAgentReader] = None
web_agent: Optional[WebAgent] = None
dataset_agent: Optional[DatasetTrainingAgent] = None
_quiz_agent: Optional[_QuizAgent] = None
_quiz_state: dict = {}  # keyed by question id -> QuizQuestion
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
wikipedia_cycle_thread: Optional[threading.Thread] = None
wikipedia_cycle_stop_event = threading.Event()
wikipedia_cycle_state = {
    "active": False,
    "message": "Idle.",
    "topic": None,
    "page_title": None,
    "phase": "idle",
    "topics_processed": 0,
    "pages_added": 0,
}
wikipedia_topics_state = {
    "topics_text": "",
    "topics": [],
    "message": "No Wikipedia topics generated yet.",
}
CORPUS_LABEL = "alice_mini.txt"


def _build_reason_trace(question: str) -> dict:
    if reader is None:
        return {
            "question": question,
            "intent": "unavailable",
            "mode": "unavailable",
            "method": "auto",
            "answer": "Reader not ready.",
        }
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is not None:
        return reasoning_agent.reason_with_trace(question)
    return {
        "question": question,
        "intent": "path",
        "mode": "default",
        "method": "auto",
        "answer": reader.reason(question),
    }


def _reasoning_index_status() -> dict:
    if reader is None:
        return {"state": "unavailable", "dirty": False}
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None or not hasattr(reasoning_agent, "index_status"):
        return {"state": "unavailable", "dirty": False}
    try:
        return reasoning_agent.index_status()
    except Exception:
        return {"state": "unknown", "dirty": False}


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
    .section-badge.state-ready { color: var(--accent); border-color: rgba(0,229,160,0.35); background: var(--success-dim); }
    .section-badge.state-indexing { color: #ffd166; border-color: rgba(255,209,102,0.35); background: rgba(255,209,102,0.10); }
    .section-badge.state-dirty { color: #ffb84d; border-color: rgba(255,184,77,0.35); background: rgba(255,184,77,0.10); }
    .section-badge.state-error { color: var(--danger); border-color: rgba(255,92,92,0.35); background: var(--danger-dim); }
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
    .output-meta {
      margin-top: 10px;
      font-family: var(--mono);
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.02em;
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

    /* ── Analysis sidebar ── */
    .sidebar-toggle {
      font-family: var(--mono);
      font-size: 0.75rem;
      padding: 5px 12px;
      border-radius: 6px;
      border: 1px solid var(--border2);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      transition: all 0.15s;
    }
    .sidebar-toggle:hover { border-color: var(--accent); color: var(--accent); }
    .sidebar-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.4);
      z-index: 200;
    }
    .sidebar-overlay.open { display: block; }
    .sidebar {
      position: fixed;
      top: 0; right: -420px;
      width: 420px;
      height: 100vh;
      background: var(--surface);
      border-left: 1px solid var(--border);
      z-index: 201;
      display: flex;
      flex-direction: column;
      transition: right 0.3s ease;
      overflow: hidden;
    }
    .sidebar.open { right: 0; }
    .sidebar-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .sidebar-title {
      font-family: var(--mono);
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent);
    }
    .sidebar-close {
      background: none;
      border: none;
      color: var(--muted);
      cursor: pointer;
      font-size: 1.1rem;
      padding: 0;
    }
    .sidebar-close:hover { color: var(--text); }
    .sidebar-body {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    }
    .sidebar-actions {
      padding: 16px 20px;
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }
    .analysis-section {
      margin-bottom: 24px;
    }
    .analysis-section-title {
      font-family: var(--mono);
      font-size: 0.68rem;
      color: var(--accent);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }
    .analysis-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 4px 0;
      font-family: var(--mono);
      font-size: 0.8rem;
    }
    .analysis-label { color: var(--muted); }
    .analysis-value { color: var(--text); font-weight: 600; }
    .analysis-value.accent { color: var(--accent); }
    .hub-entry {
      padding: 6px 0;
      border-bottom: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 0.78rem;
    }
    .hub-name { color: var(--text); margin-bottom: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .hub-meta { color: var(--muted); font-size: 0.7rem; }
    .analysis-timestamp {
      font-family: var(--mono);
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 16px;
      text-align: center;
    }
    .analysis-empty {
      font-family: var(--mono);
      font-size: 0.82rem;
      color: var(--muted);
      text-align: center;
      padding: 40px 0;
    }

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
    .cycle-status {
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--accent);
      margin-top: 10px;
      min-height: 1.25em;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header class="header">
    <div class="header-logo">HPM v6</div>
    <div class="header-meta">
      <span class="badge">{{ corpus }}</span>
      <button class="sidebar-toggle" onclick="toggleSidebar()">&#9782; Analyse</button>
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
          <span class="section-badge">web links, gutenberg &amp; wikipedia</span>
          <span class="chevron">&#9660;</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="pill-tabs">
              <button type="button" class="pill active" onclick="return showSub('train', 'links', this, event)">Web Links</button>
              <button type="button" class="pill" onclick="return showSub('train', 'gutenberg', this, event)">Gutenberg</button>
              <button type="button" class="pill" onclick="return showSub('train', 'wikipedia', this, event)">Wikipedia</button>
            </div>

            <!-- Web Links sub-panel -->
            <div class="sub-panel active" id="train-links">
              <form method="post" onsubmit="showFormResult(this, event)">
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
              <form method="post" onsubmit="showFormResult(this, event)">
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
                <div class="cycle-desc">Reads Gutenberg sequentially from book 1, chapter by chapter. Resumes from last checkpoint.</div>
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

            <!-- Wikipedia sub-panel -->
            <div class="sub-panel" id="train-wikipedia">
              <form method="post" onsubmit="showFormResult(this, event)">
                <input type="hidden" name="action" value="generate_wikipedia_topics" id="wikipedia-action">
                <div class="field">
                  <label for="wikipedia_topics">Topics</label>
                  <textarea id="wikipedia_topics" name="wikipedia_topics" placeholder="Generate topics from learned patterns and edit before starting.">{{ wikipedia_topics }}</textarea>
                </div>
                <div class="form-row">
                  <div class="field">
                    <label for="wikipedia_top_k">Top sentences</label>
                    <input type="number" id="wikipedia_top_k" name="wikipedia_top_k" value="{{ wikipedia_top_k }}" min="1" max="500">
                  </div>
                  <div class="field">
                    <label for="wikipedia_min_score">Min entropy score</label>
                    <input type="number" id="wikipedia_min_score" name="wikipedia_min_score" value="{{ wikipedia_min_score }}" min="0" step="0.05">
                  </div>
                </div>
                <div class="btn-row">
                  <button type="button" class="btn btn-outline" onclick="submitWikipediaAction('generate_wikipedia_topics', this)">
                    <span class="spinner"></span>
                    Generate topics
                  </button>
                  <button type="button" class="btn btn-primary" onclick="submitWikipediaAction('start_wikipedia_cycle', this)">
                    <span class="spinner"></span>
                    &#9654; Start Wikipedia cycle
                  </button>
                  <button type="button" class="btn btn-danger" onclick="submitWikipediaAction('stop_wikipedia_cycle', this)">
                    <span class="spinner"></span>
                    &#9632; Stop
                  </button>
                </div>
                <div class="cycle-status" id="wikipedia-cycle-status">{{ wikipedia_cycle_message }}</div>
              </form>
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
          <span class="section-badge" id="reasoning-index-badge">question answering</span>
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
                {% if reasoning_trace %}
                <div class="output-meta">
                  intent={{ reasoning_trace.intent }} | mode={{ reasoning_trace.mode }} | method={{ reasoning_trace.method }}
                  {% if reasoning_trace.explanation_method %}
                  | explanation={{ reasoning_trace.explanation_method }}
                  {% endif %}
                </div>
                {% endif %}
              </div>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

      <div class="section open" id="sec-quiz">
        <div class="section-header" onclick="toggleSection('sec-quiz')">
          <span class="step-num">04</span>
          <span class="section-title">Quiz</span>
          <span class="section-badge" id="quiz-badge">multi-choice</span>
          <span class="chevron">&#9660;</span>
        </div>
        <div class="section-body">
          <div class="section-inner">
            <div class="pill-tabs">
              <button class="pill active" onclick="showTab('quiz','take')">Take Quiz</button>
              <button class="pill" onclick="showTab('quiz','download')">Download Banks</button>
            </div>

            <div id="quiz-take" class="sub-panel active">
              <div id="quiz-setup">
                <div style="margin-bottom:12px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Source</label>
                  <label><input type="radio" name="quiz-source" value="bank" checked> Question Bank</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-source" value="model"> Model-generated</label>
                </div>
                <div style="margin-bottom:12px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Difficulty</label>
                  <label><input type="radio" name="quiz-diff" value="easy" checked> Easy</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-diff" value="medium"> Medium</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-diff" value="hard"> Hard</label>
                </div>
                <div style="margin-bottom:16px">
                  <label style="font-weight:600;display:block;margin-bottom:6px">Questions</label>
                  <label><input type="radio" name="quiz-n" value="5" checked> 5</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-n" value="10"> 10</label>
                  &nbsp;&nbsp;
                  <label><input type="radio" name="quiz-n" value="20"> 20</label>
                </div>
                <button class="btn" onclick="startQuiz()">Start Quiz</button>
              </div>

              <div id="quiz-play" style="display:none">
                <div id="quiz-progress" style="margin-bottom:12px;font-size:13px;color:var(--muted)"></div>
                <div style="height:4px;background:var(--surface2);border-radius:2px;margin-bottom:20px">
                  <div id="quiz-progress-fill" style="height:4px;background:var(--accent);border-radius:2px;width:0%;transition:width 0.3s"></div>
                </div>
                <div id="quiz-question" style="font-size:16px;font-weight:600;margin-bottom:16px"></div>
                <div id="quiz-options" style="display:flex;flex-direction:column;gap:8px"></div>
                <div id="quiz-feedback" style="display:none;margin-top:16px;padding:12px;border-radius:6px;font-size:14px"></div>
                <button id="quiz-next-btn" class="btn" style="display:none;margin-top:16px" onclick="nextQuestion()">Next &rarr;</button>
              </div>

              <div id="quiz-complete" style="display:none">
                <h3 style="margin-bottom:12px">Quiz Complete</h3>
                <div id="quiz-score" style="font-size:24px;font-weight:700;color:var(--accent);margin-bottom:16px"></div>
                <div id="quiz-gap-list" style="margin-bottom:16px"></div>
                <button id="train-gaps-btn" class="btn" style="display:none" onclick="trainGaps()">Train on gaps</button>
                <div id="train-gaps-status" style="margin-top:10px;font-size:13px;color:var(--muted)"></div>
              </div>
            </div>

            <div id="quiz-download" class="sub-panel" style="display:none">
              <table style="width:100%;border-collapse:collapse">
                <tr style="border-bottom:1px solid var(--border)">
                  <td style="padding:12px 0"><strong>Easy</strong> &mdash; Single-concept factual recall</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/easy" download class="btn">Download</a></td>
                </tr>
                <tr style="border-bottom:1px solid var(--border)">
                  <td style="padding:12px 0"><strong>Medium</strong> &mdash; Concept application and moderate inference</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/medium" download class="btn">Download</a></td>
                </tr>
                <tr>
                  <td style="padding:12px 0"><strong>Hard</strong> &mdash; Relational reasoning across multiple concepts</td>
                  <td style="text-align:right"><a href="/api/quiz/banks/hard" download class="btn">Download</a></td>
                </tr>
              </table>
            </div>
          </div>
        </div>
      </div>

    </div><!-- /accordion -->
  </main>

  <div class="sidebar-overlay" id="sidebar-overlay" onclick="toggleSidebar()"></div>
  <div class="sidebar" id="analysis-sidebar">
    <div class="sidebar-header">
      <span class="sidebar-title">Pattern Analysis</span>
      <button class="sidebar-close" onclick="toggleSidebar()">&#10005;</button>
    </div>
    <div class="sidebar-body" id="analysis-body">
      <div class="analysis-empty">Click "Generate Snapshot" to analyse learned patterns.</div>
    </div>
    <div class="sidebar-actions">
      <button type="button" class="btn btn-primary" id="analyse-btn" onclick="runAnalysis()" style="width:100%">
        <span class="spinner" id="analyse-spinner"></span>
        Generate Snapshot
      </button>
    </div>
  </div>

  <script>
    // ── Accordion ──
    function toggleSection(id) {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.toggle('open');
      try { localStorage.setItem('hpm_sec_' + id, el.classList.contains('open') ? '1' : '0'); } catch(e){}
    }
    ['sec-train','sec-generate','sec-reason','sec-quiz'].forEach(function(id) {
      try {
        const v = localStorage.getItem('hpm_sec_' + id);
        const el = document.getElementById(id);
        if (v === '0' && el) el.classList.remove('open');
      } catch(e){}
    });
    try {
      const savedTrain = localStorage.getItem('hpm_train_train');
      const trainSection = document.getElementById('sec-train');
      if (savedTrain && trainSection) {
        const target = document.getElementById('train-' + savedTrain);
        if (target) {
          trainSection.querySelectorAll('.sub-panel').forEach(function(p){ p.classList.remove('active'); });
          target.classList.add('active');
          trainSection.querySelectorAll('.pill').forEach(function(p){ p.classList.remove('active'); });
          const activePill = Array.from(trainSection.querySelectorAll('.pill')).find(function(p){
            return (p.textContent || '').trim().toLowerCase().indexOf(savedTrain) === 0 ||
              (savedTrain === 'links' && (p.textContent || '').trim().toLowerCase().indexOf('web links') === 0);
          });
          if (activePill) activePill.classList.add('active');
        }
      }
    } catch(e) {}

    renderReasoningStatus({{ reasoning_index_status | tojson }});
    refreshReasoningStatus();
    renderWikipediaStatus({{ wikipedia_cycle_state | tojson }});
    refreshWikipediaStatus();

    // ── Sub-tabs ──
    function showSub(group, name, pill, event) {
      if (event && event.preventDefault) event.preventDefault();
      const section = document.getElementById('sec-' + group);
      if (!section) return false;
      section.querySelectorAll('.sub-panel').forEach(function(p){ p.classList.remove('active'); });
      const target = document.getElementById(group + '-' + name);
      if (target) target.classList.add('active');
      section.querySelectorAll('.pill').forEach(function(p){ p.classList.remove('active'); });
      if (pill) pill.classList.add('active');
      try { localStorage.setItem('hpm_train_' + group, name); } catch(e){}
      return false;
    }

    // ── Form status ──
    function showFormResult(form, event) {
      const btn = event && event.submitter ? event.submitter : form.querySelector('button[type="submit"]');
      const spinner = btn ? btn.querySelector('.spinner') : form.querySelector('.spinner');
      if (btn) btn.disabled = true;
      if (spinner) spinner.style.display = 'inline-block';
    }

    // ── Reasoning index status ──
    let reasoningStatusTimer = null;
    function renderReasoningStatus(status) {
      const badge = document.getElementById('reasoning-index-badge');
      if (!badge || !status) return;
      const state = status.state || 'unknown';
      badge.className = 'section-badge state-' + state;
      if (state === 'indexing') {
        badge.textContent = 'indexing reasoning graph';
      } else if (state === 'ready') {
        badge.textContent = 'reasoning ready';
      } else if (state === 'dirty') {
        badge.textContent = 'reasoning stale';
      } else if (state === 'error') {
        badge.textContent = 'reasoning error';
      } else {
        badge.textContent = 'question answering';
      }
    }

    async function refreshReasoningStatus() {
      try {
        const res = await fetch('/api/reasoning_index_status', { method: 'GET' });
        const data = await res.json();
        renderReasoningStatus(data);
        if (reasoningStatusTimer) clearTimeout(reasoningStatusTimer);
        if (data && data.state === 'indexing') {
          reasoningStatusTimer = setTimeout(refreshReasoningStatus, 2000);
        }
      } catch(e) {}
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
          container.innerHTML = `<div class="output-card"><div class="output-card-header"><span class="output-label">Reasoning</span><button class="copy-btn" onclick="copyText('reason-text-ajax')">copy</button></div><div class="output-text" id="reason-text-ajax"></div><div class="output-meta" id="reason-meta-ajax"></div></div>`;
          document.getElementById('reason-text-ajax').textContent = data.reasoning;
          const meta = document.getElementById('reason-meta-ajax');
          const trace = data.trace || {};
          const parts = [];
          if (trace.intent) parts.push('intent=' + trace.intent);
          if (trace.mode) parts.push('mode=' + trace.mode);
          if (trace.method) parts.push('method=' + trace.method);
          if (trace.explanation_method) parts.push('explanation=' + trace.explanation_method);
          meta.textContent = parts.join(' | ');
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

    function submitWikipediaAction(action, btn) {
      const form = btn ? btn.closest('form') : null;
      const hidden = document.getElementById('wikipedia-action');
      if (hidden) hidden.value = action;
      if (btn) {
        btn.disabled = true;
        const spinner = btn.querySelector('.spinner');
        if (spinner) spinner.style.display = 'inline-block';
      }
      if (form) {
        form.submit();
      }
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
    let cycleStatusTimer = null;

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
          if (cycleStatusTimer) {
            clearTimeout(cycleStatusTimer);
            cycleStatusTimer = null;
          }
        }
        if (data.active) {
          if (cycleStatusTimer) clearTimeout(cycleStatusTimer);
          cycleStatusTimer = setTimeout(refreshCycleStatus, 4000);
        }
      } catch(e) {}
    }
    if ({{ 'true' if cycle_active else 'false' }}) {
      refreshCycleStatus();
    }

    // ── Wikipedia cycle status ──
    let wikipediaStatusTimer = null;

    function renderWikipediaStatus(status) {
      const statusEl = document.getElementById('wikipedia-cycle-status');
      const topicsEl = document.getElementById('wikipedia_topics');
      if (topicsEl && status && Array.isArray(status.topics) && !topicsEl.value.trim()) {
        topicsEl.value = status.topics.join('\\n');
      }
      if (!statusEl || !status) return;
      const parts = [];
      const state = status.phase || status.state || 'idle';
      if (status.message) {
        parts.push(status.message);
      } else if (state === 'running' || status.active) {
        parts.push('Wikipedia cycle running.');
      } else {
        parts.push('Wikipedia cycle idle.');
      }
      if (status.topic) parts.push('topic=' + status.topic);
      if (status.page_title) parts.push('page=' + status.page_title);
      if (status.topics_processed != null) parts.push('topics=' + status.topics_processed);
      if (status.pages_added != null) parts.push('added=' + status.pages_added);
      statusEl.textContent = parts.join(' | ');
    }

    async function refreshWikipediaStatus() {
      try {
        const res = await fetch('/api/wikipedia_cycle_status', { method: 'GET' });
        const data = await res.json();
        renderWikipediaStatus(data);
        if (wikipediaStatusTimer) clearTimeout(wikipediaStatusTimer);
        if (data && data.active) {
          wikipediaStatusTimer = setTimeout(refreshWikipediaStatus, 2000);
        }
      } catch(e) {}
    }

    // ── Quiz ──
    function showTab(group, name) {
      const section = document.getElementById('sec-' + group);
      if (!section) return;
      section.querySelectorAll('.sub-panel').forEach(function(p){ p.classList.remove('active'); });
      const target = document.getElementById(group + '-' + name);
      if (target) target.classList.add('active');
      section.querySelectorAll('.pill').forEach(function(p){ p.classList.remove('active'); });
      // find the pill that was clicked by matching the onclick call
      section.querySelectorAll('.pill').forEach(function(p) {
        if (p.getAttribute('onclick').indexOf("'" + name + "'") !== -1) {
          p.classList.add('active');
        }
      });
    }

    var _quizQuestions = [];
    var _quizIndex = 0;
    var _quizCorrect = 0;
    var _quizFailedTopics = [];

    async function startQuiz() {
      var source = document.querySelector('input[name="quiz-source"]:checked').value;
      var diff   = document.querySelector('input[name="quiz-diff"]:checked').value;
      var n      = parseInt(document.querySelector('input[name="quiz-n"]:checked').value);
      document.getElementById('quiz-badge').textContent = 'loading…';
      var res = await fetch('/api/quiz/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({n: n, difficulty: diff, source: source})
      });
      var data = await res.json();
      if (data.error) { alert(data.error); return; }
      _quizQuestions = data.questions;
      _quizIndex = 0; _quizCorrect = 0; _quizFailedTopics = [];
      document.getElementById('quiz-setup').style.display = 'none';
      document.getElementById('quiz-complete').style.display = 'none';
      document.getElementById('quiz-play').style.display = 'block';
      document.getElementById('quiz-badge').textContent = diff + ' · ' + source;
      showQuestion();
    }

    function showQuestion() {
      var q = _quizQuestions[_quizIndex];
      var total = _quizQuestions.length;
      document.getElementById('quiz-progress').textContent = 'Question ' + (_quizIndex + 1) + ' of ' + total;
      document.getElementById('quiz-progress-fill').style.width = Math.round((_quizIndex / total) * 100) + '%';
      document.getElementById('quiz-question').textContent = q.question;
      document.getElementById('quiz-feedback').style.display = 'none';
      document.getElementById('quiz-next-btn').style.display = 'none';
      var labels = ['A', 'B', 'C', 'D'];
      var optDiv = document.getElementById('quiz-options');
      optDiv.innerHTML = '';
      q.options.forEach(function(opt, i) {
        var btn = document.createElement('button');
        btn.className = 'btn';
        btn.style.textAlign = 'left';
        btn.style.width = '100%';
        btn.textContent = labels[i] + '. ' + opt;
        btn.onclick = (function(idx){ return function(){ submitAnswer(idx); }; })(i);
        optDiv.appendChild(btn);
      });
    }

    async function submitAnswer(answerIndex) {
      var q = _quizQuestions[_quizIndex];
      document.getElementById('quiz-options').querySelectorAll('button').forEach(function(b){ b.disabled = true; });
      var res = await fetch('/api/quiz/submit', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question_id: q.id, answer_index: answerIndex})
      });
      var data = await res.json();
      var fb = document.getElementById('quiz-feedback');
      var labels = ['A', 'B', 'C', 'D'];
      fb.style.display = 'block';
      if (data.correct) {
        _quizCorrect++;
        fb.style.background = 'var(--success-dim)';
        fb.style.color = 'var(--accent)';
        fb.innerHTML = '✓ Correct! ' + data.explanation;
      } else {
        _quizFailedTopics.push(q.topic);
        fb.style.background = 'var(--danger-dim)';
        fb.style.color = 'var(--danger)';
        fb.innerHTML = '✗ Incorrect. Correct answer: ' + labels[data.correct_index] + '. ' + data.explanation;
      }
      document.getElementById('quiz-next-btn').style.display = 'inline-block';
    }

    function nextQuestion() {
      _quizIndex++;
      if (_quizIndex >= _quizQuestions.length) { showCompletion(); } else { showQuestion(); }
    }

    function showCompletion() {
      document.getElementById('quiz-play').style.display = 'none';
      document.getElementById('quiz-complete').style.display = 'block';
      document.getElementById('quiz-progress-fill').style.width = '100%';
      document.getElementById('quiz-score').textContent = _quizCorrect + ' / ' + _quizQuestions.length + ' correct';
      var uniqueTopics = _quizFailedTopics.filter(function(t, i, a){ return a.indexOf(t) === i; });
      var gapDiv = document.getElementById('quiz-gap-list');
      var trainBtn = document.getElementById('train-gaps-btn');
      if (uniqueTopics.length > 0) {
        gapDiv.innerHTML = '<p style="color:var(--muted);margin-bottom:8px">Weak topics:</p>' +
          uniqueTopics.map(function(t){ return '<span style="display:inline-block;padding:3px 10px;margin:3px;border-radius:4px;background:var(--surface2);font-size:13px">' + t + '</span>'; }).join('');
        trainBtn.style.display = 'inline-block';
        trainBtn.dataset.topics = JSON.stringify(uniqueTopics);
      } else {
        gapDiv.innerHTML = '<p style="color:var(--accent)">Perfect score — no gaps!</p>';
      }
    }

    async function trainGaps() {
      var btn = document.getElementById('train-gaps-btn');
      var topics = JSON.parse(btn.dataset.topics);
      btn.disabled = true;
      document.getElementById('train-gaps-status').textContent = 'Starting training…';
      var res = await fetch('/api/quiz/train_gaps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({topics: topics})
      });
      var data = await res.json();
      document.getElementById('train-gaps-status').textContent = data.message || 'Training started.';
    }

    async function generateWikipediaTopics(btn) {
      const spinner = document.getElementById('wiki-topics-spinner');
      if (btn) btn.disabled = true;
      if (spinner) spinner.style.display = 'inline-block';
      try {
        const res = await fetch('/api/wikipedia_topics', { method: 'GET' });
        const data = await res.json();
        const topicsEl = document.getElementById('wikipedia_topics');
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (data.success) {
          const topics = data.topics || [];
          if (topicsEl) topicsEl.value = topics.join('\\n');
          if (statusEl) {
            statusEl.textContent = topics.length
              ? 'Generated ' + topics.length + ' topic(s).'
              : 'No topics generated yet.';
          }
        } else if (statusEl) {
          statusEl.textContent = data.error || 'Could not generate topics.';
        }
      } catch (err) {
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (statusEl) statusEl.textContent = 'Network error while generating topics.';
      } finally {
        if (btn) btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
      }
    }

    async function startWikipediaCycle(btn) {
      const spinner = document.getElementById('wiki-start-spinner');
      const topicsEl = document.getElementById('wikipedia_topics');
      const topKEl = document.getElementById('wikipedia_top_k');
      const minScoreEl = document.getElementById('wikipedia_min_score');
      if (btn) btn.disabled = true;
      if (spinner) spinner.style.display = 'inline-block';
      try {
        const topics = (topicsEl && topicsEl.value ? topicsEl.value : '')
          .split(/\\r?\\n|,/)
          .map(function(topic) { return topic.trim(); })
          .filter(Boolean);
        const res = await fetch('/api/wikipedia_cycle', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            action: 'start',
            topics: topics,
            top_k: topKEl ? parseInt(topKEl.value || '8') : 8,
            min_score: minScoreEl ? parseFloat(minScoreEl.value || '0.05') : 0.05,
          })
        });
        const data = await res.json();
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (statusEl) statusEl.textContent = data.message || (data.success ? 'Wikipedia cycle started.' : 'Failed to start Wikipedia cycle.');
        refreshWikipediaStatus();
      } catch (err) {
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (statusEl) statusEl.textContent = 'Network error while starting Wikipedia cycle.';
      } finally {
        if (btn) btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
      }
    }

    async function stopWikipediaCycle(btn) {
      const spinner = document.getElementById('wiki-stop-spinner');
      if (btn) btn.disabled = true;
      if (spinner) spinner.style.display = 'inline-block';
      try {
        const res = await fetch('/api/wikipedia_cycle', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: 'stop'})
        });
        const data = await res.json();
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (statusEl) statusEl.textContent = data.message || (data.success ? 'Wikipedia cycle stopped.' : 'Failed to stop Wikipedia cycle.');
        refreshWikipediaStatus();
      } catch (err) {
        const statusEl = document.getElementById('wikipedia-cycle-status');
        if (statusEl) statusEl.textContent = 'Network error while stopping Wikipedia cycle.';
      } finally {
        if (btn) btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
      }
    }

    function toggleSidebar() {
      const sidebar = document.getElementById('analysis-sidebar');
      const overlay = document.getElementById('sidebar-overlay');
      sidebar.classList.toggle('open');
      overlay.classList.toggle('open');
    }

    async function runAnalysis() {
      const btn = document.getElementById('analyse-btn');
      const spinner = document.getElementById('analyse-spinner');
      btn.disabled = true;
      spinner.style.display = 'inline-block';
      try {
        const res = await fetch('/api/pattern_analysis', { method: 'GET' });
        const data = await res.json();
        const body = document.getElementById('analysis-body');
        if (!data.success) {
          body.textContent = data.error || 'Analysis failed.';
          return;
        }
        body.innerHTML = renderAnalysis(data.report);
      } catch(err) {
        document.getElementById('analysis-body').textContent = 'Network error.';
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    function renderAnalysis(r) {
      const frag = document.createDocumentFragment();

      function makeSection(title) {
        const s = document.createElement('div');
        s.className = 'analysis-section';
        const t = document.createElement('div');
        t.className = 'analysis-section-title';
        t.textContent = title;
        s.appendChild(t);
        return s;
      }

      function makeRow(label, value) {
        const d = document.createElement('div');
        d.className = 'analysis-row';
        const l = document.createElement('span');
        l.className = 'analysis-label';
        l.textContent = label;
        const v = document.createElement('span');
        v.className = 'analysis-value';
        v.textContent = String(value);
        d.appendChild(l);
        d.appendChild(v);
        return d;
      }

      const summary = makeSection('Summary');
      summary.appendChild(makeRow('Total patterns', r.total_patterns));
      summary.appendChild(makeRow('Unique nodes', r.total_nodes));
      summary.appendChild(makeRow('Total edges', r.total_edges));
      frag.appendChild(summary);

      if (r.agents && Object.keys(r.agents).length) {
        const sec = makeSection('Patterns by Agent');
        const sorted = Object.entries(r.agents).sort(function(a,b){ return b[1].count - a[1].count; });
        for (const [name, stats] of sorted) {
          sec.appendChild(makeRow(name, stats.count));
          if (stats.count > 0) {
            const rr = document.createElement('div');
            rr.className = 'analysis-row';
            const l = document.createElement('span');
            l.className = 'analysis-label';
            l.style.paddingLeft = '12px';
            l.textContent = 'weight range';
            const v = document.createElement('span');
            v.className = 'analysis-value';
            v.style.fontSize = '0.72rem';
            v.textContent = stats.weight_min.toFixed(3) + ' – ' + stats.weight_max.toFixed(3) + ' (mean ' + stats.weight_mean.toFixed(3) + ')';
            rr.appendChild(l);
            rr.appendChild(v);
            sec.appendChild(rr);
          }
        }
        frag.appendChild(sec);
      }

      if (r.relations && Object.keys(r.relations).length) {
        const sec = makeSection('Relationship Types');
        const sorted = Object.entries(r.relations).sort(function(a,b){ return b[1] - a[1]; });
        for (const [rel, count] of sorted) {
          sec.appendChild(makeRow(rel, count));
        }
        frag.appendChild(sec);
      }

      if (r.components) {
        const sec = makeSection('Graph Structure');
        sec.appendChild(makeRow('Connected components', r.components.count));
        sec.appendChild(makeRow('Largest component', r.components.largest_size + ' nodes'));
        sec.appendChild(makeRow('Isolated nodes', r.components.isolated_nodes));
        frag.appendChild(sec);
      }

      if (r.top_hubs && r.top_hubs.length) {
        const sec = makeSection('Top Hubs');
        for (const hub of r.top_hubs) {
          const entry = document.createElement('div');
          entry.className = 'hub-entry';
          const hn = document.createElement('div');
          hn.className = 'hub-name';
          hn.textContent = hub.node;
          const hm = document.createElement('div');
          hm.className = 'hub-meta';
          hm.textContent = 'degree=' + hub.total_degree + ' (out=' + hub.out_degree + ' in=' + hub.in_degree + ') · ' + hub.top_relation;
          entry.appendChild(hn);
          entry.appendChild(hm);
          sec.appendChild(entry);
        }
        frag.appendChild(sec);
      }

      const ts = new Date(r.generated_at * 1000).toLocaleTimeString();
      const stamp = document.createElement('div');
      stamp.className = 'analysis-timestamp';
      stamp.textContent = 'Snapshot at ' + ts;
      frag.appendChild(stamp);

      const wrapper = document.createElement('div');
      wrapper.appendChild(frag);
      return wrapper.innerHTML;
    }
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
    if os.environ.get("HPM_WEB_DEMO_RETRAIN", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        local_reader.train(
            episodes=2,
            max_chunks=50,
            max_words_per_chunk=120,
            enable_pruning=False,
            enable_causal=False,
        )
    reasoning_agent = getattr(local_reader, "reasoning_agent", None)
    response_agent = getattr(local_reader, "response_agent", None)
    if reasoning_agent is not None:
        if hasattr(response_agent, "clear_reasoning_guidance_cache"):
            response_agent.clear_reasoning_guidance_cache()

        def _warm_reasoning_index() -> None:
            try:
                reasoning_agent.refresh()
            finally:
                if hasattr(response_agent, "clear_reasoning_guidance_cache"):
                    response_agent.clear_reasoning_guidance_cache()

        threading.Thread(target=_warm_reasoning_index, daemon=True).start()
    return local_reader


def _build_web_agent(local_reader: MultiAgentReader) -> WebAgent:
    return WebAgent(local_reader, corpus_path=_corpus_path(), min_sentence_len=20)


def _build_dataset_agent(local_reader: MultiAgentReader) -> DatasetTrainingAgent:
    return DatasetTrainingAgent(local_reader, corpus_path=_corpus_path(), min_sentence_len=20)


def _build_quiz_agent(local_reader: MultiAgentReader) -> _QuizAgent:
    reasoning_agent = getattr(local_reader, "reasoning_agent", None)
    return _QuizAgent(local_reader, reasoning_agent)


def _gutenberg_checkpoint_path() -> str:
    corpus_dir = os.path.dirname(_corpus_path())
    return os.path.join(corpus_dir, "gutenberg_progress.json")


def _load_gutenberg_checkpoint() -> dict:
    import json
    path = _gutenberg_checkpoint_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "book_id" in data and "chapter" in data:
                return data
        except Exception:
            pass
    return {"book_id": 1, "chapter": 1}


def _save_gutenberg_checkpoint(book_id: int, chapter: int) -> None:
    import json
    path = _gutenberg_checkpoint_path()
    try:
        with open(path, "w") as f:
            json.dump({"book_id": book_id, "chapter": chapter}, f)
    except Exception:
        pass


def _start_gutenberg_cycle() -> str:
    global gutenberg_cycle_thread
    if dataset_agent is None:
        return "Dataset agent is not ready."
    if gutenberg_cycle_state["active"]:
        return "Gutenberg sequential cycle is already running."

    checkpoint = _load_gutenberg_checkpoint()
    start_book_id = checkpoint["book_id"]
    start_chapter = checkpoint["chapter"]

    gutenberg_cycle_stop_event.clear()
    gutenberg_cycle_state["active"] = True
    if start_book_id == 1 and start_chapter == 1:
        gutenberg_cycle_state["message"] = "Running Gutenberg sequential cycle from the beginning."
    else:
        gutenberg_cycle_state["message"] = (
            f"Resuming Gutenberg sequential cycle from book {start_book_id}, chapter {start_chapter}."
        )

    def _worker() -> None:
        try:
            reports = dataset_agent.train_gutenberg_cycle(
                book_ids=None,
                top_k_per_chapter=8,
                min_score=0.05,
                retrain_epochs=1,
                split_on_paragraphs=False,
                stop_event=gutenberg_cycle_stop_event,
                repeat_books=False,
                report_progress=True,
                progress_callback=_update_gutenberg_cycle_state,
                maintenance_callback=_update_gutenberg_cycle_state,
                start_book_id=start_book_id,
                start_chapter=start_chapter,
                checkpoint_callback=_save_gutenberg_checkpoint,
            )
            final_books = len(reports)
            gutenberg_cycle_state["message"] = (
                f"Gutenberg sequential cycle stopped after {final_books} books."
            )
        except Exception as exc:  # pragma: no cover - defensive demo path
            gutenberg_cycle_state["message"] = f"Gutenberg sequential cycle failed: {exc}"
        finally:
            gutenberg_cycle_state["active"] = False

    gutenberg_cycle_thread = threading.Thread(target=_worker, daemon=True)
    gutenberg_cycle_thread.start()
    return gutenberg_cycle_state["message"]


def _stop_gutenberg_cycle() -> str:
    if not gutenberg_cycle_state["active"]:
        return "Gutenberg sequential cycle is not running."
    gutenberg_cycle_stop_event.set()
    gutenberg_cycle_state["phase"] = "stopping"
    return "Stopping Gutenberg sequential cycle."


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


def _start_wikipedia_cycle(topics: Optional[list[str]] = None, top_k: int = 8, min_score: float = 0.05) -> str:
    global wikipedia_cycle_thread
    if dataset_agent is None:
        return "Dataset agent is not ready."
    if wikipedia_cycle_state["active"]:
        return "Wikipedia cycle is already running."

    topics = [topic.strip() for topic in (topics or []) if topic and topic.strip()]
    if not topics:
        topics = dataset_agent.generate_wikipedia_topics(max_topics=8)
    if not topics:
        return "No Wikipedia topics available yet."

    wikipedia_topics_state["topics"] = list(topics)
    wikipedia_topics_state["topics_text"] = "\n".join(topics)
    wikipedia_topics_state["message"] = f"Generated {len(topics)} Wikipedia topics."

    wikipedia_cycle_stop_event.clear()
    wikipedia_cycle_state["active"] = True
    wikipedia_cycle_state["message"] = f"Running Wikipedia cycle over {len(topics)} topic(s) in the background."
    wikipedia_cycle_state["topics_processed"] = 0
    wikipedia_cycle_state["pages_added"] = 0

    def _worker() -> None:
        try:
            reports = dataset_agent.train_wikipedia_cycle(
                topics=topics,
                top_k_per_topic=top_k,
                min_score=min_score,
                retrain_epochs=1,
                stop_event=wikipedia_cycle_stop_event,
                repeat_topics=True,
                report_progress=True,
                progress_callback=_update_wikipedia_cycle_state,
                maintenance_callback=_update_wikipedia_cycle_state,
            )
            wikipedia_cycle_state["message"] = f"Wikipedia cycle stopped after {len(reports)} topics."
        except Exception as exc:  # pragma: no cover - defensive demo path
            wikipedia_cycle_state["message"] = f"Wikipedia cycle failed: {exc}"
        finally:
            wikipedia_cycle_state["active"] = False

    wikipedia_cycle_thread = threading.Thread(target=_worker, daemon=True)
    wikipedia_cycle_thread.start()
    return wikipedia_cycle_state["message"]


def _stop_wikipedia_cycle() -> str:
    if not wikipedia_cycle_state["active"]:
        return "Wikipedia cycle is not running."
    wikipedia_cycle_stop_event.set()
    wikipedia_cycle_state["phase"] = "stopping"
    return "Stopping Wikipedia cycle."


def _update_wikipedia_cycle_state(event: dict) -> None:
    event_type = event.get("type", "unknown")
    if event_type == "topic_start":
        wikipedia_cycle_state["topic"] = event.get("topic")
        wikipedia_cycle_state["page_title"] = None
        wikipedia_cycle_state["phase"] = "topic_start"
        wikipedia_cycle_state["topics_processed"] = event.get("topics_processed", wikipedia_cycle_state["topics_processed"])
    elif event_type == "topic_skip":
        wikipedia_cycle_state["topic"] = event.get("topic", wikipedia_cycle_state["topic"])
        wikipedia_cycle_state["page_title"] = None
        wikipedia_cycle_state["phase"] = "topic_skip"
    elif event_type == "page_start":
        wikipedia_cycle_state["topic"] = event.get("topic", wikipedia_cycle_state["topic"])
        wikipedia_cycle_state["page_title"] = event.get("page_title")
        wikipedia_cycle_state["phase"] = "page_start"
    elif event_type == "page_done":
        wikipedia_cycle_state["topic"] = event.get("topic", wikipedia_cycle_state["topic"])
        wikipedia_cycle_state["page_title"] = event.get("page_title", wikipedia_cycle_state["page_title"])
        wikipedia_cycle_state["phase"] = "page_done"
        wikipedia_cycle_state["pages_added"] = wikipedia_cycle_state.get("pages_added", 0) + int(event.get("added", 0))
    elif event_type == "maintenance_done":
        wikipedia_cycle_state["topic"] = event.get("topic", wikipedia_cycle_state["topic"])
        wikipedia_cycle_state["phase"] = "maintenance_done"
        maintenance_report = event.get("report", {})
        summary = maintenance_report.get("_summary", {}) if isinstance(maintenance_report, dict) else {}
        improved = ", ".join(summary.get("improved_agents", [])) or "none"
        loaded = summary.get("loaded", 0)
        wikipedia_cycle_state["message"] = (
            f"Processed Wikipedia topic {wikipedia_cycle_state['topic']} with maintenance loaded {loaded} patterns; "
            f"improved agents: {improved}."
        )
    elif event_type == "cycle_done":
        wikipedia_cycle_state["phase"] = "cycle_done"
        wikipedia_cycle_state["topics_processed"] = event.get("topics_processed", wikipedia_cycle_state["topics_processed"])
    elif event_type == "topic_done":
        wikipedia_cycle_state["topic"] = event.get("topic", wikipedia_cycle_state["topic"])
        wikipedia_cycle_state["phase"] = "topic_done"
        wikipedia_cycle_state["topics_processed"] = event.get("topics_processed", wikipedia_cycle_state["topics_processed"])


@app.route("/", methods=["GET", "POST"])
def index():
    seed = "Alice was"
    max_len = 50
    urls = ""
    rss_url = ""
    top_k = 20
    min_score = 0.05
    gutenberg_ids = ", ".join(str(book_id) for book_id in DatasetTrainingAgent.curated_gutenberg_book_ids())
    gutenberg_top_k = 12
    gutenberg_min_score = 0.05
    wikipedia_topics = wikipedia_topics_state.get("topics_text", "")
    wikipedia_top_k = 12
    wikipedia_min_score = 0.05
    generated = None
    error = None
    training_message = None
    question = "Why did Alice follow the rabbit?"
    reasoning = None
    reasoning_trace = None

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
        wikipedia_topics = request.form.get("wikipedia_topics", wikipedia_topics)
        wikipedia_top_k = int(request.form.get("wikipedia_top_k", wikipedia_top_k))
        wikipedia_min_score = float(request.form.get("wikipedia_min_score", wikipedia_min_score))
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
            elif action == "start_wikipedia_cycle":
                parsed_topics = [line.strip() for line in wikipedia_topics.splitlines() if line.strip()]
                training_message = _start_wikipedia_cycle(parsed_topics, top_k=wikipedia_top_k, min_score=wikipedia_min_score)
            elif action == "stop_wikipedia_cycle":
                training_message = _stop_wikipedia_cycle()
            elif action == "generate_wikipedia_topics":
                generated_topics = dataset_agent.generate_wikipedia_topics(max_topics=8) if dataset_agent is not None else []
                wikipedia_topics = "\n".join(generated_topics)
                wikipedia_topics_state["topics"] = list(generated_topics)
                wikipedia_topics_state["topics_text"] = wikipedia_topics
                wikipedia_topics_state["message"] = (
                    f"Generated {len(generated_topics)} Wikipedia topics from learned patterns."
                    if generated_topics
                    else "No Wikipedia topics were generated from learned patterns."
                )
                training_message = wikipedia_topics_state["message"]
            elif action == "reason":
                reasoning_trace = _build_reason_trace(question)
                reasoning = reasoning_trace.get("answer") if reasoning_trace else "Reader not ready."
            else:
                generated = reader.generate(seed, max_length=max_len) if reader is not None else None
        except Exception as exc:  # pragma: no cover - defensive demo path
            error = str(exc)

    cycle_message = gutenberg_cycle_state["message"]
    cycle_active = gutenberg_cycle_state["active"]

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
        wikipedia_topics=wikipedia_topics,
        wikipedia_top_k=wikipedia_top_k,
        wikipedia_min_score=wikipedia_min_score,
        generated=generated,
        error=error,
        training_message=training_message,
        question=question,
        reasoning=reasoning,
        reasoning_trace=reasoning_trace,
        reasoning_index_status=_reasoning_index_status(),
        cycle_message=cycle_message,
        cycle_active=cycle_active,
        wikipedia_cycle_message=wikipedia_cycle_state["message"],
        wikipedia_cycle_state=wikipedia_cycle_state,
        wikipedia_cycle_active=wikipedia_cycle_state["active"],
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
    min_score = float(payload.get("min_score", 0.05))
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
        trace = _build_reason_trace(question)
        return jsonify({"success": True, "reasoning": trace.get("answer", ""), "trace": trace})
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/reasoning_index_status", methods=["GET"])
def api_reasoning_index_status():
    return jsonify({"success": True, **_reasoning_index_status()})


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


@app.route("/api/wikipedia_topics", methods=["GET"])
def api_wikipedia_topics():
    if dataset_agent is None:
        return jsonify({"success": False, "error": "Dataset agent is not ready."}), 503
    try:
        topics = dataset_agent.generate_wikipedia_topics(max_topics=8)
        wikipedia_topics_state["topics"] = list(topics)
        wikipedia_topics_state["topics_text"] = "\n".join(topics)
        wikipedia_topics_state["message"] = (
            f"Generated {len(topics)} Wikipedia topics from learned patterns."
            if topics
            else "No Wikipedia topics were generated from learned patterns."
        )
        return jsonify({
            "success": True,
            "topics": topics,
            "message": wikipedia_topics_state["message"],
        })
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/wikipedia_cycle", methods=["POST"])
def api_wikipedia_cycle():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "start")).strip().lower()
    topics = payload.get("topics", [])
    top_k = int(payload.get("top_k", 8))
    min_score = float(payload.get("min_score", 0.05))

    try:
        if action == "start":
            if isinstance(topics, str):
                topics = [line.strip() for line in topics.splitlines() if line.strip()]
            message = _start_wikipedia_cycle(topics=topics, top_k=top_k, min_score=min_score)
            return jsonify({"success": True, "message": message})
        if action == "stop":
            message = _stop_wikipedia_cycle()
            return jsonify({"success": True, "message": message})
        return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400
    except Exception as exc:  # pragma: no cover - defensive demo path
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/pattern_analysis", methods=["GET"])
def api_pattern_analysis():
    if reader is None:
        return jsonify({"success": False, "error": "Reader not ready."}), 503
    try:
        report = reader.reasoning_agent.analyse_patterns()
        return jsonify({"success": True, "report": report})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/wikipedia_cycle_status", methods=["GET"])
def api_wikipedia_cycle_status():
    return jsonify({
        "success": True,
        "active": wikipedia_cycle_state["active"],
        "message": wikipedia_cycle_state["message"],
        "topic": wikipedia_cycle_state["topic"],
        "page_title": wikipedia_cycle_state["page_title"],
        "phase": wikipedia_cycle_state["phase"],
        "topics_processed": wikipedia_cycle_state["topics_processed"],
        "pages_added": wikipedia_cycle_state["pages_added"],
        "topics": wikipedia_topics_state.get("topics", []),
    })


@app.route("/api/quiz/generate", methods=["POST"])
def api_quiz_generate():
    global _quiz_state
    if _quiz_agent is None:
        return jsonify({"error": "Quiz agent not initialised"}), 503
    data = request.get_json(force=True)
    n = int(data.get("n", 5))
    difficulty = data.get("difficulty", "easy")
    source = data.get("source", "bank")
    questions = _quiz_agent.generate_quiz(n=n, difficulty=difficulty, source=source)
    _quiz_state = {q.id: q for q in questions}
    return jsonify({
        "questions": [
            {"id": q.id, "question": q.question, "options": q.options,
             "topic": q.topic, "difficulty": q.difficulty, "source": q.source}
            for q in questions
        ]
    })


@app.route("/api/quiz/submit", methods=["POST"])
def api_quiz_submit():
    data = request.get_json(force=True)
    question_id = data.get("question_id")
    answer_index = int(data.get("answer_index", -1))
    if question_id not in _quiz_state:
        return jsonify({"error": "Unknown question id"}), 404
    q = _quiz_state[question_id]
    return jsonify({
        "correct": answer_index == q.correct_index,
        "correct_index": q.correct_index,
        "explanation": q.explanation,
    })


@app.route("/api/quiz/ai_answer", methods=["POST"])
def api_quiz_ai_answer():
    """Use ReasoningAgent to answer a quiz question automatically."""
    data = request.get_json(force=True)
    question_id = data.get("question_id")
    if not question_id or question_id not in _quiz_state:
        return jsonify({"error": "Unknown question_id"}), 400

    q = _quiz_state[question_id]
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        return jsonify({"error": "ReasoningAgent not available"}), 503

    prompt = (
        "Given this question and these 4 options, which is most likely correct based on "
        "what you know? "
        f"Question: {q.question}. "
        f"Options: A) {q.options[0]} B) {q.options[1]} C) {q.options[2]} D) {q.options[3]}. "
        "Reply with ONLY the letter (A, B, C, or D) followed by a brief explanation."
    )
    try:
        response = reasoning_agent.reason(prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    first_letter = next(
        (ch for ch in response.strip().upper() if ch in letter_map), None
    )
    answer_index = letter_map.get(first_letter, 0)
    return jsonify({"answer_index": answer_index, "reasoning": response})


@app.route("/api/quiz/train_gaps", methods=["POST"])
def api_quiz_train_gaps():
    data = request.get_json(force=True)
    topics = data.get("topics", [])
    if not topics:
        return jsonify({"status": "ok", "message": "No topics to train on"})
    msg = _start_wikipedia_cycle(topics=topics)
    return jsonify({"status": "ok", "message": msg})


@app.route("/api/quiz/banks/<difficulty>", methods=["GET"])
def api_quiz_banks(difficulty):
    if difficulty not in ("easy", "medium", "hard"):
        return jsonify({"error": "Invalid difficulty"}), 400
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "quiz_banks", f"{difficulty}.json"
    )
    return send_file(
        bank_path,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"quiz_bank_{difficulty}.json",
    )


def main():
    global reader, web_agent, dataset_agent, _quiz_agent
    print("Loading MultiAgentReader...")
    reader = _build_reader()
    web_agent = _build_web_agent(reader)
    dataset_agent = _build_dataset_agent(reader)
    _quiz_agent = _build_quiz_agent(reader)
    print("Reader ready.")
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
