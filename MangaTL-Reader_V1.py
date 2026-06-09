#!/usr/bin/env python3
"""
MangaTL-Reader  —  Single-file manga translation tool
======================================================

Translates any MangaDex chapter that has a non-English translation
(Vietnamese, Korean, Indonesian, etc.) into English or your chosen language
using OCR (EasyOCR / Gemini Vision) + AI translation (Gemini / DeepSeek).

USAGE
  python MangaTL-Reader.py        — starts the server and opens the browser
  (Windows) double-click the file — same, if Python is installed

FIRST RUN
  Packages install automatically (Flask, EasyOCR, OpenCV …).
  EasyOCR downloads a ~100–400 MB language model — this takes 2-5 minutes.
  The browser opens as soon as the server is ready.

REQUIREMENTS
  Python 3.9+  ·  Internet connection  ·  A Gemini or DeepSeek API key

API KEYS (free options)
  Gemini  → https://aistudio.google.com/app/apikey  (free tier, no credit card)
  DeepSeek → https://platform.deepseek.com           (~$0.02–0.05 / chapter)
"""

# ─── Auto-install missing dependencies ───────────────────────────────────────
# Runs before every other import so we can guarantee packages exist.
# On first run this takes a few minutes; subsequent runs skip instantly.

import sys
import subprocess
import importlib.util

_REQUIRED = {
    "flask":                  "flask",
    "requests":               "requests",
    "easyocr":                "easyocr",
    "pillow":                 "PIL",
    "numpy":                  "numpy",
    "opencv-python-headless": "cv2",
}

def _bootstrap():
    missing = [pkg for pkg, mod in _REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║      MangaTL — First-Time Setup              ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print(f"  Installing {len(missing)} missing package(s):")
    for p in missing:
        print(f"    •  {p}")
    print()
    print("  EasyOCR includes a language model (~100–400 MB).")
    print("  This may take 2–5 minutes. Please wait.")
    print("  The browser will open automatically when ready.")
    print()

    # --break-system-packages is needed on modern Debian/Ubuntu systems but is
    # silently ignored on Windows and macOS — safe to pass unconditionally.
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--break-system-packages"] + missing
        )
        print("  ✓  Setup complete!\n")
    except subprocess.CalledProcessError:
        print()
        print("  ✗  Auto-install failed.")
        print("  Run this manually, then try again:")
        print("     pip install " + " ".join(missing))
        sys.exit(1)

_bootstrap()

# ─── All other imports (safe after bootstrap) ─────────────────────────────────

import io
import socket
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, Response, abort, jsonify, request
from PIL import Image

HOST         = "127.0.0.1"
PORT         = 8080
# ─── Embedded frontend (generated — do not edit the HTML here directly) ─────
_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>MangaTL Reader</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #0c0c10;
  --bg2:      #111116;
  --bg3:      #16161e;
  --bg4:      #1c1c26;
  --text:     #d4d0c8;
  --dim:      #454050;
  --mid:      #807888;
  --border:   #222230;
  --border2:  #2e2e40;
  --ui:       #00e5b0;
  --danger:   #ff3a3a;
  --speech:   #a78bfa;
  --thought:  #60a5fa;
  --sfx:      #f87171;
  --narr:     #fbbf24;
  --sign:     #34d399;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 1.05rem;
  line-height: 1.8;
  min-height: 100dvh;
  -webkit-tap-highlight-color: transparent;
}

.screen { display: none; }
.screen.active { display: flex; flex-direction: column; min-height: 100dvh; }

/* ─── HOME ───────────────────────────────────── */
#screen-home {
  align-items: center;
  justify-content: center;
  padding: 2.5rem 1.4rem;
}
.home-inner { width: 100%; max-width: 460px; }
.home-brand { text-align: center; margin-bottom: 2.8rem; }
.home-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3.2rem;
  letter-spacing: 0.14em;
  color: var(--dim);
  line-height: 1;
}
.home-title span { color: var(--ui); }
.home-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.22em;
  color: var(--dim);
  opacity: 0.7;
  text-transform: uppercase;
  margin-top: 0.35rem;
}
.form-group { margin-bottom: 1.1rem; }
.merge-slider-row {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin-bottom: 1.1rem;
}
.merge-slider-row input[type=range] {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}
.merge-slider-row input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 13px; height: 13px;
  border-radius: 50%;
  background: var(--ui);
  cursor: pointer;
}
.merge-slider-row input[type=range]::-moz-range-thumb {
  width: 13px; height: 13px;
  border-radius: 50%;
  background: var(--ui);
  border: none;
  cursor: pointer;
}
.merge-slider-val {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem;
  color: var(--ui);
  min-width: 2.8rem;
  text-align: right;
}
.form-label {
  display: block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.16em;
  color: var(--dim);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.form-input {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  padding: 0.75rem 0.95rem;
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus { border-color: var(--border2); }
.form-input::placeholder { color: var(--dim); opacity: 0.6; }
.form-row { display: flex; gap: 0.8rem; }
.form-row .form-group { flex: 1; margin-bottom: 0; }
.form-row-wrap { margin-bottom: 1.1rem; }
.btn-go {
  width: 100%;
  background: transparent;
  border: 1px solid var(--ui);
  color: var(--ui);
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.2rem;
  letter-spacing: 0.25em;
  padding: 0.9rem;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 0.6rem;
  transition: background 0.15s, color 0.15s;
}
.btn-go:hover   { background: rgba(0,229,176,0.07); }
.btn-go:active  { background: var(--ui); color: #000; }
.btn-go:disabled { border-color: var(--dim); color: var(--dim); cursor: not-allowed; }
.home-hint {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.08em;
  color: var(--dim);
  opacity: 0.6;
  text-align: center;
  margin-top: 1.2rem;
  line-height: 2;
}
.home-hint a { color: inherit; text-decoration: underline; cursor: pointer; }

/* MangaDex login accordion */
.md-login-wrap {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 1.1rem;
  overflow: hidden;
}
.md-login-toggle {
  width: 100%; background: none; border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.6rem 0.85rem;
  color: var(--mid); font-family: inherit; font-size: 0.78rem; letter-spacing: 0.04em;
  transition: color 0.15s;
}
.md-login-toggle:hover { color: var(--text); }
.md-login-toggle .md-status {
  display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem;
}
.md-status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--dim); flex-shrink: 0; transition: background 0.2s;
}
.md-status-dot.online { background: var(--ui); }
.md-login-toggle .arrow { font-size: 0.65rem; transition: transform 0.2s; }
.md-login-wrap.open .arrow { transform: rotate(90deg); }
.md-login-body {
  display: none; padding: 0.75rem 0.85rem 0.85rem;
  border-top: 1px solid var(--border);
  background: var(--bg2);
}
.md-login-wrap.open .md-login-body { display: block; }
.md-login-body .form-group { margin-bottom: 0.75rem; }
.md-login-body .form-group:last-of-type { margin-bottom: 0.85rem; }
.md-login-hint {
  font-size: 0.72rem; color: var(--dim); margin-bottom: 0.85rem; line-height: 1.5;
}
.md-login-hint a { color: var(--mid); text-decoration: underline; }
.btn-md-login {
  width: 100%; padding: 0.55rem; border-radius: 5px;
  border: 1px solid var(--border2); background: none;
  color: var(--text); font-family: inherit; font-size: 0.82rem;
  cursor: pointer; transition: border-color 0.15s, color 0.15s;
}
.btn-md-login:hover    { border-color: var(--ui); color: var(--ui); }
.btn-md-login:disabled { opacity: 0.45; cursor: not-allowed; }

/* ─── READER ─────────────────────────────────── */
#screen-reader { padding: 0; background: var(--bg); }
.reader-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 1.2rem 0.6rem;
}
.reader-header-top {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  margin-bottom: 0.45rem;
}
.btn-back {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  padding: 0.4rem 0.7rem;
  border-radius: 3px;
  cursor: pointer;
  letter-spacing: 0.1em;
  white-space: nowrap;
  flex-shrink: 0;
  transition: border-color 0.15s, color 0.15s;
}
.btn-back:active { border-color: var(--mid); color: var(--mid); }
.reader-titles  { flex: 1; min-width: 0; }
.reader-manga-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.05em;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.reader-ch-info {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  color: var(--dim);
  margin-top: 0.05rem;
  letter-spacing: 0.05em;
}
.reader-credit {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  color: var(--dim);
  margin-top: 0.18rem;
  letter-spacing: 0.04em;
  opacity: 0.7;
}
.reader-credit a {
  color: var(--mid);
  text-decoration: none;
  border-bottom: 1px solid var(--border2);
  transition: color 0.15s, border-color 0.15s;
}
.reader-credit a:hover { color: var(--ui); border-color: var(--ui); }
.reader-header-bottom {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  margin-bottom: 0.38rem;
}
.reader-status {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: var(--mid);
  letter-spacing: 0.08em;
  flex: 1;
}
.cache-pill {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 0.18rem 0.55rem 0.18rem 0.45rem;
  flex-shrink: 0;
  white-space: nowrap;
}
.cache-pill-icon {
  font-size: 0.6rem;
  opacity: 0.6;
}
.cache-pill-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.56rem;
  color: var(--dim);
  letter-spacing: 0.08em;
}
.cache-pill-label span {
  color: var(--ui);
  font-weight: bold;
}
.btn-cache-clear {
  background: transparent;
  border: none;
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  letter-spacing: 0.06em;
  cursor: pointer;
  padding: 0 0.1rem;
  border-left: 1px solid var(--border);
  margin-left: 0.3rem;
  padding-left: 0.4rem;
  transition: color 0.15s;
  line-height: 1;
}
.btn-cache-clear:hover { color: var(--danger); }

/* ─── HOME CACHE STRIP ───────────────────────── */
.home-cache-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.55rem 0.85rem;
  margin-bottom: 1.1rem;
  gap: 0.8rem;
}
.home-cache-info {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: var(--dim);
  letter-spacing: 0.06em;
  line-height: 1.7;
}
.home-cache-info strong {
  color: var(--ui);
  font-weight: normal;
}
.home-cache-info .cache-empty {
  color: var(--dim);
  opacity: 0.5;
}
.btn-clear-cache-home {
  background: transparent;
  border: 1px solid var(--border2);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.1em;
  padding: 0.35rem 0.7rem;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
  transition: border-color 0.15s, color 0.15s;
}
.btn-clear-cache-home:hover    { border-color: var(--danger); color: var(--danger); }
.btn-clear-cache-home:disabled { opacity: 0.3; cursor: not-allowed; }
.progress-track {
  height: 2px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--ui);
  border-radius: 2px;
  width: 0%;
  transition: width 0.35s ease;
}

/* ─── PAGES ──────────────────────────────────── */
#pages-container { flex: 1; padding: 1.2rem 1rem 4rem; }
.page-card {
  max-width: 700px;
  margin: 0 auto 2.8rem;
  animation: fadeUp 0.3s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.img-wrap { position: relative; width: 100%; line-height: 0; }
.page-img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 3px 3px 0 0;
  border: 1px solid var(--border);
  border-bottom: none;
}
.page-img-only {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 3px;
  border: 1px solid var(--border);
}
.pg-label {
  position: absolute;
  bottom: 8px;
  right: 8px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  color: rgba(255,255,255,0.6);
  background: rgba(0,0,0,0.6);
  padding: 0.2rem 0.45rem;
  border-radius: 2px;
  pointer-events: none;
}
.badge {
  position: absolute;
  transform: translate(-50%, -50%);
  width: 18px; height: 18px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  font-weight: bold;
  pointer-events: none;
  border: 1.5px solid rgba(0,0,0,0.4);
  line-height: 1;
  z-index: 2;
}
.badge.t-speech    { background: var(--speech); color: #000; }
.badge.t-thought   { background: var(--thought); color: #000; }
.badge.t-sfx       { background: var(--sfx); color: #000; }
.badge.t-narration { background: var(--narr); color: #000; }
.badge.t-sign      { background: var(--sign); color: #000; }

.trans-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 4px 4px;
}
.t-row {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.45rem 0.85rem;
  border-bottom: 1px solid var(--border);
  line-height: 1.55;
}
.t-row:last-child { border-bottom: none; }
.t-num {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: var(--dim);
  min-width: 1.1rem;
  text-align: right;
  flex-shrink: 0;
}
.t-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.06em;
  padding: 0.1rem 0.32rem;
  border-radius: 2px;
  flex-shrink: 0;
  text-transform: uppercase;
  border: 1px solid;
  opacity: 0.7;
}
.t-tag.speech    { color: var(--speech); border-color: var(--speech); }
.t-tag.thought   { color: var(--thought); border-color: var(--thought); }
.t-tag.sfx       { color: var(--sfx); border-color: var(--sfx); }
.t-tag.narration { color: var(--narr); border-color: var(--narr); }
.t-tag.sign      { color: var(--sign); border-color: var(--sign); }
.t-text {
  font-family: 'Crimson Pro', serif;
  font-size: 0.97rem;
  color: var(--text);
  line-height: 1.5;
}
.no-text-note {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 0.6rem 0.85rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: var(--dim);
  letter-spacing: 0.1em;
  text-align: center;
  opacity: 0.6;
}

/* ─── READING ORDER TOGGLE ───────────────────── */
.read-order-row {
  display: flex;
  align-items: center;
  gap: 0;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 1.1rem;
}
.read-order-btn {
  flex: 1;
  background: transparent;
  border: none;
  border-right: 1px solid var(--border);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  padding: 0.6rem 0.3rem;
  cursor: pointer;
  transition: background 0.14s, color 0.14s;
  text-align: center;
  line-height: 1.5;
}
.read-order-btn:last-child { border-right: none; }
.read-order-btn:hover { color: var(--text); background: rgba(255,255,255,0.04); }
.read-order-btn.active { color: var(--ui); background: rgba(0,229,176,0.08); }
.read-order-label {
  display: block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.16em;
  color: var(--dim);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

/* ─── INLINE REORDER PANEL ───────────────────── */
.btn-reorder-page {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0,0,0,0.65);
  border: 1px solid var(--border2);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  padding: 0.22rem 0.5rem;
  border-radius: 3px;
  cursor: pointer;
  z-index: 3;
  transition: color 0.15s, border-color 0.15s;
}
.btn-reorder-page:hover  { color: var(--ui); border-color: var(--ui); }
.btn-reorder-page.active { color: var(--ui); border-color: var(--ui); background: rgba(0,229,176,0.08); }

.reorder-panel {
  background: var(--bg2);
  border: 1px solid var(--ui);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 0.55rem 0.75rem 0.75rem;
}
.reorder-panel-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}
.reorder-panel-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  color: var(--ui);
  text-transform: uppercase;
}
.reorder-hint {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.55rem;
  color: var(--dim);
  opacity: 0.7;
  letter-spacing: 0.06em;
}
.reorder-list { list-style: none; }
.reorder-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.28rem 0;
  border-bottom: 1px solid var(--border);
  cursor: grab;
  user-select: none;
  transition: background 0.1s;
  border-radius: 2px;
}
.reorder-item:last-child { border-bottom: none; }
.reorder-item.dragging { opacity: 0.45; cursor: grabbing; }
.reorder-item.drag-over { background: rgba(0,229,176,0.08); }
.reorder-drag-handle {
  color: var(--dim);
  font-size: 0.75rem;
  flex-shrink: 0;
  opacity: 0.5;
  cursor: grab;
  padding: 0 0.15rem;
}
.reorder-badge-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px; height: 18px;
  border-radius: 50%;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.52rem;
  font-weight: bold;
  border: 1.5px solid rgba(0,0,0,0.4);
  flex-shrink: 0;
  line-height: 1;
}
.reorder-badge-num.t-speech    { background: var(--speech); color: #000; }
.reorder-badge-num.t-thought   { background: var(--thought); color: #000; }
.reorder-badge-num.t-sfx       { background: var(--sfx); color: #000; }
.reorder-badge-num.t-narration { background: var(--narr); color: #000; }
.reorder-badge-num.t-sign      { background: var(--sign); color: #000; }
.reorder-item-text {
  flex: 1;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  color: var(--mid);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
.reorder-arrow-btns { display: flex; gap: 2px; flex-shrink: 0; }
.reorder-arrow-btns button {
  background: transparent;
  border: 1px solid var(--border2);
  color: var(--dim);
  font-size: 0.65rem;
  width: 22px; height: 22px;
  border-radius: 3px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: color 0.12s, border-color 0.12s;
  line-height: 1;
  padding: 0;
}
.reorder-arrow-btns button:hover { color: var(--ui); border-color: var(--ui); }
.reorder-arrow-btns button:disabled { opacity: 0.25; cursor: not-allowed; }
.btn-apply-order {
  margin-top: 0.55rem;
  width: 100%;
  background: transparent;
  border: 1px solid var(--ui);
  color: var(--ui);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  padding: 0.45rem;
  border-radius: 3px;
  cursor: pointer;
  transition: background 0.14s;
}
.btn-apply-order:hover { background: rgba(0,229,176,0.08); }
.page-err-note {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  background: var(--bg2);
  border: 1px solid rgba(255,58,58,0.3);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 0.55rem 0.85rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: var(--danger);
  letter-spacing: 0.05em;
}
.btn-retry {
  background: transparent;
  border: 1px solid var(--danger);
  color: var(--danger);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  padding: 0.3rem 0.65rem;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  letter-spacing: 0.1em;
  flex-shrink: 0;
  transition: background 0.15s;
}
.btn-retry:hover    { background: rgba(255,58,58,0.1); }
.btn-retry:disabled { opacity: 0.4; cursor: not-allowed; }

/* Skeleton */
.page-skeleton {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.4rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.sk-num {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  color: var(--border2);
  min-width: 2rem;
  line-height: 1;
}
.sk-bar {
  flex: 1;
  height: 2px;
  background: var(--border);
  border-radius: 2px;
  position: relative;
  overflow: hidden;
}
.sk-bar::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, var(--border2), transparent);
  animation: shimmer 1.6s infinite;
}
@keyframes shimmer {
  from { transform: translateX(-100%); }
  to   { transform: translateX(100%); }
}

/* ─── NAV BAR ─────────────────────────────────── */
.nav-bar {
  display: none;
  position: sticky;
  bottom: 0;
  z-index: 100;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  padding: 0.7rem 1.2rem;
  justify-content: space-between;
  align-items: center;
}
.btn-nav {
  background: transparent;
  border: 1px solid var(--border2);
  color: var(--mid);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  padding: 0.5rem 1rem;
  border-radius: 3px;
  cursor: pointer;
  letter-spacing: 0.12em;
  transition: border-color 0.15s, color 0.15s;
}
.btn-nav:hover  { border-color: var(--ui); color: var(--ui); }
.btn-nav:active { background: rgba(0,229,176,0.08); }

/* ─── TOAST ───────────────────────────────────── */
#toast {
  display: none;
  position: fixed;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg4);
  border: 1px solid var(--border2);
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
  padding: 0.65rem 1.1rem;
  border-radius: 4px;
  z-index: 9999;
  max-width: calc(100vw - 2.4rem);
  text-align: center;
  letter-spacing: 0.05em;
  line-height: 1.6;
}

/* ─── CORRECTION UI ──────────────────────────── */
.btn-correct {
  position: absolute;
  top: 8px; left: 8px;
  background: rgba(0,0,0,0.65);
  border: 1px solid var(--border2);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  padding: 0.22rem 0.5rem;
  border-radius: 3px;
  cursor: pointer;
  z-index: 3;
  transition: color 0.15s, border-color 0.15s;
}
.btn-correct:hover { color: var(--ui); border-color: var(--ui); }
.btn-correct.active { color: var(--ui); border-color: var(--ui); background: rgba(0,229,176,0.08); }

.page-card.correcting { max-width: min(95vw, 1300px); }

.corr-layout {
  display: flex;
  align-items: flex-start;
  border: 1px solid var(--border);
  border-radius: 3px 3px 0 0;
  overflow: hidden;
}
.corr-left {
  flex: 0 0 55%;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
  min-width: 0;
}
.corr-toolbar {
  display: flex;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.corr-tool {
  flex: 1;
  background: transparent;
  border: none;
  border-right: 1px solid var(--border);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.56rem;
  letter-spacing: 0.08em;
  padding: 0.5rem 0.2rem;
  cursor: pointer;
  transition: color 0.12s, background 0.12s;
}
.corr-tool:last-child { border-right: none; }
.corr-tool:hover { color: var(--text); }
.corr-tool.active { color: var(--ui); background: rgba(0,229,176,0.06); }

.corr-img-wrap {
  position: relative;
  width: 100%;
  line-height: 0;
}
.corr-img { width: 100%; height: auto; display: block; user-select: none; }

.corr-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.corr-overlay.mode-select,
.corr-overlay.mode-delete,
.corr-overlay.mode-draw { pointer-events: auto; cursor: crosshair; }

.corr-rbox {
  position: absolute;
  border: 2px solid rgba(0,229,176,0.6);
  background: rgba(0,229,176,0.07);
  cursor: pointer;
  box-sizing: border-box;
  transition: background 0.1s;
}
.corr-rbox:hover { background: rgba(0,229,176,0.18); }
.corr-rbox.selected { border-color: var(--ui); background: rgba(0,229,176,0.22); z-index: 2; }
.corr-rbox.mode-delete { border-color: rgba(255,58,58,0.6); background: rgba(255,58,58,0.07); }
.corr-rbox.mode-delete:hover { background: rgba(255,58,58,0.22); }
.rbox-num {
  position: absolute;
  top: -1px; left: -1px;
  background: var(--ui);
  color: #000;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.5rem;
  font-weight: bold;
  padding: 0.05rem 0.28rem;
  border-radius: 0 0 3px 0;
  pointer-events: none;
  line-height: 1.7;
}
.rbox-num.raw { background: #fbbf24; }

.corr-raw-box {
  position: absolute;
  border: 1.5px dashed rgba(251,191,36,0.8);
  background: rgba(251,191,36,0.05);
  pointer-events: none;
  box-sizing: border-box;
}
.draw-preview {
  position: absolute;
  border: 2px dashed var(--ui);
  background: rgba(0,229,176,0.07);
  pointer-events: none;
  box-sizing: border-box;
}

.corr-sidebar {
  flex: 1;
  background: var(--bg2);
  padding: 0.85rem;
  overflow-y: auto;
  max-height: 80vh;
  min-height: 220px;
  min-width: 0;
}
.corr-empty-hint {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  color: var(--dim);
  opacity: 0.5;
  text-align: center;
  padding: 2.5rem 0;
  line-height: 2.2;
  letter-spacing: 0.05em;
}
.corr-sid-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 0.9rem;
  letter-spacing: 0.2em;
  color: var(--ui);
  margin-bottom: 0.65rem;
}
.corr-sid-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.56rem;
  letter-spacing: 0.14em;
  color: var(--dim);
  text-transform: uppercase;
  margin: 0.65rem 0 0.28rem;
}
.corr-textarea {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem;
  padding: 0.45rem 0.55rem;
  outline: none;
  resize: vertical;
  min-height: 62px;
  line-height: 1.6;
  transition: border-color 0.15s;
  box-sizing: border-box;
}
.corr-textarea:focus { border-color: var(--border2); }
.corr-type-sel {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
  padding: 0.38rem 0.5rem;
  outline: none;
  cursor: pointer;
  appearance: none;
}
.corr-action-row {
  display: flex;
  gap: 0.38rem;
  flex-wrap: wrap;
  margin: 0.65rem 0 0.3rem;
  align-items: center;
}
.corr-action-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 0.08em;
  padding: 0.28rem 0.58rem;
  border-radius: 3px;
  cursor: pointer;
  transition: color 0.12s, border-color 0.12s;
  white-space: nowrap;
}
.corr-action-btn:hover { color: var(--ui); border-color: var(--ui); }
.corr-action-btn.danger:hover { color: var(--danger); border-color: var(--danger); }
.corr-action-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.corr-split-list { margin: 0.4rem 0; display: flex; flex-direction: column; gap: 0.22rem; }
.corr-split-item {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.63rem;
  color: var(--text);
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0.32rem 0.5rem;
}
.corr-split-line-btn {
  display: block;
  width: 100%;
  background: transparent;
  border: 1px dashed var(--border);
  color: var(--dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.56rem;
  letter-spacing: 0.1em;
  padding: 0.22rem;
  cursor: pointer;
  border-radius: 2px;
  transition: color 0.12s, border-color 0.12s;
  text-align: center;
}
.corr-split-line-btn:hover { color: var(--ui); border-color: var(--ui); }

.corr-order-hint {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.56rem;
  color: var(--dim); opacity: 0.55;
  margin-bottom: 0.55rem;
  letter-spacing: 0.05em;
}
.corr-order-list { display: flex; flex-direction: column; gap: 0.28rem; }
.corr-order-item {
  display: flex; align-items: center; gap: 0.45rem;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0.32rem 0.45rem;
}
.corr-order-num {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; color: var(--ui);
  min-width: 1.1rem; flex-shrink: 0;
}
.corr-order-text {
  flex: 1; font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; color: var(--dim);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.corr-order-btns { display: flex; gap: 0.25rem; flex-shrink: 0; }
.corr-order-btns button {
  background: transparent; border: 1px solid var(--border);
  color: var(--dim); font-size: 0.62rem;
  padding: 0.12rem 0.28rem; border-radius: 2px;
  cursor: pointer; line-height: 1;
  transition: color 0.12s, border-color 0.12s;
}
.corr-order-btns button:hover { color: var(--ui); border-color: var(--ui); }

.corr-tl-text {
  font-family: 'Crimson Pro', serif;
  font-size: 0.9rem; color: var(--text); line-height: 1.5;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 3px; padding: 0.38rem 0.55rem;
  margin-top: 0.28rem;
}
.corr-footer {
  display: flex; gap: 0.45rem;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 0.5rem 0.65rem;
}
.corr-btn-retrans, .corr-btn-close {
  flex: 1; background: transparent;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.1em;
  padding: 0.42rem 0.5rem; border-radius: 3px; cursor: pointer;
  transition: background 0.12s, color 0.12s;
}
.corr-btn-retrans {
  border: 1px solid var(--ui); color: var(--ui);
}
.corr-btn-retrans:hover { background: rgba(0,229,176,0.09); }
.corr-btn-retrans:disabled { opacity: 0.38; cursor: not-allowed; }
.corr-btn-close {
  border: 1px solid var(--border); color: var(--dim);
}
.corr-btn-close:hover { color: var(--danger); border-color: var(--danger); }
</style>
</head>
<body>

<!-- ════════ HOME ════════ -->
<div id="screen-home" class="screen active">
  <div class="home-inner">

    <div class="home-brand">
      <div class="home-title">Manga<span>TL</span></div>
      <div class="home-sub">MangaDex · EasyOCR+CLAHE · Multi-Provider AI</div>
    </div>

    <div class="form-group">
      <label class="form-label" for="ai-model">AI Model</label>
      <select id="ai-model" class="form-input" onchange="onModelChange()">
        <optgroup label="✦ Free Tier (Google AI)">
          <option value="gemini|gemini-3.5-flash">Gemini 3.5 Flash — Free tier (best quality)</option>
          <option value="gemini|gemini-3.1-flash-lite">Gemini 3.1 Flash-Lite — Free tier (fastest, budget)</option>
          <option value="gemini|gemini-2.5-flash">Gemini 2.5 Flash — Paid</option>
        </optgroup>
        <optgroup label="Google AI (Paid)">
          <option value="gemini|gemini-3.1-pro-preview">Gemini 3.1 Pro — Paid (flagship)</option>
        </optgroup>
        <optgroup label="DeepSeek (Cheap ~$0.02–0.05/chapter)">
          <option value="deepseek|deepseek-v4-flash">DeepSeek V4 Flash</option>
          <option value="deepseek|deepseek-v4-pro">DeepSeek V4 Pro (best quality)</option>
        </optgroup>
      </select>
    </div>

    <div class="form-group">
      <label class="form-label" for="ai-key">API Key</label>
      <input id="ai-key" class="form-input" type="password"
             placeholder="AIza…" autocomplete="off" spellcheck="false">
    </div>

    <div class="form-group" id="vision-ocr-group">
      <label class="form-label" for="vision-ocr-mode">Vision OCR
        <span style="font-size:0.7rem;opacity:0.55;font-weight:400;margin-left:0.3rem">(uses Gemini quota per page)</span>
      </label>
      <select id="vision-ocr-mode" class="form-input"
              onchange="localStorage.setItem('mtl_vision_mode', this.value)">
        <option value="smart">Smart — complex scripts only (saves quota)</option>
        <option value="all">All languages — max quality</option>
        <option value="off">Off — EasyOCR only, no quota used</option>
      </select>
    </div>

    <div class="form-group">
      <label class="form-label" for="chapter-url">MangaDex Chapter URL</label>
      <input id="chapter-url" class="form-input" type="url" inputmode="url"
             placeholder="https://mangadex.org/chapter/…"
             autocomplete="off" spellcheck="false">
    </div>

    <div class="form-row-wrap">
      <div class="form-row">
        <div class="form-group">
          <label class="form-label" for="target-lang">Translate To</label>
          <select id="target-lang" class="form-input" onchange="onTargetLangChange()">
            <optgroup label="— Common —">
              <option value="English" selected>English</option>
              <option value="Malay">Bahasa Melayu — Malay</option>
              <option value="Indonesian">Bahasa Indonesia — Indonesian</option>
              <option value="Filipino">Tagalog — Filipino</option>
            </optgroup>
            <optgroup label="— East Asian —">
              <option value="Japanese">日本語 — Japanese</option>
              <option value="Chinese (Simplified)">中文（简体）— Chinese (Simplified)</option>
              <option value="Chinese (Traditional)">中文（繁體）— Chinese (Traditional)</option>
              <option value="Korean">한국어 — Korean</option>
            </optgroup>
            <optgroup label="— South / Southeast Asian —">
              <option value="Thai">ภาษาไทย — Thai</option>
              <option value="Vietnamese">Tiếng Việt — Vietnamese</option>
              <option value="Hindi">हिन्दी — Hindi</option>
              <option value="Bengali">বাংলা — Bengali</option>
              <option value="Tamil">தமிழ் — Tamil</option>
              <option value="Burmese">မြန်မာဘာသာ — Burmese</option>
              <option value="Khmer">ភាសាខ្មែរ — Khmer</option>
            </optgroup>
            <optgroup label="— European —">
              <option value="Spanish">Español — Spanish</option>
              <option value="French">Français — French</option>
              <option value="German">Deutsch — German</option>
              <option value="Portuguese">Português — Portuguese</option>
              <option value="Portuguese (Brazil)">Português (Brasil) — Portuguese (BR)</option>
              <option value="Italian">Italiano — Italian</option>
              <option value="Russian">Русский — Russian</option>
              <option value="Polish">Polski — Polish</option>
              <option value="Dutch">Nederlands — Dutch</option>
              <option value="Turkish">Türkçe — Turkish</option>
              <option value="Ukrainian">Українська — Ukrainian</option>
              <option value="Czech">Čeština — Czech</option>
              <option value="Romanian">Română — Romanian</option>
              <option value="Hungarian">Magyar — Hungarian</option>
              <option value="Swedish">Svenska — Swedish</option>
              <option value="Danish">Dansk — Danish</option>
              <option value="Norwegian">Norsk — Norwegian</option>
              <option value="Finnish">Suomi — Finnish</option>
              <option value="Greek">Ελληνικά — Greek</option>
            </optgroup>
            <optgroup label="— Middle East / Africa —">
              <option value="Arabic">العربية — Arabic</option>
              <option value="Persian">فارسی — Persian</option>
              <option value="Hebrew">עברית — Hebrew</option>
              <option value="Swahili">Kiswahili — Swahili</option>
            </optgroup>
            <option value="__custom__">✏ Custom language…</option>
          </select>
          <input id="target-lang-custom" class="form-input" type="text"
                 placeholder="Type language name, e.g. Javanese"
                 autocomplete="off" spellcheck="false"
                 style="display:none;margin-top:0.45rem">
        </div>
        <div class="form-group">
          <label class="form-label" for="quality">Image Quality</label>
          <select id="quality" class="form-input">
            <option value="data-saver" selected>Data Saver (faster)</option>
            <option value="data">Full Quality</option>
          </select>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label" for="merge-scale">
          Bubble Merge Sensitivity
        </label>
        <div class="merge-slider-row">
          <input id="merge-scale" type="range"
                 min="0.1" max="1.5" step="0.05" value="0.5"
                 oninput="document.getElementById('merge-scale-val').textContent = parseFloat(this.value).toFixed(2)">
          <span id="merge-scale-val" class="merge-slider-val">0.50</span>
        </div>
      </div>
    </div>

    <div>
      <span class="read-order-label">Badge Reading Order</span>
      <div class="read-order-row">
        <button class="read-order-btn active" id="ro-auto-rtl" onclick="setReadOrder('auto-rtl')" title="Right&#8594;Left then Top&#8594;Bottom (standard manga)">
          AUTO<br>&#8592; RTL
        </button>
        <button class="read-order-btn" id="ro-auto-ltr" onclick="setReadOrder('auto-ltr')" title="Left&#8594;Right then Top&#8594;Bottom (manhwa / webtoon)">
          AUTO<br>LTR &#8594;
        </button>
        <button class="read-order-btn" id="ro-manual" onclick="setReadOrder('manual')" title="Keep OCR order &#8212; reorder per-page manually">
          MANUAL<br>&#8597; DRAG
        </button>
      </div>
    </div>

    <button class="btn-go" onclick="startPipeline()">TRANSLATE CHAPTER</button>

    <div class="md-login-wrap" id="md-login-wrap">
      <button class="md-login-toggle" onclick="toggleMdLogin()">
        <span>MangaDex Account <span style="font-size:0.7rem;opacity:0.6">(removes 10-chapter guest limit)</span></span>
        <span style="display:flex;align-items:center;gap:0.6rem">
          <span class="md-status">
            <span class="md-status-dot" id="md-status-dot"></span>
            <span id="md-status-text">Guest</span>
          </span>
          <span class="arrow">▶</span>
        </span>
      </button>
      <div class="md-login-body">
        <div class="md-login-hint">
          Requires a <strong>personal API client</strong> on MangaDex.
          Create one at <a href="https://mangadex.org/settings" target="_blank">mangadex.org → Settings → API Clients</a>,
          then paste the Client ID and Secret below.
        </div>
        <div class="form-row" style="gap:0.6rem;margin-bottom:0.75rem">
          <div class="form-group" style="flex:1;margin-bottom:0">
            <label class="form-label" for="md-client-id">Client ID</label>
            <input id="md-client-id" class="form-input" type="text"
                   placeholder="…" autocomplete="off" spellcheck="false">
          </div>
          <div class="form-group" style="flex:1;margin-bottom:0">
            <label class="form-label" for="md-client-secret">Client Secret</label>
            <input id="md-client-secret" class="form-input" type="password"
                   placeholder="…" autocomplete="off" spellcheck="false">
          </div>
        </div>
        <div class="form-row" style="gap:0.6rem;margin-bottom:0.85rem">
          <div class="form-group" style="flex:1;margin-bottom:0">
            <label class="form-label" for="md-username">Username</label>
            <input id="md-username" class="form-input" type="text"
                   placeholder="your MangaDex username" autocomplete="off" spellcheck="false">
          </div>
          <div class="form-group" style="flex:1;margin-bottom:0">
            <label class="form-label" for="md-password">Password</label>
            <input id="md-password" class="form-input" type="password"
                   placeholder="••••••••" autocomplete="off">
          </div>
        </div>
        <button class="btn-md-login" id="md-login-btn" onclick="loginMangaDex()">Login</button>
      </div>
    </div>

    <div class="home-cache-strip" id="home-cache-strip">
      <div class="home-cache-info" id="home-cache-info">
        <span class="cache-empty">No chapters cached yet</span>
      </div>
      <button class="btn-clear-cache-home" id="btn-clear-cache-home"
              onclick="clearCacheFromHome()" disabled>🗑 CLEAR</button>
    </div>

    <div class="home-hint" id="ai-hint">
      Get a free key at <a id="ai-key-link" href="https://aistudio.google.com/app/apikey" target="_blank">aistudio.google.com</a>
      &nbsp;·&nbsp; English chapters display as-is
    </div>

  </div>
</div>

<!-- ════════ READER ════════ -->
<div id="screen-reader" class="screen">

  <div class="reader-header">
    <div class="reader-header-top">
      <button class="btn-back" onclick="goBack()">← BACK</button>
      <div class="reader-titles">
        <div class="reader-manga-title" id="manga-title">Loading…</div>
        <div class="reader-ch-info" id="chapter-info">—</div>
        <div class="reader-credit" id="chapter-credit"></div>
      </div>
    </div>
    <div class="reader-header-bottom">
      <div class="reader-status" id="reader-status">Fetching chapter…</div>
      <div class="cache-pill" id="reader-cache-pill" title="Translation cache">
        <span class="cache-pill-icon">💾</span>
        <span class="cache-pill-label" id="reader-cache-label"><span>0</span> cached</span>
        <button class="btn-cache-clear" id="reader-cache-clear-btn"
                onclick="clearCacheFromReader()" title="Clear all cached chapters">✕ clear</button>
      </div>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  </div>

  <div id="pages-container"></div>

  <div id="nav-bar" class="nav-bar">
    <button id="btn-prev" class="btn-nav" onclick="goToPrev()">← PREV CHAPTER</button>
    <button id="btn-next" class="btn-nav" onclick="goToNext()">NEXT CHAPTER →</button>
  </div>

</div>

<div id="toast"></div>

<script>
// ══════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════
let cancelled        = false;
let abortController  = null;
let toastTimer       = null;
let prevChapterId    = null;
let nextChapterId    = null;
let _activeChapterId = null;

// ── Badge reading order ────────────────────────
// 'auto-rtl' = right-to-left then top-to-bottom (manga default)
// 'auto-ltr' = left-to-right then top-to-bottom (manhwa / webtoon)
// 'manual'   = keep raw OCR order; per-page drag reorder available
let _readOrder = localStorage.getItem('mtl_read_order') || 'auto-rtl';

// Per-page manual order overrides: Map<"chId_pageIdx", number[]> (indices into original regions)
const _manualOrder = new Map();

// ══════════════════════════════════════════════
// LANGUAGE NAMES
// ══════════════════════════════════════════════
const LANG_NAMES = {
  vi: 'Vietnamese', it: 'Italian',    pt: 'Portuguese',
  'pt-br': 'Portuguese (BR)',                           // FIX #7
  ru: 'Russian',    fr: 'French',     es: 'Spanish',   de: 'German',
  pl: 'Polish',     nl: 'Dutch',      tr: 'Turkish',   id: 'Indonesian',
  ko: 'Korean',     ja: 'Japanese',   zh: 'Chinese',   'zh-hk': 'Chinese (Trad.)',
  th: 'Thai',       ar: 'Arabic',     uk: 'Ukrainian', cs: 'Czech',
  hu: 'Hungarian',  ro: 'Romanian',   sv: 'Swedish',   da: 'Danish',
  fi: 'Finnish',    no: 'Norwegian',  ms: 'Malay',     hr: 'Croatian',
  sk: 'Slovak',     bg: 'Bulgarian',  lt: 'Lithuanian', lv: 'Latvian',
  en: 'English',
};
function getLangName(code) {
  return LANG_NAMES[code?.toLowerCase()] ?? (code?.toUpperCase() ?? 'Unknown');
}

// ── Reading order control ─────────────────────
function setReadOrder(mode) {
  _readOrder = mode;
  localStorage.setItem('mtl_read_order', mode);
  ['auto-rtl', 'auto-ltr', 'manual'].forEach(m => {
    document.getElementById('ro-' + m)?.classList.toggle('active', m === mode);
  });
}

function _sortRegions(regions) {
  if (_readOrder === 'auto-ltr') {
    // Left-to-right, then top-to-bottom (manhwa)
    return [...regions].sort((a, b) => a.cy - b.cy || a.cx - b.cx);
  } else if (_readOrder === 'auto-rtl') {
    // Right-to-left, then top-to-bottom (traditional manga)
    return [...regions].sort((a, b) => a.cy - b.cy || b.cx - a.cx);
  }
  // manual — keep raw OCR order as returned by server
  return [...regions];
}

// ══════════════════════════════════════════════
// CACHE  (regions only — page URLs always re-fetched)
// ══════════════════════════════════════════════
const CACHE_PREFIX = 'mtl_ch_';
const CACHE_TTL    = 7 * 24 * 60 * 60 * 1000;
const CACHE_MAX    = 20;   // FIX #6: proactive entry-count cap

function getCachedChapter(chapterId) {
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + chapterId);
    if (!raw) return null;
    const cached = JSON.parse(raw);
    if (Date.now() - cached.timestamp > CACHE_TTL) {
      localStorage.removeItem(CACHE_PREFIX + chapterId);
      return null;
    }
    return cached;
  } catch { return null; }
}

function setCachedChapter(chapterId, data) {
  // FIX #6: proactively drop oldest entries when over the cap, then keep
  // evicting on quota errors until the write eventually succeeds.
  const getKeys = () => Object.keys(localStorage).filter(k => k.startsWith(CACHE_PREFIX));
  let keys = getKeys();
  while (keys.length >= CACHE_MAX) {
    if (!evictOldestCache()) break;
    keys = getKeys();
  }
  const entry = JSON.stringify({ ...data, timestamp: Date.now() });
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      localStorage.setItem(CACHE_PREFIX + chapterId, entry);
      return;
    } catch {
      if (!evictOldestCache()) return;  // nothing left to evict
    }
  }
}

// Returns the removed key, or null if nothing to evict
function evictOldestCache() {
  const keys = Object.keys(localStorage).filter(k => k.startsWith(CACHE_PREFIX));
  if (!keys.length) return null;
  let oldestKey = keys[0], oldestTime = Infinity;
  keys.forEach(k => {
    try {
      const d = JSON.parse(localStorage.getItem(k));
      if ((d.timestamp ?? 0) < oldestTime) { oldestTime = d.timestamp; oldestKey = k; }
    } catch {}
  });
  localStorage.removeItem(oldestKey);
  return oldestKey;
}

// ── Cache info helpers ────────────────────────
function _getCacheKeys() {
  return Object.keys(localStorage).filter(k => k.startsWith(CACHE_PREFIX));
}

function _getCacheSize() {
  const keys = _getCacheKeys();
  let bytes = 0;
  keys.forEach(k => { try { bytes += (localStorage.getItem(k) || '').length * 2; } catch {} });
  return { count: keys.length, bytes };
}

function _fmtBytes(b) {
  if (b < 1024)       return b + ' B';
  if (b < 1024*1024)  return (b/1024).toFixed(1) + ' KB';
  return (b/1024/1024).toFixed(2) + ' MB';
}

function refreshCacheUI() {
  const { count, bytes } = _getCacheSize();

  // ── Reader header pill ──
  const lbl = document.getElementById('reader-cache-label');
  const clrBtn = document.getElementById('reader-cache-clear-btn');
  if (lbl) {
    lbl.innerHTML = count > 0
      ? `<span>${count}</span> chapter${count !== 1 ? 's' : ''} · ${_fmtBytes(bytes)}`
      : `<span style="color:var(--dim);font-weight:normal">empty</span>`;
  }
  if (clrBtn) clrBtn.style.display = count > 0 ? '' : 'none';

  // ── Home screen strip ──
  const info   = document.getElementById('home-cache-info');
  const btn    = document.getElementById('btn-clear-cache-home');
  if (info) {
    if (count === 0) {
      info.innerHTML = '<span class="cache-empty">No chapters cached yet</span>';
    } else {
      info.innerHTML =
        `💾 <strong>${count}</strong> chapter${count !== 1 ? 's' : ''} cached` +
        ` &nbsp;·&nbsp; <strong>${_fmtBytes(bytes)}</strong>`;
    }
  }
  if (btn) btn.disabled = count === 0;
}

function clearCache() {
  const keys = _getCacheKeys();
  keys.forEach(k => localStorage.removeItem(k));
  const n = keys.length;
  toast(`Cleared ${n} cached chapter${n !== 1 ? 's' : ''}.`);
  refreshCacheUI();
}

function clearCacheFromHome()   { clearCache(); }
function clearCacheFromReader() {
  clearCache();
  // Also show confirmation near the pill
  const lbl = document.getElementById('reader-cache-label');
  if (lbl) {
    lbl.innerHTML = '<span style="color:var(--ui)">cleared ✓</span>';
    setTimeout(refreshCacheUI, 1800);
  }
}

function updatePageInCache(pageIdx, regions) {
  if (!_activeChapterId) return;
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + _activeChapterId);
    if (!raw) return;
    const cached = JSON.parse(raw);
    if (cached.pageRegions?.[pageIdx] !== undefined) {
      cached.pageRegions[pageIdx] = regions;
      localStorage.setItem(CACHE_PREFIX + _activeChapterId, JSON.stringify(cached));
    }
  } catch {}
}

// ══════════════════════════════════════════════
// CONCURRENCY  (3 pages in parallel)           // FIX #4: removed stale Gemini comment
// ══════════════════════════════════════════════
async function runConcurrent(taskFns, limit = 3) {
  let nextIdx = 0;
  async function worker() {
    while (nextIdx < taskFns.length && !cancelled) {
      const i = nextIdx++;
      await taskFns[i]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, taskFns.length) }, worker));
}

// ══════════════════════════════════════════════
// UI HELPERS
// ══════════════════════════════════════════════
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}
function toast(msg, dur = 6000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.style.display = 'none'; }, dur);
}
function setStatus(msg) { document.getElementById('reader-status').textContent = msg; }
function setProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
}
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function goBack() {
  cancelled = true;
  if (abortController) abortController.abort();
  // Release per-chapter memory — OCR data, corrections, manual order.
  _pageStore.clear();
  _manualOrder.clear();
  Object.keys(_corrWork).forEach(k => delete _corrWork[k]);
  Object.keys(_corrMode).forEach(k => delete _corrMode[k]);
  Object.keys(_corrSelId).forEach(k => delete _corrSelId[k]);
  Object.keys(_corrDraw).forEach(k => delete _corrDraw[k]);
  document.getElementById('pages-container').innerHTML = '';
  show('screen-home');
  refreshCacheUI();  // refresh home strip when returning
}

// ── Chapter navigation ──────────────────────
function updateNavButtons() {
  const bar = document.getElementById('nav-bar');
  const p   = document.getElementById('btn-prev');
  const n   = document.getElementById('btn-next');
  const has = prevChapterId || nextChapterId;
  bar.style.display     = has ? 'flex' : 'none';
  p.style.visibility    = prevChapterId ? 'visible' : 'hidden';
  p.style.pointerEvents = prevChapterId ? 'auto'    : 'none';
  n.style.visibility    = nextChapterId ? 'visible' : 'hidden';
  n.style.pointerEvents = nextChapterId ? 'auto'    : 'none';
}
function goToPrev() { if (prevChapterId) goToChapter(prevChapterId); }
function goToNext() { if (nextChapterId) goToChapter(nextChapterId); }
function goToChapter(chapterId) {
  if (!chapterId) return;
  window.scrollTo({ top: 0, behavior: 'instant' });
  document.getElementById('chapter-url').value = `https://mangadex.org/chapter/${chapterId}`;
  startPipelineWithId(chapterId);
}

// ══════════════════════════════════════════════
// MANGADEX API  (routed via local proxy)
// ══════════════════════════════════════════════
function parseChapterId(url) {
  const m = url.match(/chapter\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i);
  return m ? m[1] : null;
}

async function fetchChapterMeta(id, signal) {
  const authHeaders = await getMdHeaders();
  const r = await fetch(`/mangadex/chapter/${id}?includes[]=manga&includes[]=scanlation_group`, { signal, headers: authHeaders });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j?.errors?.[0]?.detail || `MangaDex error ${r.status}`);
  }
  const { data } = await r.json();
  const attrs    = data.attributes;
  const mangaRel = data.relationships.find(x => x.type === 'manga');
  const titles   = mangaRel?.attributes?.title ?? {};

  // Collect all scanlation groups (a chapter can have more than one)
  const groups = data.relationships
    .filter(x => x.type === 'scanlation_group' && x.attributes?.name)
    .map(x => ({ name: x.attributes.name, id: x.id }));

  return {
    mangaTitle:         titles.en ?? Object.values(titles)[0] ?? 'Unknown Manga',
    mangaId:            mangaRel?.id ?? null,
    chapter:            attrs.chapter ?? '?',
    chapterTitle:       attrs.title   ?? '',
    volume:             attrs.volume  ?? null,
    translatedLanguage: attrs.translatedLanguage ?? 'en',
    groups,
  };
}

// FIX #12: returns {cdn, img} pairs instead of plain strings.
//   cdn  — raw CDN HTTPS URL, used by /ocr (must be HTTPS for the proxy to accept)
//   img  — routed through /proxy so all image traffic goes through the local server
async function fetchPageUrls(id, quality, signal) {
  const authHeaders = await getMdHeaders();
  const r = await fetch(`/mangadex/at-home/server/${id}`, { signal, headers: authHeaders });
  if (!r.ok) throw new Error(`Failed to get page server: ${r.status}`);
  const { baseUrl, chapter } = await r.json();
  let files, tier;
  if (quality === 'data-saver' && chapter.dataSaver?.length) {
    files = chapter.dataSaver; tier = 'data-saver';
  } else {
    files = chapter.data; tier = 'data';
  }
  return files.map(f => {
    const cdn = `${baseUrl}/${tier}/${chapter.hash}/${f}`;
    return { cdn, img: `/proxy?url=${encodeURIComponent(cdn)}` };
  });
}

// FIX #5: paginate with offset instead of assuming 500 covers everything.
//   Handles manga with 500+ chapters per language without silently missing
//   the adjacent-chapter lookup.
async function fetchAdjacentChapters(mangaId, currentId, lang, signal) {
  const LIMIT = 500;
  const base  = `/mangadex/manga/${mangaId}/feed`
    + `?translatedLanguage[]=${lang}&order[chapter]=asc&limit=${LIMIT}`
    + `&contentRating[]=safe&contentRating[]=suggestive`
    + `&contentRating[]=erotica&contentRating[]=pornographic`;
  const authHeaders = await getMdHeaders();
  try {
    let all = [], offset = 0, total = Infinity;
    while (offset < total) {
      const r = await fetch(`${base}&offset=${offset}`, { signal, headers: authHeaders });
      if (!r.ok) return { prev: null, next: null };
      const body = await r.json();
      total  = body.total ?? 0;
      if (!body.data?.length) break;
      all    = all.concat(body.data);
      offset += body.data.length;
    }
    const idx = all.findIndex(ch => ch.id === currentId);
    if (idx === -1) return { prev: null, next: null };
    return {
      prev: idx > 0            ? all[idx - 1].id : null,
      next: idx < all.length-1 ? all[idx + 1].id : null,
    };
  } catch { return { prev: null, next: null }; }
}

// ══════════════════════════════════════════════
// OCR  (runs on local proxy — no rate limit)
// ══════════════════════════════════════════════
// ── Per-page runtime store ─────────────────────────────────────────────────
// Keyed by `${chapterId}_${pageIdx}`.  Holds cdnUrl, imgSrc, sourceLang,
// rawBoxes (pre-merge fragments), and autoRegions from the last OCR run.
const _pageStore = new Map();

async function ocrPage(cdnUrl, lang, signal) {
  const marginScale = parseFloat(document.getElementById('merge-scale')?.value ?? '0.5');
  const info    = getModelInfo();
  // Pass the Gemini key + model so the proxy can use Vision OCR.
  // For DeepSeek users with no Gemini key, Vision OCR is skipped and
  // EasyOCR is used as normal.
  const aiKey   = info.provider === 'gemini'
    ? (document.getElementById('ai-key')?.value?.trim() ?? '')
    : '';
  const aiModel = info.provider === 'gemini' ? getModelId() : '';
  // vision_mode controls when Vision OCR fires:
  //   'smart' — only for complex scripts (ja/zh/ko/ar/th)  ← saves free-tier quota
  //   'all'   — every language
  //   'off'   — never (EasyOCR only)
  // DeepSeek users always get 'off' regardless of the select value.
  const visionMode = info.provider === 'gemini'
    ? (document.getElementById('vision-ocr-mode')?.value ?? 'smart')
    : 'off';

  const r = await fetch('/ocr', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url: cdnUrl, lang, margin_scale: marginScale,
      ai_key: aiKey, ai_model: aiModel, vision_mode: visionMode,
    })
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err?.description || `OCR error ${r.status}`);
  }
  const data = await r.json();
  return {
    regions:  data.regions   ?? [],
    rawBoxes: data.raw_boxes ?? [],
  };
}

// ══════════════════════════════════════════════
// ══════════════════════════════════════════════
// MANGADEX AUTH
// ══════════════════════════════════════════════

let _mdAccessToken  = null;
let _mdRefreshToken = null;
let _mdTokenExpiry  = 0;       // ms timestamp
let _mdClientId     = '';
let _mdClientSecret = '';
let _mdUsername     = '';

function toggleMdLogin() {
  document.getElementById('md-login-wrap').classList.toggle('open');
}

function _setMdStatus(loggedIn, username = '') {
  document.getElementById('md-status-dot').className =
    'md-status-dot' + (loggedIn ? ' online' : '');
  document.getElementById('md-status-text').textContent =
    loggedIn ? `Logged in as ${username}` : 'Guest';
  const btn = document.getElementById('md-login-btn');
  btn.textContent = loggedIn ? 'Logout' : 'Login';
  btn.onclick     = loggedIn ? logoutMangaDex : loginMangaDex;
}

function _saveMdTokens(accessToken, refreshToken, expiresIn, username) {
  _mdAccessToken  = accessToken;
  _mdRefreshToken = refreshToken;
  _mdTokenExpiry  = Date.now() + expiresIn * 1000;
  _mdUsername     = username;
  localStorage.setItem('mtl_md_access',   accessToken);
  localStorage.setItem('mtl_md_refresh',  refreshToken);
  localStorage.setItem('mtl_md_expiry',   String(_mdTokenExpiry));
  localStorage.setItem('mtl_md_username', username);
}

function _clearMdTokens() {
  _mdAccessToken = _mdRefreshToken = null;
  _mdTokenExpiry = 0; _mdUsername = '';
  ['mtl_md_access','mtl_md_refresh','mtl_md_expiry','mtl_md_username'].forEach(k =>
    localStorage.removeItem(k));
}

// Returns a valid access token, refreshing first if needed. Returns null if not logged in.
async function getMdToken() {
  if (!_mdAccessToken) return null;
  if (Date.now() < _mdTokenExpiry - 60_000) return _mdAccessToken;  // still valid
  // Try to refresh
  try {
    const res = await fetch('/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        refresh_token: _mdRefreshToken,
        client_id:     _mdClientId,
        client_secret: _mdClientSecret,
      }),
    });
    if (!res.ok) { _clearMdTokens(); _setMdStatus(false); return null; }
    const d = await res.json();
    _saveMdTokens(d.access_token, d.refresh_token, d.expires_in, _mdUsername);
    return _mdAccessToken;
  } catch {
    return _mdAccessToken;  // network hiccup — try with old token
  }
}

// Returns headers object with Authorization if logged in, otherwise just User-Agent.
async function getMdHeaders() {
  const token = await getMdToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function loginMangaDex() {
  const clientId     = document.getElementById('md-client-id').value.trim();
  const clientSecret = document.getElementById('md-client-secret').value.trim();
  const username     = document.getElementById('md-username').value.trim();
  const password     = document.getElementById('md-password').value.trim();
  if (!clientId || !clientSecret || !username || !password) {
    toast('Fill in all four MangaDex fields.'); return;
  }
  const btn = document.getElementById('md-login-btn');
  btn.disabled = true; btn.textContent = 'Logging in…';
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, client_id: clientId, client_secret: clientSecret }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error_description || err?.error || `Login failed (${res.status})`);
    }
    const d = await res.json();
    _mdClientId     = clientId;
    _mdClientSecret = clientSecret;
    _saveMdTokens(d.access_token, d.refresh_token, d.expires_in, username);
    localStorage.setItem('mtl_md_client_id',     clientId);
    localStorage.setItem('mtl_md_client_secret', clientSecret);
    _setMdStatus(true, username);
    // Clear password field — don't persist it
    document.getElementById('md-password').value = '';
    toast('MangaDex login successful ✓');
  } catch (e) {
    toast(`Login failed: ${e.message}`);
    btn.disabled = false; btn.textContent = 'Login';
  }
}

function logoutMangaDex() {
  _clearMdTokens();
  _setMdStatus(false);
  toast('Logged out of MangaDex.');
}

// ══════════════════════════════════════════════
// TRANSLATION  (multi-provider — proxied server-side)
// ══════════════════════════════════════════════

// FIX #2: valid type set used to sanitise AI classification output
const VALID_TEXT_TYPES = new Set(['speech', 'thought', 'sfx', 'narration', 'sign']);

// Model registry — value format: "provider|model-id"
const MODEL_INFO = {
  // Gemini models (free tier)
  'gemini|gemini-3.5-flash':              { provider: 'gemini',   placeholder: 'AIza…', label: 'Gemini 3.5 Flash',       keyUrl: 'https://aistudio.google.com/app/apikey', keySite: 'aistudio.google.com' },
  'gemini|gemini-3.1-flash-lite':         { provider: 'gemini',   placeholder: 'AIza…', label: 'Gemini 3.1 Flash-Lite',  keyUrl: 'https://aistudio.google.com/app/apikey', keySite: 'aistudio.google.com' },
  'gemini|gemini-2.5-flash':              { provider: 'gemini',   placeholder: 'AIza…', label: 'Gemini 2.5 Flash',       keyUrl: 'https://aistudio.google.com/app/apikey', keySite: 'aistudio.google.com' },
  // Gemini models (paid flagship)
  'gemini|gemini-3.1-pro-preview':        { provider: 'gemini',   placeholder: 'AIza…', label: 'Gemini 3.1 Pro',         keyUrl: 'https://aistudio.google.com/app/apikey', keySite: 'aistudio.google.com' },
  // DeepSeek models
  'deepseek|deepseek-v4-flash':           { provider: 'deepseek', placeholder: 'sk-…',  label: 'DeepSeek V4 Flash',      keyUrl: 'https://platform.deepseek.com',          keySite: 'platform.deepseek.com' },
  'deepseek|deepseek-v4-pro':             { provider: 'deepseek', placeholder: 'sk-…',  label: 'DeepSeek V4 Pro',        keyUrl: 'https://platform.deepseek.com',          keySite: 'platform.deepseek.com' },
};

function getModelInfo() {
  const val = document.getElementById('ai-model')?.value || 'gemini|gemini-3.5-flash';
  return MODEL_INFO[val] || MODEL_INFO['gemini|gemini-3.5-flash'];
}

function getModelId() {
  const val = document.getElementById('ai-model')?.value || '';
  return val.split('|')[1] || 'gemini-3.5-flash';
}

function onModelChange() {
  const info     = getModelInfo();
  const keyEl    = document.getElementById('ai-key');
  const linkEl   = document.getElementById('ai-key-link');
  const hintEl   = document.getElementById('ai-hint');

  // Save current key under the OLD provider before switching
  if (keyEl) {
    const prevProvider = keyEl.dataset.provider;
    if (prevProvider && keyEl.value.trim()) {
      localStorage.setItem(`mtl_key_${prevProvider}`, keyEl.value.trim());
    }
  }

  // Load saved key for the NEW provider
  const savedForProvider = localStorage.getItem(`mtl_key_${info.provider}`);
  if (keyEl) {
    keyEl.placeholder       = info.placeholder;
    keyEl.dataset.provider  = info.provider;
    keyEl.value             = savedForProvider || '';
  }

  if (linkEl) { linkEl.href = info.keyUrl; linkEl.textContent = info.keySite; }

  if (hintEl) {
    const freeTierModels = ['gemini|gemini-3.5-flash', 'gemini|gemini-3.1-flash-lite'];
    const modelVal = document.getElementById('ai-model')?.value || '';
    const freeNote = freeTierModels.includes(modelVal)
      ? ' (free tier — no credit card needed)'
      : info.provider === 'gemini' ? ' (paid — billing required)' : ' (~$0.02–0.05/chapter)';
    hintEl.textContent = `Get a free key at `;
    const a = document.createElement('a');
    a.id = 'ai-key-link'; a.href = info.keyUrl;
    a.target = '_blank'; a.textContent = info.keySite;
    hintEl.appendChild(a);
    hintEl.appendChild(document.createTextNode(freeNote));
  }

  // Show Vision OCR mode only for Gemini; hide entirely for DeepSeek
  const visionGroup = document.getElementById('vision-ocr-group');
  if (visionGroup) {
    const isGemini = info.provider === 'gemini';
    visionGroup.style.display = isGemini ? '' : 'none';
    // If switching TO a free-tier Gemini model, nudge toward 'smart' to protect quota —
    // but only if the user hasn't explicitly changed it from the default themselves.
    const freeTierModels2 = ['gemini|gemini-3.5-flash', 'gemini|gemini-3.1-flash-lite'];
    const modeEl = document.getElementById('vision-ocr-mode');
    if (isGemini && modeEl && !localStorage.getItem('mtl_vision_mode')) {
      modeEl.value = freeTierModels2.includes(document.getElementById('ai-model')?.value || '')
        ? 'smart' : 'smart';  // default 'smart' for all models until user changes it
    }
  }

  localStorage.setItem('mtl_ai_model', document.getElementById('ai-model').value);
}

// ── Target language dropdown ──────────────────
function onTargetLangChange() {
  const sel    = document.getElementById('target-lang');
  const custom = document.getElementById('target-lang-custom');
  const isCustom = sel.value === '__custom__';
  custom.style.display = isCustom ? 'block' : 'none';
  if (isCustom) { custom.focus(); }
  // Persist selection (value, not display label)
  localStorage.setItem('mtl_target_lang', sel.value);
}

// Returns the effective target language string for the AI prompt
function getTargetLang() {
  const sel = document.getElementById('target-lang');
  if (sel.value === '__custom__') {
    const customEl = document.getElementById('target-lang-custom');
    return (customEl.value.trim()) || localStorage.getItem('mtl_target_lang_custom') || 'English';
  }
  return sel.value || 'English';
}

// POST to /translate instead of calling provider APIs directly.
// The API key is forwarded by the proxy and never appears in DevTools.
// translateBatch accepts regions [{text,cx,cy}] — cx/cy help the AI
// understand panel layout and infer reading order.
async function translateBatch(regions, sourceLang, targetLang, signal) {
  if (!regions.length) return [];
  const key      = document.getElementById('ai-key').value.trim();
  const info     = getModelInfo();
  const modelId  = getModelId();
  if (!key) throw new Error(`${info.label} API key not set.`);

  // Attach index so the model can return items in any order and we re-map correctly
  const items = regions.map((r, i) => ({
    i,
    text: r.text,
    cx: r.cx,   // left–right position (0 = left edge, 100 = right edge)
    cy: r.cy,   // top–bottom position (0 = top, 100 = bottom)
  }));

  // ── Hybrid noise filter ─────────────────────────────────────────────────────
  // Single-character OCR detections are almost always screentone patterns,
  // stray marks, or EasyOCR false positives — not real text. Sending them to
  // the AI wastes tokens and triggers hallucinated "translations" of garbage.
  // Filter them out before the API call; their `out` slots get pre-filled
  // as { tl: '—', t: 'sfx' } below so the array length stays consistent.
  // Note: the `i` values in meaningfulItems still reference original positions
  // in `regions`, so the index-based re-mapping in the response loop is safe.
  const meaningfulItems = items.filter(it => it.text.trim().length >= 2);
  const sendItems = meaningfulItems.length ? meaningfulItems : items;

  const res = await fetch('/translate', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      provider:    info.provider,
      key,
      source_lang: sourceLang,   // proxy uses this to inject lang-specific hints
      payload: {
        model:       modelId,
        temperature: 0.3,
        max_tokens:  4000,   // raised from 3000 — Gemini 2.5+ thinking was consuming tokens
                             // before the JSON, leaving too little budget for long pages
        // DeepSeek JSON mode enforces a valid JSON object in the response.
        // Gemini ignores this field (the proxy strips it before forwarding).
        ...(info.provider === 'deepseek' ? { response_format: { type: 'json_object' } } : {}),
        messages: [
          {
            role: 'system',
            content:
              `You are a manga translation expert. Translate ${sendItems.length} OCR-extracted text regions ` +
              `from ${getLangName(sourceLang)} to ${targetLang}.\n\n` +
              `SPATIAL DATA: Each item has cx (left-right % 0–100) and cy (top-bottom % 0–100).\n` +
              `Use these to reconstruct reading order. Pages often have LEFT and RIGHT column panels — ` +
              `items at similar cy but very different cx belong to DIFFERENT panels and should not be mixed.\n` +
              `Within a single panel/column, read top-to-bottom (ascending cy).\n\n` +
              `OCR ARTIFACTS TO FIX BEFORE TRANSLATING:\n` +
              `- Words split with a hyphen (e.g. "PREGI-" followed by "DENTAL") → merge into one word ("PRESIDENTIAL")\n` +
              `- A single speech bubble split into 2–3 nearby fragments → join them into one natural sentence\n` +
              `- Stray single characters or obvious OCR noise → clean up or skip\n` +
              `- ALL-CAPS OCR input is normal; translate into natural mixed-case output\n\n` +
              `For each item classify the text type:\n` +
              `  speech    — dialogue in speech bubbles\n` +
              `  thought   — internal monologue (cloud / wavy bubbles)\n` +
              `  sfx       — sound effects, onomatopoeia\n` +
              `  narration — caption boxes, story narration\n` +
              `  sign      — signs, labels, written environmental text\n\n` +
              `SFX RULE: If a region is clearly an SFX or onomatopoeia, translate it as a brief English ` +
              `sound effect wrapped in asterisks (e.g. *Rumble*, *Crash*, *Sigh*) — do NOT return "-" for these.\n` +
              `Return ONLY a JSON object with a "translations" key containing exactly ${sendItems.length} items, ` +
              `preserving the original i values:\n` +
              `{"translations":[{"i":0,"tl":"translated text","t":"type"},...]}\n` +
              `If a region is pure noise with no translatable meaning, set tl to "-".\n` +
              `No markdown fences, no explanation, no extra keys.`
          },
          { role: 'user', content: JSON.stringify(sendItems) }
        ]
      }
    })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `${info.label} error ${res.status}`);
  }

  const data  = await res.json();
  const text  = data.choices?.[0]?.message?.content ?? '';
  const clean = text.replace(/```(?:json)?\n?/g, '').replace(/```/g, '').trim();

  // ── Diagnostic — remove once translations are working reliably ───────────
  console.log('[TL] raw length:', text.length,
              '| first 400 chars of clean:', clean.slice(0, 400));
  // ────────────────────────────────────────────────────────────────────────

  // Guard: an empty response means the proxy's upstream AI produced nothing
  // (safety block, bad model ID, etc.).  Throw so the page renders as a
  // retryable error card rather than silently showing all-"—" translations.
  if (!clean) {
    throw new Error(`${info.label} returned an empty response — check your API key / model, then retry.`);
  }
  // Accept {"translations":[...]} (DeepSeek JSON mode) or a bare array (Gemini/fallback).
  // Parse strategy:
  //   1. Try full JSON.parse — handles {"translations":[...]} and bare [...] directly.
  //   2. Regex fallback runs whenever parsedArr is STILL null after step 1,
  //      including when JSON.parse SUCCEEDS but the model used a different key name
  //      (e.g. "result", "output") — the regex finds the inner [...] regardless.
  let parsedArr = null;
  try {
    const top = JSON.parse(clean);
    if (Array.isArray(top))                        parsedArr = top;
    else if (Array.isArray(top?.translations))      parsedArr = top.translations;
  } catch { /* not valid top-level JSON — fall through to regex */ }
  // Regex fallback — runs whenever parsedArr is still null after step 1.
  // Strategy: look for "translations":[ ... ] specifically first (more precise),
  // then fall back to any bare [...] as a last resort.
  // The generic /\[[\s\S]*\]/ is INTENTIONALLY avoided as the first option because
  // it is greedy and breaks when the model adds ANY "[" before the array in its reply
  // (e.g. "Based on these [15] regions..." would grab from that "[" to the last "]").
  if (!parsedArr) {
    // Targeted: find the array that follows the "translations" key
    const tlM = clean.match(/"translations"\s*:\s*\[/);
    if (tlM) {
      const arrStart = tlM.index + tlM[0].length - 1;   // position of '['
      const arrEnd   = clean.lastIndexOf(']');
      if (arrEnd > arrStart) {
        try { parsedArr = JSON.parse(clean.slice(arrStart, arrEnd + 1)); } catch {}
      }
    }
  }
  if (!parsedArr) {
    // Generic last resort — any [...] in the response
    const m = clean.match(/\[[\s\S]*\]/);
    if (m) { try { parsedArr = JSON.parse(m[0]); } catch {} }
  }
  const fallback = () => regions.map(() => ({ tl: '—', t: 'speech' }));
  if (!parsedArr) {
    // All three parse strategies failed — the model returned something that is
    // not valid JSON at all. Log the full response and throw so the pipeline
    // renders a retryable error card instead of silently showing all-"—" bubbles.
    const preview = clean.slice(0, 300);
    console.error('[TL] All JSON parse strategies failed. Full response:', clean);
    throw new Error(
      `AI response could not be parsed as JSON (all 3 strategies failed).\n` +
      `Response preview: ${preview || '(empty)'}\n` +
      `Try a different model or retry — the model may have returned plain text.`
    );
  }
  try {
    const parsed = parsedArr;
    if (!Array.isArray(parsed)) {
      // JSON parsed but is an object without a translations array — unexpected schema.
      console.error('[TL] Unexpected response schema (not array):', parsed);
      throw new Error(
        `AI returned unexpected JSON schema (expected array, got ${typeof parsed}). Retry the page.`
      );
    }
    // Map back to original indices so overlay positions stay correct.
    // BUG FIX: some models return "i" as a quoted string (e.g. "i":"0") rather
    // than an integer.  parseInt handles both; the old `typeof item.i === 'number'`
    // check would silently drop every item and produce all-"—" translations.
    // Noise slots (single-char items filtered before the API call) are pre-filled
    // as sfx so they render as small red badges rather than empty speech bubbles.
    const out = regions.map((_, i) =>
      items[i].text.trim().length < 2
        ? { tl: '—', t: 'sfx' }
        : { tl: '—', t: 'speech' }
    );
    parsed.forEach(item => {
      if (typeof item === 'string') return; // model ignored schema — skip
      const idx = parseInt(String(item.i ?? ''), 10);
      if (isNaN(idx) || idx < 0 || idx >= regions.length) return;
      out[idx] = {
        tl: String(item.tl ?? item.translation ?? item.text ?? '—'),
        t:  VALID_TEXT_TYPES.has(item.t) ? item.t : 'speech',
      };
    });
    return out;
  } catch { return fallback(); }
}

// ══════════════════════════════════════════════
// RENDERING
// ══════════════════════════════════════════════
function addSkeleton(i) {
  const el     = document.createElement('div');
  el.className = 'page-card';
  el.id        = `page-${i}`;
  el.innerHTML = `<div class="page-skeleton">
    <span class="sk-num">${i + 1}</span>
    <div class="sk-bar"></div>
  </div>`;
  document.getElementById('pages-container').appendChild(el);
  return el;
}

// Display-only: no translation panel (English chapters / full-art pages)
function renderPageDisplay(el, pageIdx, total, imgSrc) {
  el.innerHTML = `
    <div class="img-wrap">
      <img src="${esc(imgSrc)}" class="page-img page-img-only"
           loading="lazy" alt="Page ${pageIdx + 1}">
      <div class="pg-label">${pageIdx + 1} / ${total}</div>
    </div>`;
}

// Full render: image + numbered badges + translation panel
function renderPage(el, pageIdx, total, imgSrc, regions) {
  // Apply any stored manual reorder for this page
  const moKey = `${_activeChapterId}_${pageIdx}`;
  const moIdx = _manualOrder.get(moKey);
  const displayRegions = moIdx
    ? moIdx.map(i => regions[i]).filter(Boolean)
    : regions;

  let badgesHtml = '';
  let rowsHtml   = '';
  displayRegions.forEach((r, i) => {
    const tag = (r.t || 'speech').toLowerCase();
    const cx  = r.x ?? 50;
    const cy  = r.y ?? 50;
    badgesHtml += `<div class="badge t-${tag}" style="left:${cx}%;top:${cy}%">${i + 1}</div>`;
    rowsHtml   += `<div class="t-row">
      <span class="t-num">${i + 1}</span>
      <span class="t-tag ${tag}">${tag}</span>
      <span class="t-text">${esc(r.tl || '—')}</span>
    </div>`;
  });

  const hasRegions = displayRegions.length > 0;
  const reorderBtn = hasRegions
    ? `<button class="btn-reorder-page" id="ro-btn-${pageIdx}"
         onclick="toggleReorderPanel(${pageIdx})" title="Manually reorder badges">⇅ ORDER</button>`
    : '';
  const panel = hasRegions
    ? `<div class="trans-panel" id="trans-panel-${pageIdx}">${rowsHtml}</div>`
    : `<div class="no-text-note">— no text detected —</div>`;

  el.innerHTML = `
    <div class="img-wrap">
      <img src="${esc(imgSrc)}" class="page-img" loading="lazy" alt="Page ${pageIdx + 1}">
      ${badgesHtml}
      <div class="pg-label">${pageIdx + 1} / ${total}</div>
      <button class="btn-correct" onclick="openCorrection(${pageIdx})" title="Correct this page">✏ CORRECT</button>
      ${reorderBtn}
    </div>
    ${panel}
    <div class="reorder-panel" id="reorder-panel-${pageIdx}" style="display:none"></div>`;

  // Store regions on the element for reorder access
  el._regions = regions;
}

// FIX #12: cdnUrl (for OCR retries) and imgSrc (for display) are now separate.
//          data-cdn / data-img are stored on the button so retryPage can use each correctly.
function renderPageError(el, pageIdx, total, cdnUrl, imgSrc, errMsg, sourceLang) {
  el.innerHTML = `
    <div class="img-wrap">
      <img src="${esc(imgSrc)}" class="page-img" loading="lazy" alt="Page ${pageIdx + 1}">
      <div class="pg-label">${pageIdx + 1} / ${total}</div>
    </div>
    <div class="page-err-note">
      <span>${esc(errMsg)}</span>
      <button class="btn-retry"
        data-idx="${pageIdx}"
        data-total="${total}"
        data-cdn="${esc(cdnUrl)}"
        data-img="${esc(imgSrc)}"
        data-lang="${esc(sourceLang ?? '')}">↺ Retry</button>
    </div>`;
}

document.addEventListener('click', e => {
  const btn = e.target.closest('.btn-retry');
  if (!btn || btn.disabled) return;
  const el = btn.closest('.page-card');
  retryPage(btn, el,
    +btn.dataset.idx, +btn.dataset.total,
    btn.dataset.cdn, btn.dataset.img, btn.dataset.lang);
});

// FIX #2: regions now use translated[j].t for type (was always hardcoded 'speech')
// FIX #12: uses cdnUrl for OCR, imgSrc for display
async function retryPage(btn, el, pageIdx, total, cdnUrl, imgSrc, sourceLang) {
  btn.disabled    = true;
  btn.textContent = 'Retrying…';
  const targetLang = getTargetLang();
  try {
    const ocrData    = await ocrPage(cdnUrl, sourceLang);
    const ocrResult  = ocrData.regions;
    // Store raw data so the correction UI can access it
    _pageStore.set(`${_activeChapterId}_${pageIdx}`, {
      cdnUrl, imgSrc, sourceLang, total,
      rawBoxes: ocrData.rawBoxes,
      autoRegions: ocrResult,
    });
    // Sort regions per user's reading order preference
    const sortedOcr = _sortRegions(ocrResult);
    const translated = await translateBatch(sortedOcr, sourceLang, targetLang);
    const regions    = sortedOcr.map((r, j) => ({
      t:  translated[j]?.t  || 'speech',
      x:  r.cx,
      y:  r.cy,
      box: r.box,
      tl: translated[j]?.tl || '—',
    }));
    // Save translated data so correction UI gets real tl values
    const _se = _pageStore.get(`${_activeChapterId}_${pageIdx}`);
    if (_se) _se.sortedRegions = sortedOcr.map((r, j) => ({
      text: r.text || '', t: translated[j]?.t || 'speech',
      cx: r.cx, cy: r.cy, box: r.box,
      raw_box_ids: r.raw_box_ids || [],
      tl: translated[j]?.tl || '—',
    }));
    renderPage(el, pageIdx, total, imgSrc, regions);
    updatePageInCache(pageIdx, regions);
  } catch (err) {
    btn.disabled    = false;
    btn.textContent = '↺ Retry';
    // Update the error note in-place so the user can read why the retry failed
    // without relying on the disappearing toast alone.
    const errNote = el?.querySelector('.page-err-note span');
    if (errNote) errNote.textContent = err.message;
    toast(`Retry failed: ${err.message}`);
  }
}

// ══════════════════════════════════════════════
// MAIN PIPELINE
// ══════════════════════════════════════════════
async function startPipeline() {
  const key        = document.getElementById('ai-key').value.trim();
  const rawUrl     = document.getElementById('chapter-url').value.trim();
  const targetLang = getTargetLang();
  const quality    = document.getElementById('quality').value;
  const info       = getModelInfo();

  if (!key) { toast(`Enter your ${info.label} API key.`); return; }

  // Validate key format matches the selected provider
  const keyIsGemini   = key.startsWith('AIza');
  const keyIsDeepSeek = key.startsWith('sk-');
  if (info.provider === 'gemini' && keyIsDeepSeek) {
    toast('That looks like a DeepSeek key (sk-…).\nGemini keys start with AIza — get one at aistudio.google.com');
    return;
  }
  if (info.provider === 'deepseek' && keyIsGemini) {
    toast('That looks like a Gemini key (AIza…).\nDeepSeek keys start with sk- — get one at platform.deepseek.com');
    return;
  }

  if (!rawUrl) { toast('Paste a MangaDex chapter URL.'); return; }

  const chapterId = parseChapterId(rawUrl);
  if (!chapterId) {
    toast("Could not find a chapter ID.\nMake sure it's a mangadex.org/chapter/… link.");
    return;
  }
  localStorage.setItem(`mtl_key_${info.provider}`, key);
  startPipelineWithId(chapterId, quality, targetLang);
}

async function startPipelineWithId(chapterId, quality, targetLang) {
  quality    = quality    || document.getElementById('quality').value;
  targetLang = targetLang || getTargetLang();

  cancelled = false;
  if (abortController) abortController.abort();
  abortController = new AbortController();
  const signal = abortController.signal;

  _activeChapterId = chapterId;
  prevChapterId    = null;
  nextChapterId    = null;

  show('screen-reader');
  refreshCacheUI();  // update pill count when entering reader
  document.getElementById('pages-container').innerHTML = '';
  document.getElementById('manga-title').textContent   = 'Loading…';
  document.getElementById('chapter-info').textContent  = '';
  updateNavButtons();
  setProgress(0, 1);
  setStatus('Fetching chapter info…');

  try {
    // ── 1. Chapter meta ───────────────────────
    const meta       = await fetchChapterMeta(chapterId, signal);
    const sourceLang = meta.translatedLanguage;
    const isEnglish  = sourceLang === 'en';

    document.getElementById('manga-title').textContent = meta.mangaTitle;
    document.getElementById('chapter-info').textContent =
      `Ch. ${meta.chapter}${meta.chapterTitle ? ' · ' + meta.chapterTitle : ''}` +
      (meta.volume ? `  (Vol. ${meta.volume})` : '') +
      (isEnglish ? '' : `  ·  ${getLangName(sourceLang)} → ${targetLang}`);

    // Scanlation group credit
    const creditEl = document.getElementById('chapter-credit');
    if (meta.groups.length) {
      const links = meta.groups.map(g =>
        `<a href="https://mangadex.org/group/${g.id}" target="_blank" rel="noopener">${g.name}</a>`
      ).join(' &amp; ');
      creditEl.innerHTML = `Translated by ${links}`;
    } else {
      creditEl.textContent = '';
    }

    // ── 2. Page URLs ──────────────────────────
    // FIX #12: urls is now [{cdn, img}] — cdn for OCR, img for <img> display
    setStatus('Loading page list…');
    const urls  = await fetchPageUrls(chapterId, quality, signal);
    const total = urls.length;
    if (total === 0) throw new Error('No pages found for this chapter.');

    setProgress(0, total);
    const skeletons = urls.map((_, i) => addSkeleton(i));

    function resolveAdjacentChapters() {
      if (!meta.mangaId) return;
      const startedId = chapterId;
      fetchAdjacentChapters(meta.mangaId, chapterId, sourceLang, signal)
        .then(({ prev, next }) => {
          if (_activeChapterId !== startedId || cancelled) return;
          prevChapterId = prev; nextChapterId = next;
          updateNavButtons();
        });
    }

    // ── 3a. English — display only ────────────
    if (isEnglish) {
      urls.forEach((url, i) => {
        renderPageDisplay(skeletons[i], i, total, url.img);
        setProgress(i + 1, total);
      });
      setStatus(`Done · ${total} pages`);
      resolveAdjacentChapters();
      return;
    }

    // ── 3b. Non-English — cache hit? ──────────
    const cached = getCachedChapter(chapterId);
    if (cached && cached.targetLang === targetLang) {
      document.getElementById('chapter-info').textContent += '  · ✓ cached';
      setStatus('Loading from cache…');
      urls.forEach((url, i) => {
        const regions = cached.pageRegions[i];
        if (regions?.length) {
          // Populate _pageStore so ✏ CORRECT works on cached pages too.
          // We don't have raw OCR boxes, but sortedRegions is enough for the
          // correction sidebar to show real translations (tl / type).
          _pageStore.set(`${_activeChapterId}_${i}`, {
            cdnUrl: url.cdn, imgSrc: url.img, sourceLang, total,
            rawBoxes: [],
            autoRegions: regions.map(r => ({
              text: r.text || '', cx: r.x ?? 50, cy: r.y ?? 50,
              box:  r.box ?? [r.x-5, r.y-5, r.x+5, r.y+5],
              raw_box_ids: [],
            })),
            sortedRegions: regions.map(r => ({
              text: r.text || '', t: r.t || 'speech',
              cx: r.x ?? 50, cy: r.y ?? 50,
              box: r.box ?? [r.x-5, r.y-5, r.x+5, r.y+5],
              raw_box_ids: [], tl: r.tl || '—',
            })),
          });
          renderPage(skeletons[i], i, total, url.img, regions);
        } else {
          renderPageDisplay(skeletons[i], i, total, url.img);
        }
        setProgress(i + 1, total);
      });
      setStatus(`Done · ${total} pages · from cache`);
      resolveAdjacentChapters();
      return;
    }

    // ── 3c. Non-English — OCR + DeepSeek ──────
    setStatus(`0 / ${total} pages translated`);
    const pageRegions = new Array(total).fill(null);
    let doneCount     = 0;

    const tasks = urls.map((url, i) => async () => {
      if (cancelled) return;
      try {
        // OCR: send raw CDN URL (must be HTTPS for the proxy)
        const ocrData   = await ocrPage(url.cdn, sourceLang, signal);
        const ocrResult = ocrData.regions;
        if (cancelled) return;

        // Store raw OCR data so the correction UI can access it per-page
        _pageStore.set(`${_activeChapterId}_${i}`, {
          cdnUrl: url.cdn, imgSrc: url.img, sourceLang, total,
          rawBoxes: ocrData.rawBoxes,
          autoRegions: ocrResult,
        });

        if (!ocrResult.length) {
          // Full-art page — nothing to translate
          renderPageDisplay(skeletons[i], i, total, url.img);
          pageRegions[i] = [];
        } else {
          // Translate + classify via DeepSeek (proxied)
          // Sort regions per user's reading order preference
          const sortedOcr = _sortRegions(ocrResult);
          const translated = await translateBatch(sortedOcr, sourceLang, targetLang, signal);
          // FIX #2: use translated[j].t (classified type) instead of hardcoded 'speech'
          // BUG FIX: include `text` (original OCR source) so RE-TRANSLATE works correctly
          // when this chapter is reloaded from the localStorage cache. Without it,
          // retranslatePage filters out every region (r.text.trim() === '') and silently
          // aborts with "No regions to translate." on every cached chapter.
          const regions    = sortedOcr.map((r, j) => ({
            text: r.text || '',
            t:  translated[j]?.t  || 'speech',
            x:  r.cx,
            y:  r.cy,
            box: r.box,
            tl: translated[j]?.tl || '—',
          }));
          pageRegions[i] = regions;
          renderPage(skeletons[i], i, total, url.img, regions);
          // Store translated data back into _pageStore so the correction UI
          // can show real translations instead of all-"—" fallbacks.
          const _se = _pageStore.get(`${_activeChapterId}_${i}`);
          if (_se) _se.sortedRegions = sortedOcr.map((r, j) => ({
            text: r.text || '', t: translated[j]?.t || 'speech',
            cx: r.cx, cy: r.cy, box: r.box,
            raw_box_ids: r.raw_box_ids || [],
            tl: translated[j]?.tl || '—',
          }));
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
        // FIX #12: pass both cdn (for retry OCR) and img (for display)
        renderPageError(skeletons[i], i, total, url.cdn, url.img, err.message, sourceLang);
      }
      doneCount++;
      setProgress(doneCount, total);
      if (!cancelled) setStatus(`${doneCount} / ${total} pages translated`);
    });

    await runConcurrent(tasks, 3);

    if (!cancelled) {
      setStatus(`Done · ${total} pages`);
      setCachedChapter(chapterId, { meta, targetLang, pageRegions });
      refreshCacheUI();  // update pill after new chapter is cached
      resolveAdjacentChapters();
    }

  } catch (err) {
    if (err.name === 'AbortError') return;
    toast(`Error: ${err.message}`);
    show('screen-home');
  }
}

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
(function init() {
  // FIX #9: check the proxy is actually running; show a persistent warning if not
  fetch('/health')
    .then(r => { if (!r.ok) throw new Error(); })
    .catch(() => toast('⚠ Proxy not detected — run manga_proxy.py first.', 15000));

  // Restore badge reading order preference
  const savedOrder = localStorage.getItem('mtl_read_order') || 'auto-rtl';
  setReadOrder(savedOrder);

  // Populate cache info on home screen
  refreshCacheUI();

  // Restore MangaDex login state
  const savedAccess  = localStorage.getItem('mtl_md_access');
  const savedRefresh = localStorage.getItem('mtl_md_refresh');
  const savedExpiry  = parseInt(localStorage.getItem('mtl_md_expiry') || '0', 10);
  const savedMdUser  = localStorage.getItem('mtl_md_username') || '';
  _mdClientId     = localStorage.getItem('mtl_md_client_id')     || '';
  _mdClientSecret = localStorage.getItem('mtl_md_client_secret') || '';
  if (savedAccess && savedRefresh) {
    _mdAccessToken  = savedAccess;
    _mdRefreshToken = savedRefresh;
    _mdTokenExpiry  = savedExpiry;
    _mdUsername     = savedMdUser;
    _setMdStatus(true, savedMdUser);
    // Restore client ID field (not secret — keep that blank for privacy)
    if (_mdClientId) document.getElementById('md-client-id').value = _mdClientId;
    if (savedMdUser) document.getElementById('md-username').value  = savedMdUser;
  }


  const legacyKey = localStorage.getItem('mtl_ai_key');
  if (legacyKey) {
    const isGemini = legacyKey.startsWith('AIza');
    localStorage.setItem(isGemini ? 'mtl_key_gemini' : 'mtl_key_deepseek', legacyKey);
    localStorage.removeItem('mtl_ai_key');
  }

  const savedModel = localStorage.getItem('mtl_ai_model');
  if (savedModel && document.querySelector(`#ai-model option[value="${savedModel}"]`)) {
    document.getElementById('ai-model').value = savedModel;
  }
  onModelChange();  // restores per-provider key + syncs placeholder + hint + vision group visibility

  // Restore Vision OCR mode — must run AFTER onModelChange so the select exists and is visible
  const savedVisionMode = localStorage.getItem('mtl_vision_mode');
  const visionEl = document.getElementById('vision-ocr-mode');
  if (visionEl && savedVisionMode) {
    visionEl.value = savedVisionMode;
  }

  const savedScale = localStorage.getItem('mtl_merge_scale');
  if (savedScale) {
    document.getElementById('merge-scale').value = savedScale;
    document.getElementById('merge-scale-val').textContent = parseFloat(savedScale).toFixed(2);
  }
  document.getElementById('merge-scale').addEventListener('change', () => {
    localStorage.setItem('mtl_merge_scale', document.getElementById('merge-scale').value);
  });

  // Restore saved target language
  const savedTargetLang = localStorage.getItem('mtl_target_lang');
  if (savedTargetLang) {
    const sel = document.getElementById('target-lang');
    const exists = Array.from(sel.options).some(o => o.value === savedTargetLang);
    if (exists) {
      sel.value = savedTargetLang;
      onTargetLangChange();
    } else if (savedTargetLang !== '__custom__') {
      // Legacy plain-text value — put it in the custom field
      sel.value = '__custom__';
      document.getElementById('target-lang-custom').value = savedTargetLang;
      document.getElementById('target-lang-custom').style.display = 'block';
    }
  }

  document.getElementById('ai-key').addEventListener('blur', () => {
    const keyEl = document.getElementById('ai-key');
    const provider = keyEl.dataset.provider || getModelInfo().provider;
    const val = keyEl.value.trim();
    if (val) localStorage.setItem(`mtl_key_${provider}`, val);
  });

  document.getElementById('target-lang-custom').addEventListener('input', () => {
    localStorage.setItem('mtl_target_lang_custom', document.getElementById('target-lang-custom').value.trim());
  });

  document.getElementById('chapter-url').addEventListener('keydown', e => {
    if (e.key === 'Enter') startPipeline();
  });
})();

// ══════════════════════════════════════════════
// MANUAL BADGE REORDER  (per-page drag UI)
// ══════════════════════════════════════════════

function toggleReorderPanel(pageIdx) {
  const panel = document.getElementById(`reorder-panel-${pageIdx}`);
  const btn   = document.getElementById(`ro-btn-${pageIdx}`);
  if (!panel) return;
  const isOpen = panel.style.display !== 'none';
  if (isOpen) {
    panel.style.display = 'none';
    btn?.classList.remove('active');
  } else {
    _renderReorderPage(pageIdx);
    panel.style.display = 'block';
    btn?.classList.add('active');
  }
}

function _renderReorderPage(pageIdx) {
  const panel = document.getElementById(`reorder-panel-${pageIdx}`);
  if (!panel) return;
  const card = document.getElementById(`page-${pageIdx}`);
  const regions = card?._regions || [];
  const moKey = `${_activeChapterId}_${pageIdx}`;
  const order = _manualOrder.get(moKey) || regions.map((_, i) => i);

  const items = order.map((origIdx, pos) => {
    const r = regions[origIdx] || {};
    const tag = (r.t || 'speech').toLowerCase();
    const preview = (r.tl || '—').slice(0, 48) + ((r.tl || '').length > 48 ? '…' : '');
    return `<li class="reorder-item" draggable="true"
                data-pos="${pos}" data-orig="${origIdx}">
      <span class="reorder-drag-handle" title="Drag to reorder">⠿</span>
      <span class="reorder-badge-num t-${tag}">${pos + 1}</span>
      <span class="reorder-item-text" title="${esc(r.tl || '—')}">${esc(preview)}</span>
      <div class="reorder-arrow-btns">
        <button onclick="_roMove(${pageIdx},${pos},-1)" ${pos === 0 ? 'disabled' : ''} title="Move up">↑</button>
        <button onclick="_roMove(${pageIdx},${pos},1)"  ${pos === order.length - 1 ? 'disabled' : ''} title="Move down">↓</button>
      </div>
    </li>`;
  }).join('');

  panel.innerHTML = `
    <div class="reorder-panel-hdr">
      <span class="reorder-panel-title">⇅ Badge Reading Order</span>
      <span class="reorder-hint">Drag or use ↑↓ — badge 1 reads first</span>
    </div>
    <ul class="reorder-list" id="ro-list-${pageIdx}">${items}</ul>
    <button class="btn-apply-order" onclick="_applyReorder(${pageIdx})">✓ APPLY ORDER</button>`;

  _initDragReorder(pageIdx);
}

function _roMove(pageIdx, pos, dir) {
  const moKey  = `${_activeChapterId}_${pageIdx}`;
  const card   = document.getElementById(`page-${pageIdx}`);
  const regions = card?._regions || [];
  const order  = [...(_manualOrder.get(moKey) || regions.map((_, i) => i))];
  const newPos = pos + dir;
  if (newPos < 0 || newPos >= order.length) return;
  [order[pos], order[newPos]] = [order[newPos], order[pos]];
  _manualOrder.set(moKey, order);
  _renderReorderPage(pageIdx);
  // Keep panel open
  const panel = document.getElementById(`reorder-panel-${pageIdx}`);
  if (panel) panel.style.display = 'block';
}

function _applyReorder(pageIdx) {
  const moKey  = `${_activeChapterId}_${pageIdx}`;
  const card   = document.getElementById(`page-${pageIdx}`);
  const regions = card?._regions || [];
  const order  = _manualOrder.get(moKey) || regions.map((_, i) => i);

  // Re-render the translation panel with new order
  const transPanelEl = document.getElementById(`trans-panel-${pageIdx}`);
  const imgWrap = card?.querySelector('.img-wrap');
  if (!transPanelEl || !imgWrap) return;

  // Update badge numbers on image
  const badges = imgWrap.querySelectorAll('.badge');
  // Rebuild badge map: origIdx -> badge element
  const badgeByOrig = {};
  Array.from(badges).forEach((b, i) => { badgeByOrig[i] = b; });
  // Re-number badges per new order
  order.forEach((origIdx, newPos) => {
    const b = badgeByOrig[origIdx];
    if (b) b.textContent = String(newPos + 1);
  });

  // Rebuild translation rows
  let rowsHtml = '';
  order.forEach((origIdx, newPos) => {
    const r = regions[origIdx] || {};
    const tag = (r.t || 'speech').toLowerCase();
    rowsHtml += `<div class="t-row">
      <span class="t-num">${newPos + 1}</span>
      <span class="t-tag ${tag}">${tag}</span>
      <span class="t-text">${esc(r.tl || '—')}</span>
    </div>`;
  });
  transPanelEl.innerHTML = rowsHtml;

  toast('Badge order updated ✓');
  // Close panel
  const panel = document.getElementById(`reorder-panel-${pageIdx}`);
  if (panel) panel.style.display = 'none';
  document.getElementById(`ro-btn-${pageIdx}`)?.classList.remove('active');
}

function _initDragReorder(pageIdx) {
  const list = document.getElementById(`ro-list-${pageIdx}`);
  if (!list) return;
  let draggedItem = null;

  list.addEventListener('dragstart', e => {
    draggedItem = e.target.closest('.reorder-item');
    if (draggedItem) {
      draggedItem.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
    }
  });
  list.addEventListener('dragend', () => {
    draggedItem?.classList.remove('dragging');
    list.querySelectorAll('.reorder-item').forEach(i => i.classList.remove('drag-over'));
    draggedItem = null;
  });
  list.addEventListener('dragover', e => {
    e.preventDefault();
    const target = e.target.closest('.reorder-item');
    if (!target || target === draggedItem) return;
    list.querySelectorAll('.reorder-item').forEach(i => i.classList.remove('drag-over'));
    target.classList.add('drag-over');
  });
  list.addEventListener('drop', e => {
    e.preventDefault();
    const target = e.target.closest('.reorder-item');
    if (!target || !draggedItem || target === draggedItem) return;
    const fromPos = +draggedItem.dataset.pos;
    const toPos   = +target.dataset.pos;

    const moKey   = `${_activeChapterId}_${pageIdx}`;
    const card    = document.getElementById(`page-${pageIdx}`);
    const regions = card?._regions || [];
    const order   = [...(_manualOrder.get(moKey) || regions.map((_, i) => i))];

    const [moved] = order.splice(fromPos, 1);
    order.splice(toPos, 0, moved);
    _manualOrder.set(moKey, order);
    _renderReorderPage(pageIdx);
    // Keep panel open
    const panel = document.getElementById(`reorder-panel-${pageIdx}`);
    if (panel) panel.style.display = 'block';
  });
}

// ══════════════════════════════════════════════
// CORRECTION UI
// ══════════════════════════════════════════════

const _corrMode  = {};  // pageIdx → 'select'|'draw'|'delete'|'reorder'
const _corrSelId = {};  // pageIdx → selected region id (or null)
const _corrWork  = {};  // pageIdx → working regions array
const _corrDraw  = {};  // pageIdx → {active, x1, y1, x2, y2}

// ── Helpers ───────────────────────────────────
function _corrStoreKey(pageIdx)  { return `${_activeChapterId}_${pageIdx}`; }
function _corrLocalKey(pageIdx)  { return `mtl_corr_${_activeChapterId}_${pageIdx}`; }

function _saveCorrections(pageIdx) {
  try {
    localStorage.setItem(_corrLocalKey(pageIdx),
      JSON.stringify({ regions: _corrWork[pageIdx], savedAt: Date.now() }));
  } catch(e) { console.warn('Could not save corrections', e); }
}

function _loadCorrections(pageIdx) {
  try {
    const raw = localStorage.getItem(_corrLocalKey(pageIdx));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function _initWorkingRegions(pageIdx) {
  const saved = _loadCorrections(pageIdx);
  if (saved?.regions?.length) {
    _corrWork[pageIdx] = JSON.parse(JSON.stringify(saved.regions));
    return;
  }
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  if (!pd) { _corrWork[pageIdx] = []; return; }
  // Prefer sortedRegions (translated) over raw autoRegions so the correction
  // sidebar shows real tl values instead of hardcoded '—' for every bubble.
  const base = pd.sortedRegions || pd.autoRegions;
  _corrWork[pageIdx] = base.map((r, i) => ({
    id: i, text: r.text || '', t: r.t || 'speech',
    cx: r.cx, cy: r.cy,
    box: r.box || [r.cx-5, r.cy-5, r.cx+5, r.cy+5],
    rawBoxIds: r.raw_box_ids || [],
    deleted: false, isNew: false, tl: r.tl || '—',
  }));
}

// ── Open / Close ──────────────────────────────
function openCorrection(pageIdx) {
  const card = document.getElementById(`page-${pageIdx}`);
  if (!card) return;
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  if (!pd) { toast('Translate this page first, then use ✏ CORRECT.'); return; }

  _corrMode[pageIdx]  = 'select';
  _corrSelId[pageIdx] = null;
  _initWorkingRegions(pageIdx);
  card.classList.add('correcting');
  card.querySelector('.btn-correct')?.classList.add('active');
  card.innerHTML = _buildCorrHTML(pageIdx, pd.imgSrc);
  _attachCorrDrawEvents(pageIdx, pd);
  _renderCorrOverlay(pageIdx);
}

function closeCorrection(pageIdx) {
  const card = document.getElementById(`page-${pageIdx}`);
  if (!card) return;
  card.classList.remove('correcting');
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  if (!pd) return;
  // Rebuild normal page view from working (corrected) regions
  const displayRegions = (_corrWork[pageIdx] || [])
    .filter(r => !r.deleted)
    .map(r => ({ t: r.t||'speech', x: r.cx, y: r.cy, box: r.box, tl: r.tl||'—' }));
  renderPage(card, pageIdx, pd.total, pd.imgSrc, displayRegions);
}

// ── HTML builder ──────────────────────────────
function _buildCorrHTML(pageIdx, imgSrc) {
  return `
<div class="corr-layout">
  <div class="corr-left">
    <div class="corr-toolbar" id="corr-tb-${pageIdx}">
      <button class="corr-tool active" onclick="setCorrMode(${pageIdx},'select')">SELECT</button>
      <button class="corr-tool" onclick="setCorrMode(${pageIdx},'draw')">＋ DRAW</button>
      <button class="corr-tool" onclick="setCorrMode(${pageIdx},'delete')">✕ DELETE</button>
      <button class="corr-tool" onclick="setCorrMode(${pageIdx},'reorder')">⇅ ORDER</button>
    </div>
    <div class="corr-img-wrap" id="corr-iw-${pageIdx}">
      <img src="${esc(imgSrc)}" class="corr-img" id="corr-img-${pageIdx}" draggable="false">
      <div class="corr-overlay mode-select" id="corr-ov-${pageIdx}"></div>
    </div>
  </div>
  <div class="corr-sidebar" id="corr-sb-${pageIdx}">
    <div class="corr-empty-hint">Click a region to edit<br>or use ＋ DRAW to add one.</div>
  </div>
</div>
<div class="corr-footer">
  <button class="corr-btn-retrans" id="corr-retrans-${pageIdx}" onclick="retranslatePage(${pageIdx})">↺ RE-TRANSLATE</button>
  <button class="corr-btn-close" onclick="closeCorrection(${pageIdx})">CLOSE</button>
</div>`;
}

// ── Overlay rendering ─────────────────────────
function _renderCorrOverlay(pageIdx) {
  const ov = document.getElementById(`corr-ov-${pageIdx}`);
  if (!ov) return;
  const mode    = _corrMode[pageIdx] || 'select';
  const selId   = _corrSelId[pageIdx];
  const regions = (_corrWork[pageIdx] || []).filter(r => !r.deleted);

  ov.className  = `corr-overlay mode-${mode}`;
  ov.innerHTML  = regions.map((r, vi) => {
    const [x1,y1,x2,y2] = r.box;
    const sel = r.id === selId;
    return `<div class="corr-rbox${sel?' selected':''} mode-${mode}" id="rbox-${pageIdx}-${r.id}"
      style="left:${x1}%;top:${y1}%;width:${x2-x1}%;height:${y2-y1}%" data-id="${r.id}">
      <span class="rbox-num">${vi+1}</span>
    </div>`;
  }).join('');

  ov.querySelectorAll('.corr-rbox').forEach(el => {
    el.addEventListener('click', e => {
      e.stopPropagation();
      const id = parseInt(el.dataset.id);
      if ((_corrMode[pageIdx]||'select') === 'delete') _deleteCorrRegion(pageIdx, id);
      else _selectCorrRegion(pageIdx, id);
    });
  });
}

// ── Draw mode events ──────────────────────────
function _attachCorrDrawEvents(pageIdx, pd) {
  // Clean up any listeners left over from a previous openCorrection() call.
  // When openCorrection() rebuilds card.innerHTML the old overlay element is
  // destroyed without ever calling removeEventListener, orphaning the handlers
  // on document.  We find the stale refs stored on the *old* overlay (which
  // still exists in memory even after it was removed from the DOM) via the
  // data attribute we save below, then remove them before attaching new ones.
  const oldOvKey = `_corrOv_${pageIdx}`;
  const oldOv = window[oldOvKey];
  if (oldOv?._mmove) document.removeEventListener('mousemove', oldOv._mmove);
  if (oldOv?._mup)   document.removeEventListener('mouseup',   oldOv._mup);

  const ov  = document.getElementById(`corr-ov-${pageIdx}`);
  const img = document.getElementById(`corr-img-${pageIdx}`);
  if (!ov || !img) return;
  _corrDraw[pageIdx] = { active: false };

  ov.addEventListener('mousedown', e => {
    if ((_corrMode[pageIdx]||'select') !== 'draw') return;
    e.preventDefault();
    const [x, y] = _imgPct(e, img);
    _corrDraw[pageIdx] = { active:true, x1:x, y1:y, x2:x, y2:y };
    _drawPreview(pageIdx);
  });

  const mmove = e => {
    if (!_corrDraw[pageIdx]?.active) return;
    const img2 = document.getElementById(`corr-img-${pageIdx}`);
    if (!img2) return;
    const [x,y] = _imgPct(e, img2);
    _corrDraw[pageIdx].x2 = x; _corrDraw[pageIdx].y2 = y;
    _drawPreview(pageIdx);
  };
  const mup = async e => {
    if (!_corrDraw[pageIdx]?.active) return;
    _corrDraw[pageIdx].active = false;
    const d = _corrDraw[pageIdx];
    document.getElementById(`corr-ov-${pageIdx}`)?.querySelector('.draw-preview')?.remove();
    const x1=Math.min(d.x1,d.x2), y1=Math.min(d.y1,d.y2);
    const x2=Math.max(d.x1,d.x2), y2=Math.max(d.y1,d.y2);
    if ((x2-x1)<1 || (y2-y1)<1) return;
    await _finalizeBox(pageIdx, [x1,y1,x2,y2], pd);
  };
  document.addEventListener('mousemove', mmove);
  document.addEventListener('mouseup', mup);
  // store refs on element for cleanup on next openCorrection()
  ov._mmove = mmove; ov._mup = mup;
  window[`_corrOv_${pageIdx}`] = ov;
}

function _imgPct(e, imgEl) {
  const r = imgEl.getBoundingClientRect();
  return [
    Math.max(0, Math.min(100, (e.clientX-r.left)/r.width*100)),
    Math.max(0, Math.min(100, (e.clientY-r.top)/r.height*100)),
  ];
}

function _drawPreview(pageIdx) {
  const ov = document.getElementById(`corr-ov-${pageIdx}`);
  if (!ov) return;
  const d = _corrDraw[pageIdx];
  const x1=Math.min(d.x1,d.x2), y1=Math.min(d.y1,d.y2);
  const x2=Math.max(d.x1,d.x2), y2=Math.max(d.y1,d.y2);
  let p = ov.querySelector('.draw-preview');
  if (!p) { p = document.createElement('div'); p.className='draw-preview'; ov.appendChild(p); }
  p.style.cssText = `left:${x1}%;top:${y1}%;width:${x2-x1}%;height:${y2-y1}%`;
}

async function _finalizeBox(pageIdx, box, pd) {
  const img = document.getElementById(`corr-img-${pageIdx}`);
  if (!img) return;
  const nw=img.naturalWidth, nh=img.naturalHeight;
  const pxBox = [
    Math.round(box[0]/100*nw), Math.round(box[1]/100*nh),
    Math.round(box[2]/100*nw), Math.round(box[3]/100*nh),
  ];
  const sb = document.getElementById(`corr-sb-${pageIdx}`);
  if (sb) sb.innerHTML = `<div class="corr-empty-hint">Running OCR on selection…</div>`;

  let ocrText = '';
  try {
    const res = await fetch('/ocr-crop', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ url: pd.cdnUrl, box: pxBox, lang: pd.sourceLang }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toast(`OCR crop error: ${err?.description || `HTTP ${res.status}`}`);
    } else {
      const data = await res.json();
      ocrText = data.text || '';
    }
  } catch(e) { toast(`OCR crop failed: ${e.message}`); }

  const newId = Date.now();
  const cx=(box[0]+box[2])/2, cy=(box[1]+box[3])/2;
  _corrWork[pageIdx].push({ id:newId, text:ocrText, t:'speech', cx, cy, box, rawBoxIds:[], deleted:false, isNew:true, tl:'—' });
  _corrMode[pageIdx] = 'select';
  _updateToolbar(pageIdx);
  _renderCorrOverlay(pageIdx);
  _selectCorrRegion(pageIdx, newId);
  _saveCorrections(pageIdx);
}

// ── Select + sidebar ──────────────────────────
function _selectCorrRegion(pageIdx, id) {
  _corrSelId[pageIdx] = id;
  _renderCorrOverlay(pageIdx);
  _renderCorrSidebar(pageIdx);
}

function _renderCorrSidebar(pageIdx) {
  const sb = document.getElementById(`corr-sb-${pageIdx}`);
  if (!sb) return;
  const mode = _corrMode[pageIdx] || 'select';
  if (mode === 'reorder') { _renderReorderSidebar(pageIdx); return; }
  const id = _corrSelId[pageIdx];
  if (id == null) { sb.innerHTML=`<div class="corr-empty-hint">Click a region to edit<br>or use ＋ DRAW to add one.</div>`; return; }
  const regions = _corrWork[pageIdx] || [];
  const r = regions.find(x => x.id === id);
  if (!r) return;
  const vis     = regions.filter(x=>!x.deleted).findIndex(x=>x.id===id)+1;
  const others  = regions.filter(x=>!x.deleted && x.id!==id);
  const canSplit = (r.rawBoxIds||[]).length > 1;
  const mergeOpts = others.map((o,oi)=>{
    const ovi = regions.filter(x=>!x.deleted).findIndex(x=>x.id===o.id)+1;
    return `<option value="${o.id}">Region ${ovi}</option>`;
  }).join('');

  sb.innerHTML = `
    <div class="corr-sid-title">REGION ${vis}</div>
    <div class="corr-sid-label">OCR TEXT</div>
    <textarea class="corr-textarea" id="cta-${pageIdx}-${id}" rows="4">${esc(r.text)}</textarea>
    <div class="corr-sid-label">TYPE</div>
    <select class="corr-type-sel" id="ctype-${pageIdx}-${id}">
      ${['speech','thought','sfx','narration','sign'].map(t=>`<option value="${t}"${r.t===t?' selected':''}>${t}</option>`).join('')}
    </select>
    <div class="corr-action-row">
      ${canSplit?`<button class="corr-action-btn" onclick="_showSplitUI(${pageIdx},${id})">SPLIT</button>`:''}
      ${others.length?`
        <select class="corr-type-sel" id="cmerge-${pageIdx}-${id}" style="flex:1">
          <option value="">Merge with…</option>${mergeOpts}
        </select>
        <button class="corr-action-btn" onclick="_doMerge(${pageIdx},${id})">MERGE</button>`:''}
    </div>
    <button class="corr-action-btn danger" style="width:100%;margin-top:0.5rem" onclick="_deleteCorrRegion(${pageIdx},${id})">DELETE REGION</button>
    <div class="corr-sid-label" style="display:flex;align-items:center;justify-content:space-between">
      TRANSLATION
      <button id="crr-${pageIdx}-${id}" class="corr-action-btn"
        style="padding:0.1rem 0.6rem;font-size:0.75rem;margin:0"
        title="Re-translate this region using the rest of the page as context"
        onclick="retranslateRegion(${pageIdx},${id})">↺</button>
    </div>
    <div class="corr-tl-text" id="ctl-${pageIdx}-${id}">${esc(r.tl||'—')}</div>`;

  document.getElementById(`cta-${pageIdx}-${id}`)?.addEventListener('input', e=>{
    const reg = (_corrWork[pageIdx]||[]).find(x=>x.id===id);
    if (reg) { reg.text=e.target.value; _saveCorrections(pageIdx); }
  });
  document.getElementById(`ctype-${pageIdx}-${id}`)?.addEventListener('change', e=>{
    const reg = (_corrWork[pageIdx]||[]).find(x=>x.id===id);
    if (reg) { reg.t=e.target.value; _saveCorrections(pageIdx); _renderCorrOverlay(pageIdx); }
  });
}

// ── Split UI ──────────────────────────────────
function _showSplitUI(pageIdx, regionId) {
  const sb = document.getElementById(`corr-sb-${pageIdx}`);
  if (!sb) return;
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  const r  = (_corrWork[pageIdx]||[]).find(x=>x.id===regionId);
  if (!r || !pd) return;

  const rawBoxes = (r.rawBoxIds||[]).map(i=>pd.rawBoxes?.[i]).filter(Boolean)
    .sort((a,b)=>a.box[1]-b.box[1]);
  if (rawBoxes.length < 2) { toast('Not enough sub-boxes to split.'); return; }

  // Highlight raw boxes on overlay
  _renderCorrOverlay(pageIdx);
  const ov = document.getElementById(`corr-ov-${pageIdx}`);
  rawBoxes.forEach((b,i)=>{
    const d=document.createElement('div'); d.className='corr-raw-box';
    const [x1,y1,x2,y2]=b.box;
    d.style.cssText=`left:${x1}%;top:${y1}%;width:${x2-x1}%;height:${y2-y1}%`;
    d.innerHTML=`<span class="rbox-num raw">${i+1}</span>`;
    ov?.appendChild(d);
  });

  const items = rawBoxes.map((b,i)=>`
    <div class="corr-split-item">${esc(b.text)}</div>
    ${i<rawBoxes.length-1?`<button class="corr-split-line-btn" onclick="_confirmSplit(${pageIdx},${regionId},${i})">── split here ──</button>`:''}`).join('');

  sb.innerHTML = `
    <div class="corr-sid-title">SPLIT REGION</div>
    <div class="corr-split-list">${items}</div>
    <button class="corr-action-btn" style="margin-top:0.8rem;width:100%" onclick="_selectCorrRegion(${pageIdx},${regionId})">CANCEL</button>`;
}

function _confirmSplit(pageIdx, regionId, splitAfterIdx) {
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  const regions = _corrWork[pageIdx];
  if (!pd||!regions) return;
  const rIdx = regions.findIndex(x=>x.id===regionId);
  if (rIdx===-1) return;
  const r = regions[rIdx];
  const rawBoxes = (r.rawBoxIds||[]).map(i=>pd.rawBoxes?.[i]).filter(Boolean)
    .sort((a,b)=>a.box[1]-b.box[1]);

  const groupA = rawBoxes.slice(0, splitAfterIdx+1);
  const groupB = rawBoxes.slice(splitAfterIdx+1);
  if (!groupA.length||!groupB.length) return;

  function mkRegion(group, id) {
    const text = group.map(b=>b.text).join(' ');
    const x1=Math.min(...group.map(b=>b.box[0])), y1=Math.min(...group.map(b=>b.box[1]));
    const x2=Math.max(...group.map(b=>b.box[2])), y2=Math.max(...group.map(b=>b.box[3]));
    return { id, text, t:r.t, box:[x1,y1,x2,y2], cx:(x1+x2)/2, cy:(y1+y2)/2,
             rawBoxIds:group.map(b=>b.id??0), deleted:false, isNew:false, tl:'—' };
  }
  regions.splice(rIdx, 1, mkRegion(groupA, r.id), mkRegion(groupB, Date.now()));
  _corrSelId[pageIdx]=null;
  _renderCorrOverlay(pageIdx);
  _renderCorrSidebar(pageIdx);
  _saveCorrections(pageIdx);
  toast('Region split.');
}

// ── Merge ─────────────────────────────────────
function _doMerge(pageIdx, regionId) {
  const sel = document.getElementById(`cmerge-${pageIdx}-${regionId}`);
  if (!sel?.value) { toast('Select a region to merge with.'); return; }
  const otherId = parseInt(sel.value);
  const regions = _corrWork[pageIdx];
  if (!regions) return;
  const rA = regions.find(x=>x.id===regionId);
  const rBIdx = regions.findIndex(x=>x.id===otherId);
  const rB = regions[rBIdx];
  if (!rA||!rB) return;
  const allBoxes=[rA.box,rB.box];
  const box=[Math.min(...allBoxes.map(b=>b[0])),Math.min(...allBoxes.map(b=>b[1])),
             Math.max(...allBoxes.map(b=>b[2])),Math.max(...allBoxes.map(b=>b[3]))];
  rA.text=[rA.text,rB.text].filter(Boolean).join(' ');
  rA.box=box; rA.cx=(box[0]+box[2])/2; rA.cy=(box[1]+box[3])/2;
  rA.rawBoxIds=[...(rA.rawBoxIds||[]),...(rB.rawBoxIds||[])];
  regions.splice(rBIdx,1);
  _corrSelId[pageIdx]=regionId;
  _renderCorrOverlay(pageIdx); _renderCorrSidebar(pageIdx);
  _saveCorrections(pageIdx); toast('Regions merged — consider re-translating this region (↺).');
}

// ── Delete ────────────────────────────────────
function _deleteCorrRegion(pageIdx, regionId) {
  const regions = _corrWork[pageIdx];
  const r = regions?.find(x=>x.id===regionId);
  if (r) r.deleted=true;
  if (_corrSelId[pageIdx]===regionId) _corrSelId[pageIdx]=null;
  _renderCorrOverlay(pageIdx); _renderCorrSidebar(pageIdx);
  _saveCorrections(pageIdx);
}

// ── Reorder sidebar ───────────────────────────
function _renderReorderSidebar(pageIdx) {
  const sb=document.getElementById(`corr-sb-${pageIdx}`); if(!sb) return;
  const regions=(_corrWork[pageIdx]||[]).filter(r=>!r.deleted);
  sb.innerHTML=`
    <div class="corr-sid-title">READING ORDER</div>
    <div class="corr-order-hint">Use ↑↓ to set translation order</div>
    <div class="corr-order-list">${regions.map((r,i)=>`
      <div class="corr-order-item">
        <span class="corr-order-num">${i+1}</span>
        <span class="corr-order-text">${esc(r.text.slice(0,38))}${r.text.length>38?'…':''}</span>
        <div class="corr-order-btns">
          ${i>0?`<button onclick="_reorderReg(${pageIdx},${r.id},-1)">↑</button>`:'<span></span>'}
          ${i<regions.length-1?`<button onclick="_reorderReg(${pageIdx},${r.id},1)">↓</button>`:'<span></span>'}
        </div>
      </div>`).join('')||'<div class="corr-empty-hint">No regions</div>'}
    </div>`;
}

function _reorderReg(pageIdx, regionId, dir) {
  const all=_corrWork[pageIdx]; if(!all) return;
  const active=all.filter(r=>!r.deleted);
  const ci=active.findIndex(r=>r.id===regionId);
  const ni=ci+dir; if(ni<0||ni>=active.length) return;
  const i1=all.findIndex(r=>r.id===active[ci].id);
  const i2=all.findIndex(r=>r.id===active[ni].id);
  [all[i1],all[i2]]=[all[i2],all[i1]];
  _saveCorrections(pageIdx); _renderReorderSidebar(pageIdx); _renderCorrOverlay(pageIdx);
}

// ── Toolbar ───────────────────────────────────
function setCorrMode(pageIdx, mode) {
  _corrMode[pageIdx]=mode; _corrSelId[pageIdx]=null;
  _updateToolbar(pageIdx); _renderCorrOverlay(pageIdx); _renderCorrSidebar(pageIdx);
}

function _updateToolbar(pageIdx) {
  const tb=document.getElementById(`corr-tb-${pageIdx}`); if(!tb) return;
  const mode=_corrMode[pageIdx]||'select';
  const map={select:'SELECT',draw:'DRAW',delete:'DELETE',reorder:'ORDER'};
  tb.querySelectorAll('.corr-tool').forEach(btn=>{
    btn.classList.toggle('active', btn.textContent.includes(map[mode]));
  });
}

// ── Re-translate ──────────────────────────────
// ── Single-region retranslation with page context ────────────────────────────
// Sends ONE bubble to the AI but includes the rest of the page's already-
// translated regions as context so pronouns, names, and register stay consistent.
//
// CATCH: if the existing translations contain errors they will feed back as context.
// Recommended workflow: fix global/systemic errors with full-page ↺ RE-TRANSLATE
// first, then use the per-bubble ↺ to fine-tune individual regions.

async function translateSingleWithContext(region, contextRegions, sourceLang, targetLang) {
  const key     = document.getElementById('ai-key').value.trim();
  const info    = getModelInfo();
  const modelId = getModelId();
  if (!key) throw new Error(`${info.label} API key not set.`);

  // Build context from other already-translated, non-deleted regions
  const ctxLines = contextRegions
    .filter(r => r.tl && r.tl !== '—' && r.id !== region.id)
    .sort((a, b) => a.cy - b.cy || a.cx - b.cx)
    .map(r => `[${r.t ?? 'speech'}] ${r.text} → ${r.tl}`)
    .join('\n');

  const userMsg = (ctxLines
    ? `PAGE CONTEXT (already translated — use for consistency in names, pronouns, register):\n${ctxLines}\n\n`
    : '') +
    `RETRANSLATE THIS REGION:\n${JSON.stringify({ text: region.text, cx: region.cx, cy: region.cy })}`;

  const res = await fetch('/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      provider:    info.provider,
      key,
      source_lang: sourceLang,
      payload: {
        model:       modelId,
        temperature: 0.3,
        max_tokens:  400,
        ...(info.provider === 'deepseek' ? { response_format: { type: 'json_object' } } : {}),
        messages: [
          {
            role: 'system',
            content:
              `You are a manga translation expert. Re-translate ONE text region from ` +
              `${getLangName(sourceLang)} to ${targetLang}.\n` +
              `Use the page context to keep character names, pronouns, and speech register consistent.\n` +
              `Classify the text type: speech | thought | sfx | narration | sign.\n` +
              `Return ONLY a JSON object: {"tl":"translated text","t":"type"}\n` +
              `No markdown fences, no explanation, no extra keys.`,
          },
          { role: 'user', content: userMsg },
        ],
      },
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `${info.label} error ${res.status}`);
  }

  const data  = await res.json();
  const raw   = data.choices?.[0]?.message?.content ?? '';
  const clean = raw.replace(/```(?:json)?\n?/g, '').replace(/```/g, '').trim();
  try {
    const parsed = JSON.parse(clean);
    return {
      tl: String(parsed.tl ?? parsed.text ?? '—'),
      t:  VALID_TEXT_TYPES.has(parsed.t) ? parsed.t : 'speech',
    };
  } catch { return { tl: '—', t: 'speech' }; }
}

async function retranslateRegion(pageIdx, id) {
  const btn = document.getElementById(`crr-${pageIdx}-${id}`);
  if (btn) { btn.disabled = true; btn.textContent = '…'; }
  const pd = _pageStore.get(_corrStoreKey(pageIdx));
  if (!pd) {
    toast('No page data.');
    if (btn) { btn.disabled = false; btn.textContent = '↺'; }
    return;
  }
  const regions = (_corrWork[pageIdx] || []).filter(r => !r.deleted);
  const target  = regions.find(r => r.id === id);
  if (!target) { if (btn) { btn.disabled = false; btn.textContent = '↺'; } return; }
  try {
    const result = await translateSingleWithContext(target, regions, pd.sourceLang, getTargetLang());
    target.tl = result.tl;
    target.t  = result.t;
    _saveCorrections(pageIdx);
    // Re-render sidebar AND overlay — the type may have changed (e.g. speech→sfx),
    // so the badge colour on the image needs to update too.
    _renderCorrSidebar(pageIdx);
    _renderCorrOverlay(pageIdx);
    toast('Region re-translated.');
  } catch (e) { toast(`Translation failed: ${e.message}`); }
  if (btn) { btn.disabled = false; btn.textContent = '↺'; }
}

async function retranslatePage(pageIdx) {
  const btn=document.getElementById(`corr-retrans-${pageIdx}`);
  if(btn){btn.disabled=true; btn.textContent='Translating…';}
  const pd=_pageStore.get(_corrStoreKey(pageIdx));
  if(!pd){toast('No page data.');if(btn){btn.disabled=false;btn.textContent='↺ RE-TRANSLATE';} return;}
  const targetLang=getTargetLang();
  const working=(_corrWork[pageIdx]||[]).filter(r=>!r.deleted&&r.text.trim());
  if(!working.length){toast('No regions to translate.');if(btn){btn.disabled=false;btn.textContent='↺ RE-TRANSLATE';} return;}
  try {
    const ocrLike=working.map(r=>({text:r.text,cx:r.cx,cy:r.cy}));
    const translated=await translateBatch(ocrLike,pd.sourceLang,targetLang);
    working.forEach((r,j)=>{ r.tl=translated[j]?.tl||'—'; r.t=translated[j]?.t||r.t; });
    _saveCorrections(pageIdx);
    const sid=_corrSelId[pageIdx]; if(sid!=null) _renderCorrSidebar(pageIdx);
    toast('Page re-translated.');
  } catch(e){ toast(`Translation failed: ${e.message}`); }
  if(btn){btn.disabled=false; btn.textContent='↺ RE-TRANSLATE';}
}

</script>
</body>
</html>

"""
MANGADEX_API  = "https://api.mangadex.org"
MANGADEX_AUTH = "https://auth.mangadex.org/realms/mangadex/protocol/openid-connect/token"
DEEPSEEK_API  = "https://api.deepseek.com/v1/chat/completions"
GEMINI_API    = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Languages where EasyOCR struggles most with vertical/complex manga text.
# Kept for reference — the actual routing now sends ALL languages through
# Gemini Vision when an ai_key is available (see ocr_page()).
VISION_LANGS  = {'ja', 'zh', 'zh-hk', 'ko', 'ar', 'th'}

app = Flask(__name__)

# ─── MangaDex language code → EasyOCR language list ──────────────────────────
_LANG_MAP = {
    'vi':    ['vi'],      'it':    ['it'],      'pt':    ['pt'],
    'pt-br': ['pt'],      'ru':    ['ru'],      'fr':    ['fr'],
    'es':    ['es'],      'de':    ['de'],      'pl':    ['pl'],
    'nl':    ['nl'],      'tr':    ['tr'],      'id':    ['id'],
    'ko':    ['ko'],      'ja':    ['ja'],      'zh':    ['ch_sim'],
    'zh-hk': ['ch_tra'],  'th':    ['th'],      'ar':    ['ar'],
    'uk':    ['uk'],      'cs':    ['cs'],      'hu':    ['hu'],
    'ro':    ['ro'],      'sv':    ['sv'],      'da':    ['da'],
    'fi':    ['fi'],      'no':    ['no'],      'ms':    ['ms'],
    'hr':    ['hr'],      'sk':    ['sk'],      'bg':    ['bg'],
    'lt':    ['lt'],      'lv':    ['lv'],      'en':    ['en'],
}

def _easyocr_langs(chapter_lang: str) -> list:
    primary = _LANG_MAP.get(chapter_lang.lower(), ['en'])
    # Always add English as secondary so SFX / onomatopoeia get picked up
    if primary != ['en']:
        return primary + ['en']
    return primary


# ─── Per-language OCR confidence thresholds ───────────────────────────────────
# Languages with complex diacritics (Vietnamese) tend to produce more false
# positives at low confidence, while dense-script languages (Korean hangul)
# can score lower on genuine text.
_MIN_CONF_MAP = {
    'vi':    0.40,   # tonal diacritics inflate false positives
    'ko':    0.30,   # dense hangul blocks can score lower but still be correct
    'zh':    0.35,
    'zh-hk': 0.35,
    'th':    0.38,   # Thai vowel marks cause similar issues to Vietnamese
    'ar':    0.38,
}

# ─── Language-specific translation hints ──────────────────────────────────────
# Appended to the DeepSeek system prompt so the model understands
# cultural/linguistic quirks of each source language.
_LANG_HINTS = {
    'vi':    "Vietnamese comics use honorifics like 'anh/em/chị/bạn' to signal relationships and age hierarchy — preserve these dynamics in the English translation rather than flattening everyone to 'you'.",
    'ko':    "Korean webtoons use distinct speech levels (합쇼체 formal / 해요체 polite / 반말 casual). Reflect the character's social register in the English tone — formal characters should sound formal, casual characters casual.",
    'zh':    "Chinese manga may include chengyu (four-character idioms) and cultural references. Translate idioms by meaning rather than literally; add brief inline context only if the meaning would otherwise be lost.",
    'zh-hk': "This is Cantonese (Traditional Chinese). Cantonese slang and particles differ significantly from Mandarin. Prioritise natural idiomatic English over a literal rendering.",
    'id':    "Indonesian comics may use Javanese loanwords or regional slang (e.g. 'aku/gue', 'kamu/lo'). 'Gue/lo' signals casual Jakarta speech — keep dialogue informal where appropriate.",
    'th':    "Thai comics use politeness particles (ครับ for male speakers, ค่ะ/นะ for female). Reflect the speaker's politeness level and gender in the English tone where natural.",
    'ru':    "Russian manga often uses diminutives and expressive suffixes for names and nouns. Preserve endearment or mockery implied by diminutive forms rather than using the base name.",
    'fr':    "French comics distinguish 'tu' (informal) and 'vous' (formal/plural). Reflect the intimacy or formality of address in the English translation.",
    'es':    "Spanish comics may be from Spain or Latin America with regional vocabulary differences. Translate to neutral international English unless a specific dialect is obvious.",
    'de':    "German comics use 'du' (informal) vs 'Sie' (formal). Preserve the formality level in English dialogue.",
    'pl':    "Polish uses grammatical gender and case extensively in dialogue — pay attention to whether the speaker refers to themselves as male or female when choosing English phrasing.",
}


# ─── Image preprocessing ──────────────────────────────────────────────────────

def _is_colored_page(arr: np.ndarray) -> bool:
    """
    Return True if the page contains significant color (i.e. is not pure B&W).

    Checks the HSV saturation channel at 1/4 resolution for speed.
    A B&W manga page has near-zero saturation throughout. Colored panels push
    saturation up noticeably. Threshold: >5% of pixels with S > 20 (out of 255).
    Conservative enough to ignore JPEG chroma noise on B&W scans.
    """
    h, w  = arr.shape[:2]
    small = cv2.resize(arr, (max(32, w // 4), max(32, h // 4)),
                       interpolation=cv2.INTER_AREA)
    sat   = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]
    return float(np.mean(sat > 20)) > 0.05


def _preprocess_for_ocr(arr: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing: fast path for B&W pages, smart path for colored.

    B&W path  — original 3-step pipeline (grayscale → CLAHE 2.0 → denoise).
                Fast, screentone-safe, already well-tuned for classic manga.

    Colored path — 7-channel selection + adaptive inversion + CLAHE 3.0 + denoise.
                   Tries luminance gray, L* (LAB), V (HSV), S (HSV), R, G, B
                   and picks whichever channel has the most Laplacian edge variance
                   (= sharpest text edges) at 1/4 resolution.
                   Then inverts if background looks dark (border-ring sample).
                   Only triggered when _is_colored_page() returns True, so all
                   the colored-path risks (screentone scoring, border misfire)
                   are completely avoided on normal B&W content.
    """
    if not _is_colored_page(arr):
        # ── Fast B&W path (unchanged from original) ───────────────────────────
        gray     = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

    # ── Colored path ──────────────────────────────────────────────────────────
    h, w = arr.shape[:2]

    # 1. Generate 7 candidate single-channel representations
    gray_lum = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lab      = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    gray_l   = lab[:, :, 0]
    hsv      = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray_v   = hsv[:, :, 2]
    gray_s   = hsv[:, :, 1]
    r_ch     = arr[:, :, 0].copy()
    g_ch     = arr[:, :, 1].copy()
    b_ch     = arr[:, :, 2].copy()
    candidates = [gray_lum, gray_l, gray_v, gray_s, r_ch, g_ch, b_ch]

    # 2. Score each by Laplacian edge variance at 1/4 resolution (~1 ms)
    sh, sw = max(64, h // 4), max(64, w // 4)
    def _score(img):
        return float(cv2.Laplacian(
            cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA),
            cv2.CV_64F).var())
    best = max(candidates, key=_score)

    # 3. Adaptive inversion — sample border ring to estimate background
    bw, bh = max(4, w // 20), max(4, h // 20)
    border = np.concatenate([best[:bh,:].ravel(), best[-bh:,:].ravel(),
                             best[:,:bw].ravel(), best[:,-bw:].ravel()])
    if float(np.median(border)) < 127 or float(np.median(best)) < 90:
        best = cv2.bitwise_not(best)

    # 4. CLAHE + denoise
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(best)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


# ─── OCR engine — lazy-loaded, per-key events prevent blocking on download ────
#
#  FIX #3: the old code held _reader_lock for the entire model download
#  (potentially minutes).  Now we use a per-key threading.Event so that
#  concurrent requests for the same language wait without blocking other langs.
#
_readers       = {}   # tuple(langs) → EasyOCR reader (once loaded)
_reader_events = {}   # tuple(langs) → threading.Event (set when load complete/failed)
_reader_lock   = threading.Lock()
_infer_lock    = threading.Lock()   # serialises PyTorch inference (not thread-safe)

def _get_reader(chapter_lang: str):
    import easyocr                # lazy — no import cost at startup
    langs = _easyocr_langs(chapter_lang)
    key   = tuple(langs)

    # Fast path + loader/waiter decision in a single critical section.
    # Merging the two avoids a race window where a concurrent thread could
    # finish loading between the fast-path check and the loader decision,
    # causing a second (redundant) model download.
    with _reader_lock:
        if key in _readers:
            return _readers[key]
        if key not in _reader_events:
            evt       = threading.Event()
            _reader_events[key] = evt
            is_loader = True
        else:
            evt       = _reader_events[key]
            is_loader = False

    if not is_loader:
        # Wait until the loader thread finishes (or fails)
        evt.wait()
        reader = _readers.get(key)
        if reader is None:
            raise RuntimeError(f"OCR model {langs} failed to load — retry the page.")
        return reader

    # ── We are the loader thread ──────────────────────────────────────────────
    try:
        print(f"  [OCR] Loading model for {langs}  (first run may download ~100–400 MB)…")
        reader = easyocr.Reader(langs, gpu=False, verbose=False)
        print(f"  [OCR] {langs} ready.")
        with _reader_lock:
            _readers[key] = reader
        return reader
    except Exception:
        # Remove the event so the next request can attempt loading again
        with _reader_lock:
            _reader_events.pop(key, None)
        raise
    finally:
        evt.set()   # always unblock any waiters, even on failure


# ─── Gemini Vision OCR ───────────────────────────────────────────────────────

_VISION_LANG_NAMES = {
    # CJK / complex scripts (original Vision langs)
    'ja':    'Japanese',
    'zh':    'Chinese (Simplified)',
    'zh-hk': 'Chinese (Traditional / Cantonese)',
    # Korean — EasyOCR is notoriously shaky on hangul in manga
    'ko':    'Korean',
    # Southeast Asian scripts
    'vi':    'Vietnamese',
    'th':    'Thai',
    'id':    'Indonesian',
    'ms':    'Malay',
    # Arabic / right-to-left
    'ar':    'Arabic',
    # European languages
    'en':    'English',
    'fr':    'French',
    'es':    'Spanish',
    'de':    'German',
    'pt':    'Portuguese',
    'pt-br': 'Portuguese (Brazilian)',
    'it':    'Italian',
    'ru':    'Russian',
    'uk':    'Ukrainian',
    'pl':    'Polish',
    'nl':    'Dutch',
    'tr':    'Turkish',
    'cs':    'Czech',
    'hu':    'Hungarian',
    'ro':    'Romanian',
    'sv':    'Swedish',
    'da':    'Danish',
    'fi':    'Finnish',
    'no':    'Norwegian',
    'hr':    'Croatian',
    'sk':    'Slovak',
    'bg':    'Bulgarian',
    'lt':    'Lithuanian',
    'lv':    'Latvian',
}

def _ocr_gemini_vision(image_bytes: bytes, lang: str, key: str, model: str) -> list:
    """
    Send a manga page image to Gemini Vision and ask it to extract all text
    regions with approximate centre positions.

    Returns a list of dicts matching the EasyOCR output schema:
        [{"text": "…", "cx": 45.2, "cy": 23.1, "box": [x1%,y1%,x2%,y2%]}]

    Falls back to an empty list on any error so the caller can degrade gracefully.
    """
    import base64, json as _json

    lang_name = _VISION_LANG_NAMES.get(lang, 'the source language')
    b64       = base64.b64encode(image_bytes).decode()

    # Detect mime type (JPEG vs PNG) from magic bytes
    mime = "image/png" if image_bytes[:4] == b'\x89PNG' else "image/jpeg"

    prompt = (
        f"You are a manga OCR engine. Extract ALL visible text from this manga page.\n"
        f"The text is in {lang_name}. Many speech bubbles use VERTICAL text — "
        f"read each bubble top-to-bottom and output it as a single string.\n\n"
        f"Return ONLY a raw JSON array, no markdown, no explanation:\n"
        f'[{{"text":"<exact original text>","cx":<% from left 0-100>,"cy":<% from top 0-100>}},...]\n\n'
        f"Rules:\n"
        f"- Do NOT translate. Keep original characters exactly as printed.\n"
        f"- cx / cy = approximate centre of the text region as percentage of image size.\n"
        f"- One entry per speech bubble / caption box / sound effect / sign.\n"
        f"- If a panel has no text, skip it entirely.\n"
        f"- Pure decorative lines or panel borders are NOT text — skip them."
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime, "data": b64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 2048,
            # Disable thinking mode so the model doesn't burn its token budget
            # reasoning about garbled OCR fragments (same fix as _translate_gemini).
            "thinkingConfig":  {"thinkingBudget": 0},
            # API-level enforcement: even if thinkingBudget:0 is ignored, the model
            # cannot return prose — it must output valid JSON.  Fixes both the
            # token-starvation stub (19-char output) and the prose-dump failure modes.
            "responseMimeType": "application/json",
        },
    }

    url = GEMINI_API.format(model=model) + f"?key={key}"
    try:
        r = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "MangaTL-Reader/1.0"},
            timeout=60,
        )
        if not r.ok:
            if r.status_code == 429:
                print(
                    f"  [Vision OCR] Rate-limited (429) — falling back to EasyOCR. "
                    f"Free tier quota may be exhausted. Consider switching Vision OCR "
                    f"to 'Smart' mode or upgrading to a paid Gemini plan."
                )
            else:
                print(f"  [Vision OCR] Gemini error {r.status_code}: {r.text[:200]}")
            return []
        gemini_resp = r.json()
        cand  = (gemini_resp.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        # Skip thought parts — same pattern used in _translate_gemini.
        # With responseMimeType this loop normally finds JSON in the first
        # non-thought part; the fallback handles edge cases.
        text = ""
        for part in parts:
            if not part.get("thought", False):
                candidate = part.get("text", "")
                if candidate.strip():
                    text = candidate
                    break
        if not text and parts:          # absolute fallback
            text = parts[0].get("text", "")
        # Strip markdown fences if present
        clean = text.replace("```json", "").replace("```", "").strip()
        match = __import__("re").search(r"\[[\s\S]*\]", clean)
        if not match:
            return []
        items = _json.loads(match.group(0))
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            t  = str(item.get("text", "")).strip()
            cx = float(item.get("cx", 50))
            cy = float(item.get("cy", 50))
            if not t:
                continue
            # Synthesise a small bounding box around the centre point
            # (used by the correction UI; exact pixel coords unavailable from Vision)
            half_w = 8.0
            half_h = 5.0
            out.append({
                "text": t,
                "cx":   cx,
                "cy":   cy,
                "box":  [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
            })
        return out
    except Exception as e:
        print(f"  [Vision OCR] failed: {e}")
        return []


# ─── Panel border detection ───────────────────────────────────────────────────

def _find_panel_borders(gray: np.ndarray, img_w: int, img_h: int):
    """
    Detect horizontal and vertical panel border lines in a manga page.

    Strategy: morphological OPEN with a long thin kernel.  A feature only
    survives the OPEN if it spans at least 40 % of the image dimension, which
    reliably captures panel borders while ignoring speech bubble outlines,
    character art, and screentone patterns.

    Returns:
        h_borders — sorted list of y-coordinates (pixel) of horizontal borders
        v_borders — sorted list of x-coordinates (pixel) of vertical borders
    """
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # ── Horizontal borders ────────────────────────────────────────────────────
    min_h_span = max(1, int(img_w * 0.40))
    h_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_span, 1))
    h_img      = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # ── Vertical borders ──────────────────────────────────────────────────────
    min_v_span = max(1, int(img_h * 0.40))
    v_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_span))
    v_img      = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    def _cluster(indices, gap: int = 6) -> list:
        """Collapse a run of consecutive pixel indices into a single midpoint."""
        if not len(indices):
            return []
        borders, run_start, prev = [], int(indices[0]), int(indices[0])
        for idx in indices[1:]:
            idx = int(idx)
            if idx - prev > gap:
                borders.append((run_start + prev) // 2)
                run_start = idx
            prev = idx
        borders.append((run_start + prev) // 2)
        return borders

    h_rows = np.where(np.any(h_img > 0, axis=1))[0]
    v_cols = np.where(np.any(v_img > 0, axis=0))[0]

    return _cluster(h_rows), _cluster(v_cols)


def _crosses_border(
    box_a: tuple, box_b: tuple,
    h_borders: list, v_borders: list,
) -> bool:
    """
    Return True if a direct path from box_a to box_b must cross a panel border.

    We check whether any detected border line falls strictly inside the gap
    between the two boxes — not inside either box itself.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Vertical gap (potential horizontal border between them)
    gap_top    = min(ay2, by2)   # bottom of the higher box
    gap_bottom = max(ay1, by1)   # top  of the lower  box
    if gap_bottom > gap_top:
        for y in h_borders:
            if gap_top < y < gap_bottom:
                return True

    # Horizontal gap (potential vertical border between them)
    gap_left  = min(ax2, bx2)   # right edge of the left  box
    gap_right = max(ax1, bx1)   # left  edge of the right box
    if gap_right > gap_left:
        for x in v_borders:
            if gap_left < x < gap_right:
                return True

    return False


# ─── Bubble region merging ────────────────────────────────────────────────────

def _merge_bubble_regions(
    boxes,
    img_w: int,
    img_h: int,
    h_borders:    list | None  = None,
    v_borders:    list | None  = None,
    margin_scale: float        = 0.5,
):
    """
    Group OCR bounding boxes that belong to the same speech bubble, then merge
    each group into a single region with combined text.

    Algorithm:
      1. Expand every box by MERGE_MARGIN pixels on all sides.
      2. Any two expanded boxes that overlap → same bubble (union-find),
         UNLESS a panel border line falls in the gap between them.
      3. Within each group sort fragments top-to-bottom then left-to-right
         (natural reading order inside the bubble) and join their text.
      4. Return one {text, cx, cy} per group, centred on the merged bounding box.

    MERGE_MARGIN is now content-adaptive:
      margin = median_box_height x margin_scale

      Using the median height of detected boxes means the margin automatically
      scales with the actual text size on each page — dense pages with small
      bubbles get a small margin, splash pages with large text get a large one.
      margin_scale (default 0.5) is the user-tunable sensitivity knob.

      Webtoon strips (img_h / img_w > 2) use 60 % of the normal scale to
      avoid bridging vertically-stacked panels on tall narrow canvases.

      Fallback: if fewer than 3 boxes are detected (statistically unreliable),
      falls back to 3 % of the shorter image dimension.

    Panel border guard (h_borders / v_borders):
      Even if two expanded boxes overlap, they will NOT be merged if a detected
      panel border line lies in the gap between them.  This prevents speech
      bubbles from adjacent panels being collapsed into one region, which is the
      most common cause of incoherent translations.
    """
    if not boxes:
        return [], []

    h_borders = h_borders or []
    v_borders = v_borders or []

    is_webtoon = (img_h / max(img_w, 1)) > 2.0
    eff_scale  = margin_scale * (0.6 if is_webtoon else 1.0)

    if len(boxes) >= 3:
        box_heights = sorted(b[3] - b[1] for b in boxes)
        median_h    = box_heights[len(box_heights) // 2]
        margin      = max(4, int(median_h * eff_scale))
    else:
        # Too few boxes to compute a reliable median — fall back to % of image
        margin = int(min(img_w, img_h) * 0.03)

    # ── Union-Find ────────────────────────────────────────────────────────────
    n      = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x         = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def expanded(i):
        x1, y1, x2, y2, _ = boxes[i]
        return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)

    def overlaps(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2

    exp = [expanded(i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if overlaps(exp[i], exp[j]):
                # Even if expanded boxes overlap, refuse to merge them if a
                # panel border separates the original (un-expanded) boxes.
                if not _crosses_border(boxes[i][:4], boxes[j][:4],
                                       h_borders, v_borders):
                    union(i, j)

    # ── Group by root ─────────────────────────────────────────────────────────
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # ── Merge each group ──────────────────────────────────────────────────────
    regions      = []
    group_raw_ids = []   # parallel list: raw box indices per merged region
    for indices in groups.values():
        # Sort: top-to-bottom primary, left-to-right secondary
        indices.sort(key=lambda i: (boxes[i][1], boxes[i][0]))

        # Re-join fragments split across lines with a trailing hyphen.
        # e.g. ["SHUN-", "PEI."] → "SHUNPEI."
        texts  = [boxes[i][4] for i in indices]
        joined: list[str] = []
        for fragment in texts:
            if joined and joined[-1].endswith('-'):
                joined[-1] = joined[-1][:-1] + fragment
            else:
                joined.append(fragment)
        merged_text = " ".join(joined)
        mx1 = min(boxes[i][0] for i in indices)
        my1 = min(boxes[i][1] for i in indices)
        mx2 = max(boxes[i][2] for i in indices)
        my2 = max(boxes[i][3] for i in indices)

        regions.append({
            "text": merged_text,
            "cx":   round((mx1 + mx2) / 2 / img_w * 100, 1),
            "cy":   round((my1 + my2) / 2 / img_h * 100, 1),
            # Percentage bounding box so the frontend can overlay correction
            # boxes on the image without knowing the raw pixel dimensions.
            "box":  [
                round(mx1 / img_w * 100, 1), round(my1 / img_h * 100, 1),
                round(mx2 / img_w * 100, 1), round(my2 / img_h * 100, 1),
            ],
        })
        group_raw_ids.append(list(indices))

    # Sort final regions top-to-bottom, keeping group_raw_ids in sync.
    if regions:
        paired = sorted(zip(regions, group_raw_ids),
                        key=lambda p: (p[0]["cy"], p[0]["cx"]))
        regions, group_raw_ids = map(list, zip(*paired))
    return regions, group_raw_ids


# ─── Routes ───────────────────────────────────────────────────────────────────

# FIX #9 — health endpoint so the frontend can detect "proxy not running"
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return Response(_HTML, content_type="text/html; charset=utf-8")


@app.route("/mangadex/<path:api_path>")
def mangadex_api(api_path):
    url    = f"{MANGADEX_API}/{api_path}"
    params = request.query_string.decode()
    if params:
        url = f"{url}?{params}"
    headers = {"User-Agent": "MangaTL-Reader/1.0"}
    # Forward auth token if the frontend provided one
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        headers["Authorization"] = auth
    try:
        r = requests.get(url, timeout=15, headers=headers)
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))
    except requests.RequestException as e:
        abort(502, f"MangaDex API error: {e}")


# ─── MangaDex OAuth2 login / refresh ─────────────────────────────────────────
# MangaDex uses personal clients (not the public OAuth code flow).
# Users create one at: mangadex.org → Account Settings → API Clients
# Then log in with: client_id + client_secret + username + password.

@app.route("/auth/login", methods=["POST"])
def auth_login():
    """
    POST { username, password, client_id, client_secret }
    Returns { access_token, refresh_token, expires_in }
    """
    body          = request.get_json(force=True, silent=True) or {}
    username      = body.get("username",      "").strip()
    password      = body.get("password",      "").strip()
    client_id     = body.get("client_id",     "").strip()
    client_secret = body.get("client_secret", "").strip()

    if not all([username, password, client_id, client_secret]):
        abort(400, "username, password, client_id and client_secret are all required.")

    try:
        r = requests.post(
            MANGADEX_AUTH,
            data={
                "grant_type":    "password",
                "username":      username,
                "password":      password,
                "client_id":     client_id,
                "client_secret": client_secret,
            },
            headers={"User-Agent": "MangaTL-Reader/1.0"},
            timeout=15,
        )
    except requests.RequestException as e:
        abort(502, f"MangaDex auth error: {e}")

    if not r.ok:
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))

    d = r.json()
    return jsonify({
        "access_token":  d["access_token"],
        "refresh_token": d.get("refresh_token", ""),
        "expires_in":    d.get("expires_in", 900),
    })


@app.route("/auth/refresh", methods=["POST"])
def auth_refresh():
    """
    POST { refresh_token, client_id, client_secret }
    Returns { access_token, refresh_token, expires_in }
    """
    body          = request.get_json(force=True, silent=True) or {}
    refresh_token = body.get("refresh_token", "").strip()
    client_id     = body.get("client_id",     "").strip()
    client_secret = body.get("client_secret", "").strip()

    if not all([refresh_token, client_id, client_secret]):
        abort(400, "refresh_token, client_id and client_secret are required.")

    try:
        r = requests.post(
            MANGADEX_AUTH,
            data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     client_id,
                "client_secret": client_secret,
            },
            headers={"User-Agent": "MangaTL-Reader/1.0"},
            timeout=15,
        )
    except requests.RequestException as e:
        abort(502, f"MangaDex token refresh error: {e}")

    if not r.ok:
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))

    d = r.json()
    return jsonify({
        "access_token":  d["access_token"],
        "refresh_token": d.get("refresh_token", ""),
        "expires_in":    d.get("expires_in", 900),
    })


# FIX #12 — /proxy is now actively used by the frontend for all image display
@app.route("/proxy")
def proxy():
    url = request.args.get("url", "").strip()
    if not url.startswith("https://"):
        abort(400, "Only HTTPS image URLs are accepted.")
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "MangaTL-Reader/1.0"})
        r.raise_for_status()
        return Response(r.content, content_type=r.headers.get("Content-Type", "image/jpeg"))
    except requests.RequestException as e:
        abort(502, f"CDN fetch failed: {e}")


# ─── Translation helpers (one per provider) ──────────────────────────────────

def _inject_lang_hint(payload: dict, source_lang: str) -> None:
    """Append a language-specific hint to the system message in-place."""
    lang_hint = _LANG_HINTS.get(source_lang, "")
    if not lang_hint:
        return
    for msg in payload.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            msg["content"] = msg["content"] + f"\n\nLANGUAGE NOTE: {lang_hint}"
            break


def _translate_deepseek(api_key: str, payload: dict):
    """
    Forward an OpenAI-style payload to DeepSeek and return a normalised response.

    DeepSeek V4 models support dual Thinking / Non-Thinking modes.  In thinking
    mode the final answer lands in `choices[0].message.content` as usual, but the
    chain-of-thought appears in `reasoning_content`.  Occasionally (especially
    under heavy load or with certain prompt shapes) `content` comes back as null
    or an empty string while `reasoning_content` contains the actual JSON output.
    Without the normalisation below that silently becomes all-"—" translations.
    """
    import json as _json
    try:
        r = requests.post(
            DEEPSEEK_API,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent":    "MangaTL-Reader/1.0",
            },
            timeout=60,
        )
    except requests.RequestException as e:
        abort(502, f"DeepSeek API error: {e}")

    if not r.ok:
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))

    # ── Normalise to guaranteed-non-empty content ─────────────────────────────
    try:
        data    = r.json()
        choices = data.get("choices") or []
        msg     = choices[0].get("message", {}) if choices else {}
        content = msg.get("content") or ""
        # Thinking-mode fallback: the JSON may have ended up in reasoning_content
        if not content.strip():
            content = msg.get("reasoning_content") or ""
        if not content.strip():
            finish = choices[0].get("finish_reason", "") if choices else ""
            abort(422,
                  f"DeepSeek returned no content (finish_reason={finish!r}). "
                  "Retry the page.")
        return Response(
            _json.dumps({"choices": [{"message": {"content": content}}]}),
            status=200,
            content_type="application/json",
        )
    except Exception:
        # Unexpected response shape — pass it through raw; the frontend's own
        # empty-content guard will surface the error to the user.
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))


def _translate_gemini(api_key: str, payload: dict):
    """
    Convert an OpenAI-style payload to Gemini's generateContent format,
    call the Gemini API, then normalize the response back to OpenAI format
    so the frontend needs no changes.
    """
    import json as _json

    model       = payload.get("model", "gemini-2.0-flash")
    messages    = payload.get("messages", [])
    temperature = payload.get("temperature", 0.3)
    max_tokens  = payload.get("max_tokens", 3000)

    # Split system instruction from conversation turns
    system_text = ""
    user_text   = ""
    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("content", "")
        if role == "system":
            system_text = text
        elif role == "user":
            user_text = text

    gemini_payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
            # Disable thinking mode (Gemini 2.5+).
            # Without this, thinking-capable models return a multi-part response:
            #   parts[0] = {"thought": true, "text": "...8000-char analysis..."}
            #   parts[1] = {"text": "{\"translations\":[...]}"}   ← what we actually need
            # The old parts[0]-only extraction grabbed the analysis instead of the JSON,
            # causing all-"—" translations on every page with enough text to trigger thinking.
            # Setting thinkingBudget=0 requests a direct JSON response on all pages.
            # Models that don't support thinkingConfig silently ignore this field.
            "thinkingConfig": {"thinkingBudget": 0},
            # API-level JSON enforcement (Gemini 1.5+).
            # Even when thinkingBudget=0 is ignored by a model, responseMimeType forces
            # the output to be valid JSON — the model cannot return prose or markdown.
            # This is the definitive fix for "All 3 JSON parse strategies failed" errors
            # caused by the model dumping its reasoning chain as plain text.
            "responseMimeType": "application/json",
        },
    }
    if system_text:
        gemini_payload["system_instruction"] = {
            "parts": [{"text": system_text}]
        }

    url = GEMINI_API.format(model=model) + f"?key={api_key}"

    try:
        r = requests.post(
            url,
            json=gemini_payload,
            headers={"Content-Type": "application/json", "User-Agent": "MangaTL-Reader/1.0"},
            timeout=60,
        )
    except requests.RequestException as e:
        abort(502, f"Gemini API error: {e}")

    if not r.ok:
        # Surface the Gemini error directly so the frontend can show it
        return Response(r.content, status=r.status_code,
                        content_type=r.headers.get("Content-Type", "application/json"))

    # Normalize to OpenAI-compatible format (choices[0].message.content)
    try:
        gemini_resp = r.json()
        candidates  = gemini_resp.get("candidates") or []
        cand        = candidates[0] if candidates else {}
        # ── Extract text, skipping thought parts ─────────────────────────────
        # Safety net for when thinking is still active despite thinkingBudget=0
        # (e.g. a model that ignores the hint, or thinkingBudget not yet supported).
        # Gemini thinking responses look like:
        #   parts = [{"thought": True, "text": "..."}, {"text": "...JSON..."}]
        # We skip any part flagged as thought and take the first real output part.
        parts = cand.get("content", {}).get("parts", [])
        text  = ""
        for part in parts:
            if not part.get("thought", False):
                candidate = part.get("text", "")
                if candidate.strip():
                    text = candidate
                    break
        # Absolute fallback — should never be reached with thinkingBudget=0
        if not text and parts:
            text = parts[0].get("text", "")
    except Exception:
        abort(502, "Gemini response parse error.")

    # ── Guard: empty text means Gemini produced no output ────────────────────
    # This happens when the safety filter blocks the request (finishReason=SAFETY),
    # the response was truncated (MAX_TOKENS with no partial text), or the model
    # returned a pure-thinking response with no output parts.
    # Without this check the proxy returns HTTP 200 with content="", which the
    # frontend silently treats as a failed parse and falls back to all-"—"
    # translations — indistinguishable from a successful empty-chapter result.
    if not text.strip():
        finish      = cand.get("finishReason", "")
        prompt_fb   = gemini_resp.get("promptFeedback", {})
        block       = prompt_fb.get("blockReason", "")
        if block:
            abort(422, f"Gemini blocked the request ({block}). Try a different model or retry.")
        elif finish and finish != "STOP":
            abort(422, f"Gemini returned no text (finishReason={finish!r}). Retry the page.")
        else:
            abort(422, "Gemini returned an empty response. Check your API key / model and retry.")

    normalized = {"choices": [{"message": {"content": text}}]}
    return Response(
        _json.dumps(normalized),
        status=200,
        content_type="application/json",
    )


# ─── /translate  (multi-provider) ────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    """
    POST body:
        {
          "provider":    "gemini" | "deepseek",   # default: deepseek
          "key":         "<api key>",
          "payload":     { ...OpenAI-style chat-completions body... },
          "source_lang": "vi"                      # optional, for lang hints
        }

    All providers return an OpenAI-compatible JSON body so the frontend
    only needs to read choices[0].message.content regardless of provider.
    The API key is forwarded server-side and never appears in DevTools.
    """
    body        = request.get_json(force=True, silent=True) or {}
    provider    = body.get("provider", "deepseek").strip().lower()
    api_key     = body.get("key", "").strip()
    payload     = body.get("payload")
    source_lang = body.get("source_lang", "").strip().lower()

    if not api_key:
        abort(400, "API key required.")
    if not isinstance(payload, dict):
        abort(400, "payload must be a JSON object.")

    # Inject cultural/linguistic hints into the system message
    _inject_lang_hint(payload, source_lang)

    if provider == "gemini":
        return _translate_gemini(api_key, payload)
    else:
        # Default / "deepseek"
        return _translate_deepseek(api_key, payload)


@app.route("/ocr", methods=["POST"])
def ocr_page():
    """
    POST body:  { "url": "https://cdn…/page.jpg", "lang": "vi",
                  "ai_key": "AIza…",          # optional — enables Gemini Vision OCR
                  "ai_model": "gemini-2.5-flash",
                  "vision_mode": "smart" }    # 'smart' | 'all' | 'off'  (default: 'smart')
    Response:   { "regions": [{ "text": "…", "cx": 45.2, "cy": 23.1 }, …] }

    vision_mode controls when Gemini Vision OCR fires (only when ai_key is present):
      'smart' — only for languages in VISION_LANGS (complex/vertical scripts).
                Best for free-tier users: saves quota for scripts EasyOCR handles well.
      'all'   — Vision OCR for every language. Max quality but doubles API calls.
      'off'   — Always EasyOCR regardless. Zero extra quota used.
    DeepSeek users never send an ai_key so they always use EasyOCR.
    If Gemini Vision errors or returns empty, falls back to EasyOCR automatically.
    """
    body         = request.get_json(force=True, silent=True) or {}
    image_url    = body.get("url", "").strip()
    lang         = body.get("lang", "en").lower()
    margin_scale = float(body.get("margin_scale", 0.5))
    ai_key       = body.get("ai_key",       "").strip()
    ai_model     = body.get("ai_model",     "gemini-2.5-flash").strip()
    vision_mode  = body.get("vision_mode",  "smart").strip().lower()  # 'smart' | 'all' | 'off'

    if not image_url.startswith("https://"):
        abort(400, "url must be an HTTPS URL.")

    # 1. Download
    try:
        img_r = requests.get(image_url, timeout=20,
                              headers={"User-Agent": "MangaTL-Reader/1.0"})
        img_r.raise_for_status()
    except requests.RequestException as e:
        abort(502, f"Image download failed: {e}")

    image_bytes = img_r.content

    # ── Gemini Vision routing ─────────────────────────────────────────────────
    # Decide whether to use Vision OCR based on vision_mode:
    #   'all'   → always use Vision (when key present)
    #   'smart' → only for VISION_LANGS (complex/vertical scripts)
    #   'off'   → skip Vision, go straight to EasyOCR
    # Free-tier users should stick with 'smart' — each Vision call costs quota
    # on top of the translation call, so 'all' roughly halves their daily limit.
    use_vision = bool(ai_key) and vision_mode != "off" and (
        vision_mode == "all" or lang in VISION_LANGS
    )

    if use_vision:
        print(f"  [OCR] Using Gemini Vision for lang={lang} (mode={vision_mode})")
        regions = _ocr_gemini_vision(image_bytes, lang, ai_key, ai_model)
        if regions:
            # Vision regions already have cx/cy; synthesise raw_boxes in same format
            raw_boxes_out = [
                {"id": i, "text": r["text"],
                 "box": r.get("box", [r["cx"]-8, r["cy"]-5, r["cx"]+8, r["cy"]+5]),
                 "px":  [0, 0, 0, 0]}   # pixel coords unavailable from Vision
                for i, r in enumerate(regions)
            ]
            for i, region in enumerate(regions):
                region["raw_box_ids"] = [i]
            return jsonify({"regions": regions, "raw_boxes": raw_boxes_out})
        # Fall through to EasyOCR if Vision returned nothing (error or empty page)
        print(f"  [OCR] Vision returned empty — falling back to EasyOCR")

    # ── EasyOCR path ──────────────────────────────────────────────────────────
    # 2. Decode
    try:
        pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil.size
        arr  = np.array(pil)
    except Exception as e:
        abort(422, f"Image decode error: {e}")

    # 2b. Detect panel borders from the ORIGINAL grayscale image.
    #     Must be done before preprocessing because CLAHE can alter the dark
    #     border lines and make them harder to distinguish from panel content.
    gray_orig         = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h_borders, v_borders = _find_panel_borders(gray_orig, w, h)

    # 2c. Preprocess — CLAHE contrast enhancement + mild denoising.
    #     Improves OCR accuracy significantly for text printed over patterned
    #     or gradient backgrounds, and for languages with fine diacritics.
    arr = _preprocess_for_ocr(arr)

    # 3. OCR  (serialised — PyTorch is not thread-safe)
    try:
        reader = _get_reader(lang)
        with _infer_lock:
            raw = reader.readtext(
                arr,
                detail=1,
                paragraph=False,
                contrast_ths=0.1,    # default 0.1 — explicit for clarity
                adjust_contrast=0.5, # auto-boost low-contrast text regions
                text_threshold=0.6,  # slightly more permissive than default 0.7
                min_size=10,         # ignore sub-pixel noise detections
            )
    except Exception as e:
        abort(500, f"OCR failed: {e}")

    # 4. Filter low-confidence detections using a per-language threshold.
    #    Languages with complex diacritics (Vietnamese, Thai) use a higher
    #    cutoff to suppress false positives; dense-script languages (Korean)
    #    use a lower cutoff because genuine text can score conservatively.
    min_conf = _MIN_CONF_MAP.get(lang, 0.35)
    boxes = []   # each entry: (x1, y1, x2, y2, text)
    for bbox, text, conf in raw:
        text = text.strip()
        if not text or conf < min_conf:
            continue
        # bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append((min(xs), min(ys), max(xs), max(ys), text))

    # 4b. Fallback retry — only when preprocessing returned literally zero boxes.
    #     Threshold is 0, not 2, so wordless art pages (which correctly have no
    #     text) are never double-processed.  We only retry when OCR found nothing
    #     at all, suggesting preprocessing may have hurt rather than helped
    #     (e.g. an unusual panel where the selected channel was counterproductive).
    #     Uses the raw original image + EasyOCR's own max internal contrast boost.
    if len(boxes) == 0:
        print(f"  [OCR] Zero boxes from preprocessed image — retrying on raw "
              f"(lang={lang})")
        try:
            arr_raw = np.array(pil)
            with _infer_lock:
                raw2 = reader.readtext(
                    arr_raw,
                    detail=1,
                    paragraph=False,
                    contrast_ths=0.05,
                    adjust_contrast=1.0,
                    text_threshold=0.5,
                    min_size=8,
                )
            for bbox, text, conf in raw2:
                text = text.strip()
                if not text or conf < max(min_conf - 0.05, 0.20):
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                boxes.append((min(xs), min(ys), max(xs), max(ys), text))
            if boxes:
                print(f"  [OCR] Raw fallback recovered {len(boxes)} box(es)")
        except Exception as e:
            print(f"  [OCR] Raw fallback failed: {e}")

    # 5. Build raw_box output (percentage + pixel coords) before merging.
    #    The frontend stores these to support the correction UI split feature.
    raw_boxes_out = [
        {
            "id":  idx,
            "text": b[4],
            "box": [
                round(b[0] / w * 100, 1), round(b[1] / h * 100, 1),
                round(b[2] / w * 100, 1), round(b[3] / h * 100, 1),
            ],
            "px":  [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
        }
        for idx, b in enumerate(boxes)
    ]

    # 6. Merge nearby boxes — fragments from the same speech bubble get
    #    clustered together using union-find on expanded bounding boxes,
    #    with panel borders acting as hard merge barriers.
    regions, group_raw_ids = _merge_bubble_regions(
        boxes, w, h, h_borders, v_borders, margin_scale
    )

    # 7. Attach raw_box_ids so the frontend knows which raw fragments
    #    belong to each merged region (needed for the split correction tool).
    for region, raw_ids in zip(regions, group_raw_ids):
        region["raw_box_ids"] = raw_ids

    return jsonify({"regions": regions, "raw_boxes": raw_boxes_out})


@app.route("/ocr-crop", methods=["POST"])
def ocr_crop():
    """
    POST body:  { "url": "https://cdn…/page.jpg",
                  "box": [x1, y1, x2, y2],   # pixel coords
                  "lang": "vi" }
    Response:   { "text": "recognized text" }

    Crops the image to the given pixel box and runs OCR on just that region.
    Called by the correction UI when the user draws a new bounding box.
    """
    body      = request.get_json(force=True, silent=True) or {}
    image_url = body.get("url", "").strip()
    box       = body.get("box", [])
    lang      = body.get("lang", "en").lower()

    if not image_url.startswith("https://"):
        abort(400, "url must be an HTTPS URL.")
    if len(box) != 4:
        abort(400, "box must be [x1, y1, x2, y2] in pixels.")

    # Download
    try:
        img_r = requests.get(image_url, timeout=20,
                             headers={"User-Agent": "MangaTL-Reader/1.0"})
        img_r.raise_for_status()
    except requests.RequestException as e:
        abort(502, f"Image download failed: {e}")

    # Decode + crop
    try:
        pil      = Image.open(io.BytesIO(img_r.content)).convert("RGB")
        iw, ih   = pil.size
        x1, y1, x2, y2 = (max(0, min(int(v), d - 1))
                           for v, d in zip(box, [iw, ih, iw, ih]))
        if x2 <= x1 or y2 <= y1:
            abort(400, "Crop box has zero area after clamping.")
        crop = pil.crop((x1, y1, x2, y2))
        arr  = _preprocess_for_ocr(np.array(crop))
    except Exception as e:
        abort(422, f"Image decode/crop error: {e}")

    # OCR the crop (serialised)
    try:
        reader = _get_reader(lang)
        with _infer_lock:
            raw = reader.readtext(arr, detail=1, paragraph=False,
                                  contrast_ths=0.1, adjust_contrast=0.5,
                                  text_threshold=0.6, min_size=10)
    except Exception as e:
        abort(500, f"OCR failed: {e}")

    min_conf = _MIN_CONF_MAP.get(lang, 0.35)
    texts    = [t.strip() for _, t, c in raw if c >= min_conf and t.strip()]
    return jsonify({"text": " ".join(texts)})


# ─── Startup helpers ──────────────────────────────────────────────────────────

# FIX #10 — check for port conflict before Flask tries to bind
def _port_in_use(port: int) -> bool:
    """Return True if something is already listening on HOST:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) == 0


# FIX #11 — poll the socket instead of using a fixed 1-second sleep,
#            so the browser opens as soon as Flask is actually ready
def _open_when_ready():
    """Poll until the server accepts connections, then open the browser."""
    for _ in range(30):          # up to 3 seconds total
        try:
            with socket.create_connection((HOST, PORT), timeout=0.1):
                webbrowser.open(f"http://{HOST}:{PORT}")
                return
        except OSError:
            time.sleep(0.1)


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # FIX #10 — friendly error instead of a cryptic socket traceback
    if _port_in_use(PORT):
        print(f"\n  ✗  Port {PORT} is already in use.")
        print(f"     Stop the other process, or change PORT at the top of this script.\n")
        sys.exit(1)

    addr = f"http://{HOST}:{PORT}"
    print(f"\n  MangaTL  →  {addr}")
    print(  "  Ctrl+C to stop\n")
    threading.Thread(target=_open_when_ready, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
