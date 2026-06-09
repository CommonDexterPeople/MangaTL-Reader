# MangaTL-Reader

**Read any MangaDex chapter — even if there's no English translation.**  
Paste a chapter URL, get AI-powered translations in real-time. Free, runs locally, no subscription.

---

## Quick Start

1. **Download** `MangaTL-Reader_V1.py`
2. **Run it** — double-click on Windows, or `python MangaTL-Reader_V1.py` on Mac/Linux
3. **Get a free Gemini API key** at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — no credit card needed
4. **Paste any MangaDex chapter URL** into the app
5. **Read**

All dependencies install automatically on first run. No config files, no GPU required.

---

## Introduction

Hi, I'm a 19-year-old student currently in Pre-University. I built this tool to help others read manga on MangaDex that *has* a translation — just not in English.

Three years ago I ran into the same problem. Some manga I genuinely loved had stopped being translated into English. The only available translations were in languages I didn't know. After failing to find a free alternative that didn't require payment, bombard me with ads, or offer barely readable machine translations — I gave up.

Fast forward to 2026. AI has genuinely gotten better. People who know nothing about coding can build tools for themselves. It's basically a superpower. And I thought — why do we still need to struggle finding a solution when we can make one ourselves?

So, with nothing but my laptop, five dollars, basic Python knowledge, and a lot of prompting, I built this. It may not match paid translators, but I hope it helps you read something you couldn't before.

— *a 19-year-old Pre-University student*

---

# MangaTL-Reader — User Manual

MangaTL-Reader is a single-file local web app that translates any MangaDex chapter into your language of choice. It downloads pages, extracts text with OCR, sends it to an AI for translation, and overlays numbered badges directly on the page images.

---

## Table of Contents

1. [Requirements](#requirements)
2. [First Run](#first-run)
3. [Getting an API Key](#getting-an-api-key)
4. [Home Screen Settings](#home-screen-settings)
5. [The Translation Pipeline](#the-translation-pipeline)
6. [Reading the Results](#reading-the-results)
7. [Correction Tools](#correction-tools)
8. [Badge Reading Order](#badge-reading-order)
9. [Cache System](#cache-system)
10. [AI Models — Which to Use](#ai-models--which-to-use)
11. [Vision OCR Modes](#vision-ocr-modes)
12. [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python 3.14 or newer**
- Internet connection
- A Gemini or DeepSeek API key (see below)

No other manual setup is needed. Everything else installs itself on first run.

---

## First Run

**Windows:** Double-click `MangaTL-Reader_V1.py`. If Python is installed, it opens automatically.

**Mac / Linux:**
```
python MangaTL-Reader_V1.py
```

The first time you run it, the app will install all required packages automatically — Flask, EasyOCR, OpenCV, Pillow, NumPy, and Requests. This takes **2–5 minutes** depending on your connection because EasyOCR includes a language model (~100–400 MB).

You will see a setup banner in the terminal:
```
╔══════════════════════════════════════════════╗
║      MangaTL — First-Time Setup              ║
╚══════════════════════════════════════════════╝
Installing 6 missing package(s): …
```

Once complete, the app starts a local server and opens your browser automatically. Subsequent launches are instant — packages are only installed if missing.

---

## Getting an API Key

The app needs at least one AI provider key to translate. OCR (text extraction) does not need a key by default — it runs locally. A key is only needed for OCR if you enable Vision OCR mode (see [Vision OCR Modes](#vision-ocr-modes)).

### Gemini (Recommended for most users)

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key (starts with `AIza…`)

**Free tier:** No credit card needed. Generous daily limits — enough for casual reading. If you read heavily, the free tier can run low (see [Vision OCR Modes](#vision-ocr-modes) for how to manage this).

**Paid tier:** Requires billing enabled in Google Cloud. Dramatically higher limits and no daily cap in practice.

### DeepSeek (Budget alternative)

1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Create an account and top up with a small amount (chapters typically cost ~$0.02–0.05 each)
3. Generate an API key (starts with `sk-…`)

DeepSeek does not offer a free tier. It is a pay-per-use service, but it is extremely cheap for manga translation.

---

## Home Screen Settings

When you open the app, you are on the home screen. All settings here are saved automatically and restored the next time you launch.

### AI Model

A dropdown with all available models, grouped by provider and tier.

**Gemini (Google AI)**

| Option | Notes |
|--------|-------|
| Gemini 3.5 Flash — Free tier | Best balance of quality and speed on the free tier |
| Gemini 3.1 Flash-Lite — Free tier | Faster and cheaper; slightly lower quality |
| Gemini 2.5 Flash — Paid | High quality; requires billing enabled |
| Gemini 3.1 Pro — Paid | Flagship model; best output quality, slowest |

**DeepSeek**

| Option | Notes |
|--------|-------|
| DeepSeek V4 Flash | Fastest; good for straightforward manga |
| DeepSeek V4 Pro | Best translation quality from DeepSeek |

### API Key

Paste your key here. The field saves your key per provider — switching between Gemini and DeepSeek automatically loads the correct key for each so you do not have to re-enter it every time.

The key is never exposed in your browser's network tab. It is forwarded through the local Flask proxy server, which means it stays on your own machine.

### Vision OCR *(Gemini only)*

Controls when Gemini Vision is used to read text from pages. Only visible when a Gemini model is selected. See [Vision OCR Modes](#vision-ocr-modes) for a full explanation. Options:

- **Smart — complex scripts only (saves quota)** ← default
- **All languages — max quality**
- **Off — EasyOCR only, no quota used**

### MangaDex Chapter URL

Paste the URL of any MangaDex chapter page. The format is:
```
https://mangadex.org/chapter/[chapter-id]/[page]
```
The chapter ID is the important part — the page number at the end is ignored.

### Translate To

The language you want the chapter translated into. Includes a long list of common languages plus a **Custom** option at the bottom where you can type any language name (e.g. `Javanese`, `Scots`, `Malay`).

> **Note:** Translation quality is highest when targeting English. Quality may vary for other languages depending on how well-supported they are by the AI model.

### Image Quality

- **Data Saver (faster):** Downloads compressed images from MangaDex's data-saver CDN. Smaller files, faster page loads, slightly lower resolution.
- **Full Quality:** Downloads full-resolution images. Better OCR accuracy on high-detail pages at the cost of slower downloads.

### Bubble Merge Sensitivity

A slider (0.10 – 1.50, default 0.50) that controls how aggressively nearby text boxes are merged into a single speech bubble region before translation.

- **Lower values (toward 0.10):** Each OCR box stays separate. Use this if regions are incorrectly being merged together.
- **Higher values (toward 1.50):** Boxes that are far apart will still be merged. Use this if a single speech bubble is being split into multiple separate regions.

### Badge Reading Order

Sets the order in which the numbered badges (①②③…) are assigned to text regions. This also determines the order of translations in the right-hand panel.

- **AUTO ← RTL:** Right-to-left, then top-to-bottom. Standard Japanese manga reading order.
- **AUTO LTR →:** Left-to-right, then top-to-bottom. Correct for manhwa, webtoons, and Western-style comics.
- **MANUAL ↕ DRAG:** Keeps the raw order returned by OCR. On each page, a **⇅ ORDER** button appears that lets you drag badges into any custom order.

This setting persists across all chapters until you change it.

### MangaDex Account *(optional)*

Guest access to MangaDex is limited to 10 chapters before you hit a rate limit. Logging in with your own MangaDex account removes this limit.

To use this:
1. Go to [mangadex.org/settings](https://mangadex.org/settings) and create a **personal API client** under API Clients
2. Copy the **Client ID** and **Client Secret**
3. Expand the MangaDex Account section in the app
4. Enter your Client ID, Client Secret, username, and password
5. Click **Login**

The login status shows as a green dot with your username when active. Credentials are saved locally.

### Cache Strip

Shows how many chapters are currently cached. Translated chapters are stored in your browser's `localStorage` so you can re-read them instantly without re-running OCR or translation.

The **🗑 CLEAR** button wipes all cached chapters. See [Cache System](#cache-system) for details.

---

## The Translation Pipeline

When you click **TRANSLATE CHAPTER**, the app processes every page in sequence:

**1. Fetch chapter data** — The app queries the MangaDex API to get the list of page image URLs.

**2. Download the page image** — Each page is downloaded via the local Flask proxy (never directly from the browser, to avoid CORS issues).

**3. OCR — text extraction** — Depending on your Vision OCR setting and the source language:
- *EasyOCR path:* The image goes through adaptive preprocessing (CLAHE contrast enhancement + denoising), then EasyOCR scans for text regions. The raw bounding boxes are filtered by a per-language confidence threshold, then nearby boxes belonging to the same speech bubble are merged using a union-find algorithm with panel-border awareness.
- *Gemini Vision path:* The page image is sent directly to Gemini and it returns text + approximate centre positions. Handles vertical text, stylised fonts, and complex scripts better than EasyOCR.

**4. AI translation** — The extracted text regions (with their position data) are sent to your chosen AI model in a single batch. The AI translates all bubbles at once and classifies each one as `speech`, `thought`, `sfx`, `narration`, or `sign`. Position data (where each bubble sits on the page) helps the AI infer reading order and narrative flow.

**5. Render** — Numbered badges are overlaid on the page image. The translation panel on the right shows each translation with its type colour. The page is cached for instant future access.

---

## Reading the Results

### Numbered Badges

Each detected text region gets a badge overlaid on the page image at the position of the original text. The number corresponds to the entry in the translation panel on the right.

Badge colours indicate the text type classified by the AI:

| Colour | Type | Meaning |
|--------|------|---------|
| 🟣 Purple | Speech | Dialogue in speech bubbles |
| 🔵 Blue | Thought | Internal monologue (cloud/wavy bubbles) |
| 🔴 Red | Sound Effect | Onomatopoeia — *Crash*, *Rumble*, *Thud* |
| 🟡 Amber | Narration | Caption boxes, story narration text |
| 🟢 Green | Sign | Labels, environmental text, signs |

Sound effect (SFX) regions are rendered as small compact badges since they often crowd the edges of a panel. All other types render as standard-sized numbered badges.

### Translation Panel

The panel to the right of each page lists all translations in reading order. Each entry shows the badge number, type tag (coloured to match), and the translated text.

---

## Correction Tools

Every page that has been translated has a **✏ CORRECT** button. Click it to open the correction overlay — a set of tools for fixing OCR errors, wrong translations, and misdetected regions.

### Toolbar Modes

**SELECT** (default) — Click any region on the page to select it. The sidebar shows the original OCR text, the current translation, and action buttons.

**DRAW** — Click and drag on the page to draw a new bounding box. A new region is created with a text field so you can type the text manually, then translate it. Use this when OCR missed a bubble entirely.

**DELETE** — All regions turn red on hover. Click any region to delete it (removes it from the translation panel; the badge disappears).

**ORDER** — Shows a list of all regions with ↑↓ buttons. Use this to manually adjust the reading order within a page when the AI numbered the bubbles incorrectly. Changes take effect immediately in the translation panel.

### Per-Region Actions (SELECT mode)

When a region is selected, the sidebar shows several action buttons:

**SPLIT** — Appears when a region contains two separate lines. Creates a horizontal split point that you drag to the correct position, then divides the region into two independent regions. Re-translate after splitting.

**MERGE with…** — A dropdown lists nearby regions. Choose one and click **MERGE** to combine both regions into a single bounding box with their text joined. Useful when OCR split one speech bubble into two fragments. Re-translate after merging.

**DELETE REGION** — Removes this specific region. Same as DELETE mode but without switching modes.

**↺ (retranslate this region)** — Re-sends this single bubble to the AI for a fresh translation. The AI receives the current state of all other translated regions on the same page as context, so names, pronouns, and character register stay consistent with the rest of the page.

### ↺ RE-TRANSLATE (full page)

The **↺ RE-TRANSLATE** button at the bottom of the correction panel retranslates every non-deleted region on the page from scratch. Useful if the initial translation pass was poor across the whole page or if you have edited several OCR texts and want fresh translations for all of them.

All corrections (merges, splits, deletes, text edits, reorders) are saved automatically to `localStorage` and applied whenever the page is rendered, including on cached re-reads.

---

## Badge Reading Order

The global reading order setting (home screen) controls how badges are automatically numbered:

**AUTO ← RTL** assigns numbers right-to-left within each row before moving down the page. Correct for most Japanese manga where the right column reads before the left.

**AUTO LTR →** assigns numbers left-to-right within each row. Use for manhwa, Korean webtoons, and any comic that reads in Western order.

**MANUAL ↕ DRAG** skips automatic sorting entirely — badges keep the raw order returned by the OCR engine. On every page, a **⇅ ORDER** button appears at the top right. Clicking it opens a draggable list of all badges. Drag the handles to set any order. The translation panel updates live as you reorder. Manual orderings persist in your cache.

> **Tip:** If most of your manga reads correctly in AUTO RTL but one page has an unusual panel layout, switch the global setting to MANUAL and use ⇅ ORDER just for that page.

---

## Cache System

Translated results are stored in your browser's `localStorage` under the key `mtl_ch_[chapter-id]`.

**What is cached:** The full list of translated regions (text, translation, type, position) for every page in the chapter. All corrections you made via the ✏ CORRECT panel are included.

**What is not cached:** The page images themselves. Images are always re-fetched from MangaDex CDN on revisit. This keeps the cache small.

**Expiry:** Each cache entry expires after **7 days**. Expired entries are silently removed on next load.

**Capacity:** The cache holds up to **20 chapters**. When the limit is reached, the oldest chapter is evicted automatically to make room.

**Clearing the cache:** Use the 🗑 CLEAR button on the home screen, or the **✕ clear** button inside the reader. This removes all cached chapters immediately.

Cached chapters load instantly with no API calls — useful for re-reading or sharing your translated version offline.

---

## AI Models — Which to Use

### For free-tier Gemini users

Start with **Gemini 3.5 Flash (Free tier)**. It gives the best translation quality available without a credit card. If you hit rate limits mid-chapter, try switching to **Gemini 3.1 Flash-Lite** which is faster and uses less quota per request.

Free tier daily limits (approximate):
- ~250 requests/day on most Flash models
- Each chapter page = 1 translation request + 1 Vision OCR request (if Vision OCR is enabled)
- A 20-page chapter in Smart mode (Vision for ja/zh only) = ~20–40 requests
- A 20-page chapter with Vision OCR off = ~20 requests

### For paid Gemini users

**Gemini 2.5 Flash** is the best everyday choice — fast, high quality, and cost-effective. Use **Gemini 3.1 Pro** if you want the absolute best translation output and do not mind slower processing.

### For DeepSeek users

**DeepSeek V4 Pro** produces the best translation quality, especially for complex phrasing. **DeepSeek V4 Flash** is a good budget option for simpler manga. DeepSeek does not support Vision OCR — the app automatically uses EasyOCR for all pages regardless of the Vision OCR setting.

---

## Vision OCR Modes

Vision OCR is only available when a **Gemini** model is selected. DeepSeek users always use EasyOCR.

### Smart — complex scripts only *(default)*

Gemini Vision is used for languages where EasyOCR struggles most:
- Japanese (`ja`)
- Chinese Simplified (`zh`) and Traditional (`zh-hk`)

All other languages (Korean, Vietnamese, Indonesian, Arabic, Thai, French, Spanish, etc.) use local EasyOCR, which works well for them and uses no API quota.

**Best choice for free-tier users.** Saves roughly half your daily quota compared to All mode.

### All languages — max quality

Gemini Vision is used for every language on every page. OCR quality is uniformly higher — Gemini understands manga fonts, speech bubble layouts, and unusual text orientations better than EasyOCR in all scripts.

**Tradeoff:** Every page scan costs one extra Gemini API call on top of the translation call. On free tier, a 20-page chapter costs ~40 requests instead of ~20, halving your daily chapter capacity.

Recommended if you are on a paid Gemini tier or read primarily short chapters.

### Off — EasyOCR only

Gemini Vision is never called. All OCR is handled locally by EasyOCR at zero API cost. This is the original behaviour before Vision OCR was added.

Use this if you want to completely conserve your Gemini quota for translation, or if you are offline and reading a previously cached chapter.

---

## Troubleshooting

**First run takes a very long time**
Normal. EasyOCR is downloading its language model for the first time (~100–400 MB). Wait for the terminal to say the server is ready and the browser will open automatically. All subsequent runs start in seconds.

**Translations show `—` instead of text**
This usually happens with DeepSeek V4 Pro when a page has many text regions. The model uses its reasoning (thinking) budget on chain-of-thought and runs out of space before writing the JSON output. Fix: use **↺ RE-TRANSLATE** on the affected page in the correction panel. If it happens frequently, switch to **DeepSeek V4 Flash** which is less prone to this, or switch to a Gemini model.

**Rate limit error (429) in the terminal**
Your Gemini free tier daily quota has been exhausted. Options:
- Wait until midnight (Pacific Time) for the quota to reset
- Switch Vision OCR to **Smart** or **Off** to reduce calls per page
- Switch to a paid Gemini model with higher limits
- Use DeepSeek instead (separate quota, not affected by Gemini limits)

The app automatically falls back to EasyOCR when Vision OCR is rate-limited, so translation continues — you just get local OCR quality instead of Vision quality until the quota resets.

**Badge positions are off / overlapping the wrong text**
This is most common with Vision OCR, which returns approximate centre positions rather than exact pixel bounding boxes. Use the **✏ CORRECT → ORDER** mode to re-sequence the badges if the numbers are wrong, or **DRAW** to manually mark a missed region with a precise box.

For EasyOCR, adjust the **Bubble Merge Sensitivity** slider. If bubbles are being merged that should be separate, lower it. If one bubble is split into multiple badges, raise it.

**OCR missed a speech bubble entirely**
Open **✏ CORRECT**, switch to **DRAW** mode, and drag a box over the missed bubble. Type in the text you can read, then use the **↺** retranslate button to translate it with the rest of the page as context.

**Two separate bubbles were merged into one region**
Select the merged region in **✏ CORRECT → SELECT** mode and click **SPLIT**. Drag the split line to where the two bubbles divide, confirm, then retranslate both resulting regions.

**The chapter URL doesn't work**
Make sure you paste the full URL from the browser address bar, including `https://mangadex.org/chapter/`. The chapter ID must be a valid UUID-format string. If the chapter is locked or region-restricted by MangaDex, the download will fail regardless of login status.

**"Translate this page first, then use ✏ CORRECT"**
The correction panel only opens for pages that have already been translated in the current session or loaded from cache. If you are on a fresh load of a cached chapter, scroll to the page and let it render — cached pages populate instantly and the CORRECT button becomes available immediately.
