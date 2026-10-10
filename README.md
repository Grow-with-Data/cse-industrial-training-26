<p align="center">
  <img src="assets/logo.png" width="84" alt="Grow with Data">
</p>

<h1 align="center">AI/LLM Application Builder — From Zero to Mastery</h1>
<p align="center">Student workspace · CSE Industrial Training 2026</p>

<p align="center">
  10 weeks · 20 sessions · 40 contact hours · Friday &amp; Saturday, 7:30–9:30 PM · live, instructor-led<br>
  <sub>Industrial training for the Department of Computer Science and Engineering</sub>
</p>

<p align="center">
  <a href="https://growwithdata.net"><b>Grow with Data</b></a> ·
  <a href="brochure.html">Brochure</a> ·
  <a href="https://how-llm-works.kalimamzad.com/">How LLMs work</a>
</p>

<p align="center">
  <sub>
    Instructor <a href="https://kalimamzad.com">Md Kalim Amzad Chy</a>
    (<a href="https://linkedin.com/in/kalimamzad">LinkedIn</a>) ·
    Co-instructor <b>Pranab Barua Arthi</b>
  </sub>
</p>

---

One project, **ShopWise** — an e-commerce customer-support assistant — built across 20 sessions,
from a first API call to a deployed app with a React front end. Each session lands here on the day
it is taught.

- **New to LLMs?** Start with [**How LLMs work**](https://how-llm-works.kalimamzad.com/) — a
  separate talk, no code, and it makes session 1 land properly.
- **Class recordings** are shared after each session.
- **Everything installs once.** There is no per-session setup, and no Python to install.

---

## Setup (once, ~15 minutes)

| | Install | Why |
|---|---|---|
| 1 | [**Cursor**](https://cursor.com) or [**VS Code**](https://code.visualstudio.com) | where you run the notebooks |
| 2 | [**Git**](https://git-scm.com/downloads) | to get this repo and each new session |
| 3 | [**uv**](https://docs.astral.sh/uv/) | builds the Python environment (step 1 below) |
| 4 | [**Gemini API key**](https://aistudio.google.com/apikey) | free, no card |

> **Do not install Python.** `uv` downloads the exact version this course uses. Installing your own
> first is the most common way to end up with two Pythons and a notebook running the wrong one.

**1 · Install uv**

```bash
# macOS / Linux — Terminal
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows — PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the terminal and open a new one — the installer changes your PATH. Check: `uv --version`.

**2 · Clone the repo** — it's private, so sign in with the GitHub account you gave us.

- In Cursor/VS Code: **Ctrl/Cmd+Shift+P** → `Git: Clone` → paste the URL → sign in when the browser
  opens → pick a folder → **Open**.
- URL: `https://github.com/Grow-with-Data/cse-industrial-training-26.git`
- Each week, get the new session: **Source Control** → `⋯` → **Pull** (or `git pull`).
- Clone refused? Your GitHub account isn't on the repo yet — send your username to your instructor.
- **Windows:** clone into a plain folder such as `C:\dev\`, **not** a OneDrive-synced Desktop or
  Documents. OneDrive syncing a 1 GB `.venv` will break the environment and crawl.

**3 · Add two editor extensions** — **Ctrl/Cmd+Shift+X**, install **Python** and **Jupyter**. These
are plugins for Cursor/VS Code, *not* Python itself — they teach the editor to run notebooks. Without
Jupyter, `.ipynb` files open as unreadable JSON.

**4 · Build the environment** — open the terminal (**Ctrl+`**) in the repo folder:

```bash
uv sync
```

**This is where Python arrives.** uv reads `requires-python = ">=3.12"` from `pyproject.toml`,
downloads a matching Python if your machine hasn't got one, and builds `.venv/` from it. The
download lands in uv's own folder — nothing is installed system-wide, and no other Python on your
machine is touched.

Then it installs every package for all 20 sessions from `uv.lock`, so no session opens with a failed
install and the whole cohort is on identical versions. ~1 minute on a good connection and about
1 GB — have 2 GB free. If it dies halfway, run it again; it picks up where it stopped.

**5 · Add your key**

```bash
cp .env.example .env            # macOS / Linux
Copy-Item .env.example .env     # Windows
```

Paste your key after `GEMINI_API_KEY=` — no quotes, no spaces. Never commit `.env`; it is
gitignored, keep it that way.

**6 · Check it, before class**

```bash
cd session-01
uv run python check_setup.py
```

PASS/FAIL for every requirement, with the fix printed beside each failure. Uses no network and no
quota. Fix the first FAIL, run again.

**7 · Open a notebook** — either way works, use whichever you prefer.

**In the editor** — open `session-01/session-01.ipynb` → **Select Kernel**, top right → **Python
Environments** → the one ending in `cse-industrial-training-26/.venv` → **Shift+Enter** runs a cell.

**In the browser** — JupyterLab, from the repo root:

```bash
uv run jupyter lab
```

It opens in your browser; click any `session-NN/session-NN.ipynb`. There is **no kernel to pick** —
started this way, Jupyter is already running inside `.venv`, and that is the kernel the notebooks
ask for. Stop it with **Ctrl+C** in the terminal when you're done.

### Running code: `uv run` or activate

Both work. `uv run` is the one to learn.

```bash
uv run python check_setup.py          # no activation, ever
```

- Works from **any folder inside the repo** — uv walks up to `pyproject.toml` and uses *this*
  project's `.venv`, whichever session folder you are in.
- Checks the environment matches `uv.lock` first, so a missing package installs itself.
- **Outside the repo it does not.** Run it from your home folder and uv falls back to whatever
  Python it can find — which is how you end up running Anaconda's. `cd` into the repo first.

```bash
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows
python check_setup.py
deactivate
```

- The classic way. Your prompt shows `(.venv)` while it is on, and stays that way until you
  `deactivate` or close the window.
- Same interpreter, same packages — just manual. Forget to activate and `python` is your system
  one, with none of this installed.

Notebooks are the same idea by other means: `uv run jupyter lab` launches the browser route, and
the editor's kernel picker points at the same `.venv`.

### If something breaks

Nine times in ten it is the kernel: your notebook must run the `.venv` Python, not a system or
Anaconda one. Every notebook's first cell prints which interpreter it is using — read that first.

| Error | Fix |
|---|---|
| `uv: command not found` | open a new terminal (PATH) |
| `.ipynb` opens as JSON | install the Jupyter extension — step 3 |
| `ModuleNotFoundError: langchain` | wrong kernel — step 7 |
| `AttributeError: 'Client' object has no attribute 'interactions'` | wrong kernel, older `google-genai` — step 7 |
| `DefaultCredentialsError: Your default credentials were not found` | wrong kernel: Anaconda ships LangChain 0.3, which never reads `GEMINI_API_KEY` — step 7 |
| A cell misbehaves | **Kernel → Restart and Run All** before debugging |
| `Repository not found` when cloning | your GitHub account isn't on the repo — send your username to your instructor |
| `429` / `RESOURCE_EXHAUSTED` | free-tier rate limit. Wait a minute and re-run — you have not broken anything |
| You pasted the key but a cell still says no key | restart the kernel — see below |
| `uv sync` stops partway | run it again; it resumes |

### Two habits that prevent most of it

**Restart the kernel after editing `.env`.** Python reads that file once, when the notebook first
imports it. Edit the key with the notebook open and the running kernel still holds the old value —
**Kernel → Restart**, then run from the top.

**Keep your own work in a copy.** Before you experiment in a lesson notebook, duplicate it:
`session-04/my-session-04.ipynb`. Next week's `git pull` updates the files we ship, and git will
refuse to overwrite one you have edited. If that happens:

```bash
git stash        # park your changes
git pull
git stash pop    # bring them back (resolve any conflict in the editor)
```

Still stuck? Screenshot the **whole** error and bring it to the next session. A setup problem is
worth ten minutes of class time and never a whole evening of yours.

---

## How to work

Every session folder is self-contained — open the one you are on, it runs on its own. You never need
`git checkout` to catch up.

```
session-05/
├── session-05.ipynb    # the lesson — follow along in class
├── exercises.ipynb     # after class (⭐ warm-up · ⭐⭐ apply · ⭐⭐⭐ stretch)
├── slides/index.html   # the deck — any browser, works offline
└── project/            # ShopWise as it stands after this session — runnable
```

- **Notebooks ship with their outputs**, so you can read any session without a key. Run every cell
  yourself anyway — watching your own code produce the output teaches far more than reading it.
- **Solutions are published one session late**, after the exercise review, so the exercises are
  worth attempting cold.
- **`project/` is a snapshot, not a shared codebase.** `session-06/project/` is `session-05/project/`
  plus that session's changes — `diff -r session-05/project session-06/project` shows exactly what
  moved.

---

## How to submit your exercises

You hand in exercises through **your own private repo**, which your instructor creates for you. Only
you and your instructors can see it: classmates can't open your work, and you can't open theirs.
This repo stays read-only for everyone.

1. **Once:** send your GitHub username to your instructor. You'll get an email invitation to
   `cse26-exercises-<your-github-username>` in the Grow-with-Data organization: accept it, then clone that
   repo next to this one.
2. **Each session:** do the exercises here, in a copy (`session-03/my-exercises.ipynb`), because the
   notebook needs this repo's `project/` and `data/` folders.
3. **Submit:** copy the finished notebook into your repo as `session-03/exercises.ipynb`, then commit
   and push. The deadline is the start of the next session; the last push before it is what we mark.

Your submission repo's README has the exact commands. Save the notebook **with its outputs**, and
put your answers in the ✍️ cells in your own words: that is where the marks are. Never push `.env`.

---

## Assessment

| Component | Weight | What it is |
|---|---|---|
| Attendance | **10%** | 20 sessions, Fri + Sat |
| Assignments | **20%** | the per-session exercises, submitted to your private repo ([how](#how-to-submit-your-exercises)) |
| Final project | **50%** | your capstone, demoed in session 20 |
| Viva (industrial) | **20%** | on what you built here |

> Your **department runs its own final viva** as well. That is a separate examination, separately
> marked, and is not the industrial viva above.

The capstone is briefed in session 16: extend your deployed assistant with at least two additions of
your own. Marked on working end-to-end, depth of the extension, code quality, debugging from traces,
and the presentation.

---

## Session map

Two sessions a week, Friday + Saturday, 19:30–21:30.

| # | Session | What you learn | ShopWise |
|---|---|---|---|
| S01 | Hello, AI | how LLMs actually work · setup | — |
| S02 | Talking to the machines | calling the API properly · prompt engineering from the OpenAI, Anthropic and Google guides | v0.1 |
| S03 | From chat to software | Pydantic in depth · structured output you can trust · extracting facts and checking them · prompt caching | v0.2 |
| S04 | One interface to rule them all | the framework trade · observability | v1 |
| S05 | Give it hands | tools and the agent loop | v2 |
| S06 | Remember and respond | state and short-term memory | v3 |
| S07 | Safety rails | middleware · human-in-the-loop | v4 |
| S08 | It should know your policies | embeddings and retrieval | v5 |
| S09 | Answer with receipts | grounded generation, with citations | v6 |
| S10 | Open the black box | graphs — the agent, hand-built | v7 |
| S11 | Pause, resume, remember forever | interrupts · long-term memory | v8 |
| S12 | A team of specialists | multi-agent systems | v9 |
| S13 | It needs an API | serving with FastAPI | v10 |
| S14 | From PRD to tasks | a product spec, and Claude Code as collaborator | v11 |
| S15 | A face for your assistant | spec → generate → run → steer, on a React UI | v12 |
| S16 | Ship it | deployment · the hosted prompt registry | v13 |
| S17 | Your own model, part 1 | when to fine-tune · LoRA / QLoRA | v14 |
| S18 | Your own model, part 2 | honest evaluation · serving your own model | v15 |
| S19 | The AI pair becomes a teammate | agentic engineering workflow · MCP | v16 |
| S20 | Demo day | your capstone | capstone |

**What you need, and when:** `GEMINI_API_KEY` from session 1 · a free LangSmith account from
session 4 · a deploy account and Qdrant Cloud (free tiers) at session 16. Qdrant in session 8 runs
**locally, no signup**, and sessions 17–18 train on a free **Google Colab** GPU and serve the result
through **Ollama** on your own machine — no keys for either.

---

## Cost

The course runs on **Gemini's free tier** throughout, on `gemini-3.1-flash-lite`. Staying inside the
notebooks should cost you nothing.

- **Search grounding** is free only on `gemini-2.5-flash`/`-flash-lite` (500 requests/day) and
  **image generation is free on no model.** Both are instructor demos, marked 💳; those cells explain
  themselves instead of crashing on a free key.
- Free-tier limits are no longer published — check yours at
  [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) and
  in your AI Studio dashboard.

> ⚠️ **Free-tier prompts are used to train Google's models** — Google lists "content used to improve
> our products" as **Yes** for free, **No** for paid. Never paste real customer data, real names, or
> anything belonging to an employer into a free-tier notebook. Every ticket in `data/tickets.yml` is
> invented for this reason.

---

## Program brochure

What the programme covers, who runs it, and what ShopWise is — useful when someone asks what you
spent eight weeks on.

- [`brochure.html`](brochure.html) — any browser, works offline
- [`brochure.pdf`](brochure.pdf) — 5 pages, for printing or forwarding
