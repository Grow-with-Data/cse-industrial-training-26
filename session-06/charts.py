"""Two small charts for session 6, drawn as SVG so nothing extra needs installing.

    timeline(rows)     what a customer sees, second by second, for several runs
    token_bill(calls)  input tokens per model call over a long ticket

Notebook plumbing, like session 5's score_tools.py: it lives beside the
notebook, not in project/, because the assistant doesn't need it.
"""

from __future__ import annotations

from html import escape

from IPython.display import SVG

# growwithdata.net palette
DARK, MUTED, LINE = "#14212e", "#5b6b77", "#e2ede7"
GREEN, TEAL, CYAN, CYAN_SOFT = "#2e9e5b", "#2f9e8f", "#59c1e8", "#eaf7fc"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _text(x, y, s, size=12, color=DARK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{FONT}">{escape(str(s))}</text>')


def timeline(rows: list[tuple[str, dict]], title: str) -> SVG:
    """One bar per run: blank screen, then a lookup status line, then the reply.

    Each row is (label, marks), where marks has "first lookup", "first word" and
    "done" in seconds, as timed_run() returns them.
    """
    width, left, right, top, row_h = 760, 150, 30, 50, 30
    longest = max(marks["done"] for _, marks in rows)
    scale = (width - left - right) / (int(longest / 5 + 1) * 5)
    height = top + row_h * len(rows) + 60
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>',
             _text(left, 24, title, 14, weight="bold")]

    for seconds in range(0, int(longest / 5 + 1) * 5 + 1, 5):          # grid every 5 s
        x = left + seconds * scale
        parts.append(f'<line x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + row_h * len(rows)}" stroke="{LINE}"/>')
        parts.append(_text(x, top + row_h * len(rows) + 16, f"{seconds}s", 11, MUTED, "middle"))

    for i, (label, marks) in enumerate(rows):
        y = top + i * row_h
        lookup = marks["first lookup"] or marks["first word"]
        segments = [(0, lookup, LINE), (lookup, marks["first word"], CYAN),
                    (marks["first word"], marks["done"], GREEN)]
        parts.append(_text(left - 10, y + 15, label, 12, DARK, "end"))
        for start, end, color in segments:
            parts.append(f'<rect x="{left + start * scale:.1f}" y="{y}" '
                         f'width="{max((end - start) * scale, 1.5):.1f}" height="20" fill="{color}"/>')

    legend_y = top + row_h * len(rows) + 40
    for n, (color, name) in enumerate([(LINE, "nothing on screen"), (CYAN, "a lookup status could show"),
                                       (GREEN, "reply text arriving")]):
        x = left + n * 200
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(_text(x + 20, legend_y + 1, name, 12, MUTED))
    return SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(parts)}</svg>')


def token_bill(calls: list[dict], title: str) -> SVG:
    """Stacked bars, one per model call: fixed overhead, history, and how much came from cache.

    Each call is a dict with "turn", "input" (input tokens), "overhead" (the part
    every call pays: system prompt and tool schemas) and "cached".
    """
    width, left, right, top, plot_h = 760, 60, 20, 44, 220
    tallest = max(c["input"] for c in calls)
    step = 1000 if tallest > 4000 else 500
    ymax = (int(tallest / step) + 1) * step
    bar = (width - left - right) / len(calls)
    height = top + plot_h + 70
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>',
             _text(left, 22, title, 14, weight="bold")]

    for value in range(0, ymax + 1, step):
        y = top + plot_h - value / ymax * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{LINE}"/>')
        parts.append(_text(left - 6, y + 4, f"{value:,}", 10, MUTED, "end"))

    previous_turn = None
    for i, c in enumerate(calls):
        x = left + i * bar + 2
        overhead_h = c["overhead"] / ymax * plot_h
        history_h = (c["input"] - c["overhead"]) / ymax * plot_h
        base = top + plot_h
        parts.append(f'<rect x="{x:.1f}" y="{base - overhead_h:.1f}" width="{bar - 4:.1f}" '
                     f'height="{overhead_h:.1f}" fill="{TEAL}"/>')
        parts.append(f'<rect x="{x:.1f}" y="{base - overhead_h - history_h:.1f}" width="{bar - 4:.1f}" '
                     f'height="{max(history_h, 0):.1f}" fill="{CYAN}"/>')
        if c.get("cached"):
            cached_h = c["cached"] / ymax * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{base - cached_h:.1f}" width="{bar - 4:.1f}" '
                         f'height="{cached_h:.1f}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
        if c["turn"] != previous_turn:                                   # label the first call of each turn
            parts.append(_text(x + (bar - 4) / 2, base + 14, f"T{c['turn']}", 10, MUTED, "middle"))
            previous_turn = c["turn"]

    legend_y = top + plot_h + 44
    items = [(TEAL, "overhead: system prompt + tool schemas"), (CYAN, "the ticket's history"),
             ("none", "outlined: served from cache")]
    for color, name, x in zip(*zip(*items), (left, left + 300, left + 480)):
        stroke = f' stroke="{GREEN}" stroke-width="2"' if color == "none" else ""
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"{stroke}/>')
        parts.append(_text(x + 20, legend_y + 1, name, 12, MUTED))
    return SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(parts)}</svg>')
