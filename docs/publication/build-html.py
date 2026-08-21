#!/usr/bin/env python3
"""Build per-section HTML pages from the draft markdown files.

Usage (no need to install anything globally):

    uv run --with markdown --with pymdown-extensions \
        python docs/publication/build-html.py

Reads every `*.md` under `docs/publication/draft/`, sorted by filename,
converts each to an HTML body with `python-markdown` plus
`pymdownx.arithmatex` (to preserve LaTeX math through markdown
processing), wraps the body in a shared HTML template that loads
MathJax 3 from CDN, and writes the result to
`docs/publication/html/<stem>.html`.

The script also regenerates the "Technical section drafts" panel of
`docs/publication/html/index.html` so that section pages are
discoverable from the landing page.

The output pages are gitignored by the `docs/publication/` rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown

THIS_DIR = Path(__file__).resolve().parent
DRAFT_DIR = THIS_DIR / "draft"
HTML_DIR = THIS_DIR / "html"
INDEX_PATH = HTML_DIR / "index.html"

# --- Shared style + MathJax loader ------------------------------------------

PAGE_CSS = """
:root {
  --fg: #1c1c1c;
  --fg-soft: #555;
  --bg: #fafaf7;
  --rule: #d8d4c8;
  --accent: #2a4a7a;     /* steel blue for the draft section */
  --accent-soft: #6a85a8;
  --code-bg: #f1ede2;
  --max-width: 880px;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: "Source Serif Pro", "Iowan Old Style", "Georgia", serif;
  font-size: 17px;
  line-height: 1.6;
}
.wrap {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
}
header.crumb {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace;
  font-size: 0.8rem;
  color: var(--fg-soft);
  margin-bottom: 1.5rem;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
header.crumb a { color: var(--fg-soft); }
h1, h2, h3, h4 {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace;
  font-weight: 600;
  letter-spacing: -0.01em;
  line-height: 1.25;
}
h1 { font-size: 1.75rem; margin: 0 0 1rem; color: var(--accent); }
h2 {
  font-size: 1.25rem;
  margin: 2.5rem 0 0.8rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--rule);
}
h3 { font-size: 1.05rem; margin: 2rem 0 0.6rem; color: var(--accent); }
h4 { font-size: 0.95rem; margin: 1.4rem 0 0.4rem; color: var(--fg-soft); }
p { margin: 0.8rem 0; }
a { color: var(--accent); }
a:hover { color: var(--accent-soft); }
strong { color: #000; }
em { font-style: italic; }
code {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace;
  background: var(--code-bg);
  border-radius: 3px;
  padding: 0 0.3em;
  font-size: 0.9em;
}
pre code { background: none; padding: 0; }
pre {
  background: var(--code-bg);
  border-left: 3px solid var(--accent-soft);
  padding: 0.7rem 1rem;
  overflow-x: auto;
  font-size: 0.85rem;
  line-height: 1.5;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: 0.93rem;
}
th, td {
  border-bottom: 1px solid var(--rule);
  padding: 0.5rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
th {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace;
  font-size: 0.85rem;
  background: var(--code-bg);
}
ol, ul { line-height: 1.6; }
li { margin: 0.3rem 0; }
hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 2.5rem 0;
}
.arithmatex { font-size: 1em; }
.MJX-Container { font-size: inherit !important; }
.site-name {
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  font-size: 0.78rem;
}
footer.page-footer {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", monospace;
  color: var(--fg-soft);
  font-size: 0.78rem;
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}
"""

MATHJAX_CONFIG = """
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'arithmatex'
  },
  svg: { fontCache: 'global' }
};
"""

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} — ISOSTER documentation</title>
<style>
{css}
</style>
<script>
{mathjax_config}
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="wrap">
<header class="crumb">
  <span>
    <a href="index.html" class="site-name">ISOSTER</a>
    &nbsp;·&nbsp; <a href="index.html">documentation home</a>
  </span>
  <span>
{prev_link}{next_link}
  </span>
</header>
{body}
<hr>
<footer class="page-footer">
  <span>ISOSTER documentation</span>
  &nbsp;·&nbsp;
  <span>source: <code>docs/publication/draft/{stem}.md</code></span>
</footer>
</div>
</body>
</html>
"""


def extract_title(md_text: str, fallback: str) -> str:
    """Pick the first `# ...` line as the page title; otherwise fallback."""
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def md_to_html(md_text: str) -> str:
    """Convert markdown to HTML with table, code, and math support."""
    md = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "attr_list",
            "sane_lists",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,  # Wrap math in \(...\) and \[...\] for MathJax.
                "preview": False,
            },
        },
    )
    return md.convert(md_text)


def build_section_page(
    md_path: Path,
    out_path: Path,
    prev_stem: str | None,
    next_stem: str | None,
) -> str:
    """Convert one markdown section to a styled HTML page. Returns the title."""
    md_text = md_path.read_text(encoding="utf-8")
    title = extract_title(md_text, fallback=md_path.stem)
    body = md_to_html(md_text)

    prev_link = (
        f'    <a href="{prev_stem}.html">← prev</a>\n'
        if prev_stem
        else ""
    )
    next_link = (
        f'    &nbsp;·&nbsp; <a href="{next_stem}.html">next →</a>\n'
        if next_stem
        else ""
    )

    page = PAGE_TEMPLATE.format(
        title=title,
        css=PAGE_CSS,
        mathjax_config=MATHJAX_CONFIG,
        prev_link=prev_link,
        next_link=next_link,
        body=body,
        stem=md_path.stem,
    )
    out_path.write_text(page, encoding="utf-8")
    return title


# --- Index-page panel regeneration ------------------------------------------

# Current marker pair. Old `<!-- DRAFT-PANEL-* -->` markers are also detected
# for one-shot migration.
INDEX_PANEL_START = "<!-- TECHNICAL-PANEL-START -->"
INDEX_PANEL_END = "<!-- TECHNICAL-PANEL-END -->"
_LEGACY_PANEL_START = "<!-- DRAFT-PANEL-START -->"
_LEGACY_PANEL_END = "<!-- DRAFT-PANEL-END -->"


def _section_number_from_stem(stem: str) -> str:
    """Extract '1.4.5' (etc.) from 'section-1.4.5-multiband'."""
    parts = stem.split("-", 2)
    return parts[1] if len(parts) > 1 else stem


def build_index_panel(sections: list[tuple[str, str]]) -> str:
    """Build the HTML fragment listing all Technical Details pages on the index."""
    items = []
    in_group = False
    # Match any leading "§<digits and dots>" then optional ".", whitespace.
    leading_num = re.compile(r"^§[\d.]+\.?\s+")
    for stem, title in sections:
        num = _section_number_from_stem(stem)
        clean_title = leading_num.sub("", title)
        # Indent §1.4.* items so the multi-feature subsection stands out.
        is_sub = num.startswith("1.4.") and num != "1.4"
        if is_sub and not in_group:
            in_group = True
        if not is_sub and in_group:
            in_group = False
        css_class = "td-item td-sub" if is_sub else "td-item"
        items.append(
            f'  <li class="{css_class}">'
            f'<span class="td-num">§{num}</span> '
            f'<a href="{stem}.html">{clean_title}</a>'
            f'</li>'
        )
    list_html = "\n".join(items)
    return (
        f"{INDEX_PANEL_START}\n"
        f'<section class="technical-details">\n'
        f'  <h2>Technical Details</h2>\n'
        f'  <p>The technical chapter of the ISOSTER documentation, '
        f'organized as twelve linked pages covering the algorithmic '
        f'foundation, implementation, speed argument, six feature '
        f'subsections with inline demonstrations, and a comparison '
        f'with related tools. Math is rendered client-side with '
        f'MathJax 3.</p>\n'
        f'  <ol class="td-list">\n{list_html}\n  </ol>\n'
        f'</section>\n'
        f"{INDEX_PANEL_END}"
    )


def update_index_page(sections: list[tuple[str, str]]) -> None:
    """Replace (or insert) the Technical Details panel in index.html."""
    index_text = INDEX_PATH.read_text(encoding="utf-8")
    panel = build_index_panel(sections)

    # Migrate legacy DRAFT-PANEL markers to TECHNICAL-PANEL on first run.
    if _LEGACY_PANEL_START in index_text and _LEGACY_PANEL_END in index_text:
        new_text = re.sub(
            re.escape(_LEGACY_PANEL_START) + r".*?" + re.escape(_LEGACY_PANEL_END),
            panel,
            index_text,
            flags=re.DOTALL,
        )
    elif INDEX_PANEL_START in index_text and INDEX_PANEL_END in index_text:
        new_text = re.sub(
            re.escape(INDEX_PANEL_START) + r".*?" + re.escape(INDEX_PANEL_END),
            panel,
            index_text,
            flags=re.DOTALL,
        )
    else:
        # First time: inject the panel just after the hero <header>.
        hero_close = "</header>"
        if hero_close in index_text:
            new_text = index_text.replace(
                hero_close, hero_close + "\n\n" + panel + "\n", 1
            )
        else:
            new_text = index_text.replace(
                "</div>\n</body>", panel + "\n</div>\n</body>", 1
            )

    INDEX_PATH.write_text(new_text, encoding="utf-8")


# --- Main entry --------------------------------------------------------------


def main() -> None:
    md_files = sorted(DRAFT_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown found under {DRAFT_DIR}")

    sections: list[tuple[str, str]] = []
    for i, md_path in enumerate(md_files):
        prev_stem = md_files[i - 1].stem if i > 0 else None
        next_stem = md_files[i + 1].stem if i + 1 < len(md_files) else None
        out_path = HTML_DIR / f"{md_path.stem}.html"
        title = build_section_page(md_path, out_path, prev_stem, next_stem)
        sections.append((md_path.stem, title))
        print(f"  wrote {out_path.relative_to(THIS_DIR.parent.parent)}")

    update_index_page(sections)
    print(f"  updated {INDEX_PATH.relative_to(THIS_DIR.parent.parent)}")
    print(f"\nBuilt {len(sections)} section page(s).")


if __name__ == "__main__":
    main()
