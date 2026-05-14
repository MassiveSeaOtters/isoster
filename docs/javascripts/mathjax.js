// MathJax configuration for mkdocs-material + pymdownx.arithmatex generic mode.
// pymdownx.arithmatex (generic: true) wraps inline math in \(...\) and block
// math in \[...\] inside <span class="arithmatex"> elements, so we tell
// MathJax to use those delimiters and only process matching containers.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset math when mkdocs-material navigates between pages (it uses
// instant navigation, which does not trigger a full page reload).
document$.subscribe(() => {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
});

// Mermaid is initialized once at load.  pymdownx.superfences emits the
// Mermaid diagrams as <div class="mermaid"> blocks, which mermaid.run()
// picks up automatically.
document$.subscribe(() => {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      flowchart: { curve: "basis", htmlLabels: true, padding: 12 }
    });
    mermaid.run({ querySelector: ".mermaid" });
  }
});
