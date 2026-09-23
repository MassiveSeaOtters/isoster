# AGENTS.md

This file gives coding-agent instructions for working in this repository.
`CLAUDE.md` contains the longer project guide; follow both files.

## Communication Style

- Use clear, plain English for progress updates, plans, and final summaries.
- Assume the reader has an astrophysics background, not a professional software-engineering or project-management background.
- Avoid software/project jargon when a plain phrase works. For example:
  - Say "run a realistic test that writes output" instead of "write-mode smoke refresh".
  - Say "copy one galaxy's campaign folder and test there so the original data is not changed" instead of "on a copied single-galaxy campaign tree".
  - Say "find/list the matching galaxy folders" instead of "enumerate galaxies".
- If a technical term is needed, define it the first time in the same sentence.
- Prefer concrete file paths, commands, and expected outputs over abstract process words.
- Keep the tone direct and professional, but optimize for effortless understanding.

## Publication Validation Data Safety

- Never delete, replace, or modify existing test data or results under `/Volumes/galaxy`.
- Put every newly generated or downloaded image and every new result in a separate, explicitly named campaign folder.

## Publication Analysis Figures

- Cross-tool QA shows full finite residual maps without evaluation-cut masking; cuts still define numerical metrics. Tables report Truth RMS and signed Flux Bias in ALL/INNER/MIDDLE/OUTER order. Surface-brightness limits use data alone, with no I=0 reference line.

- Preserve the shared-renderer harmonic-off baseline when evaluating harmonic-enabled models. Record harmonic basis, supported orders, renderer and common pixel support explicitly; an AutoProf native ellipse model is not harmonic-enabled merely because extraction measured a3/b3/a4/b4.

- Publication benchmark analysis and QA use a fixed 2-pixel elliptical inner radial cut, not a PSF-FWHM cut (decision 2026-09-23). Apply this to Huang2013, S4G, and future tests; preserve historical outputs and record the actual cut in new measurements. This changes evaluation, not fitting radii or the physical PSF.

- Study and reuse the existing plotting code, formats, and style before creating new figures. Align new plotting scripts with the existing QA conventions.
- Do not display a scientific measurement aperture as a bad-pixel mask on ideal mock images. Use labelled inner/outer ellipses instead. QA redesigns remain opt-in demos until the user approves them as defaults.
- The individual QA layout reviewed on 2026-09-22 was approved on 2026-09-23; reuse it for new benchmark QA. Keep captions separate and preserve historical figures.

## Prepared-background mock comparisons

- For the new Huang2013 and subsequent S4G AutoProf publication campaigns,
  fix the background level to zero; background subtraction is input preparation,
  not part of the intended 1-D algorithm comparison. Treat noise estimation
  separately and preserve earlier campaigns as superseded provenance.
