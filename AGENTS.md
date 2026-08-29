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
