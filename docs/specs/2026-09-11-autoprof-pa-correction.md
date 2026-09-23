# AutoProf publication PA correction

Approved 2026-09-11. Isoster branch: `fix/autoprof-publication-pa`.

Reuse `isoster_pa_to_autoprof_init` in the exhausted-campaign options builder.
Test actual options, wrapped angles and unchanged non-PA settings. Keep the
four existing AutoProf arms and the correct output parser; do not alter the
Stage 4 timing worker or mock generator.

Document the error and its lessons in Isoster's tracked publication reference
and agent notes, and in `isophote_test/docs/LESSON.md`, plan and journal.
The external repository change is documentation-only, on a feature branch,
then tested/reviewed, committed, merged and pushed as authorized. Isoster is
committed and pushed without merging.

Use separately named PA-correction gate, Huang2013 and S4G campaigns beneath
the existing publication root. Do not modify old images or results. Check
source hashes, saved PA options, native/standardized profiles and image QA.
Run reference-only Isoster dependencies first; require their resolved
fixed-center coordinates to match the archived references within 1e-8 pixel
before running AutoProf. Dependency outputs do not replace scientific
Isoster outcomes. Freeze source/environment/configuration and keep all logs.

Validate options in under one minute, then a small actual-fit gate with both
adapters, elongated/compact and noiseless/noisy inputs, exercising all four
AutoProf arms. Correct configuration and coherent geometry are the acceptance
criteria, not a superior score. Review Huang2013 before S4G. Requested counts:
837 x 4 = 3,348 Huang2013 and 1,800 x 4 = 7,200 S4G AutoProf records, plus
2,637 reference dependencies. Use at most eight galaxy workers, single-thread
numerical libraries, existing 300-second AutoProf timeouts. Preserve numerical
failures; stop on configuration/data-integrity problems or systematic failures.

Audit every logical outcome and select corrected AutoProf results consistently,
including failures. Never select between old and new by quality. Update the
accepted analysis manifest and revalidate the Huang2013 analysis before full
scientific measurements. Keep native harmonics separate and production timing
descriptive. AutoProf is stochastic; a corrected rerun is not a controlled
single-variable deterministic experiment.
