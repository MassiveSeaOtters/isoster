# AutoProf fixed-zero-background afterburner

Approved by the user on 2026-09-21: rerun Huang2013 AutoProf alone with
`ap_set_background=0.0`, replace its accepted scientific selection, then
repeat the full analysis. Apply the same policy to S4G in a later run.

Preserve all previous campaigns; replacement means selection supersession,
not deleting files. Use a GitHub-cloned feature branch outside Dropbox
because its active Git index has been replaced by an older synchronized copy.

Reuse the PA correction runner, four-arm roster, images, environments and
eight-process maximum. Add a fixed-background mode that copies accepted
Isoster reference profiles for the fixed-center dependency without fitting
Isoster again. Record dependency sources and hashes. Read previous AutoProf
records from the audited PA-corrected campaign, not the original PA-bugged run.
Keep the corrected PA conversion and all other arm options unchanged.

Forward the explicit finite background value in the shared AutoProf wrapper;
absence of that setting preserves the existing behavior. Do not change noise
estimation or introduce a noise floor without inspecting the small test and
obtaining direction if a material change is needed.

Validate a sub-minute small run including noiseless and noisy conditions,
then check a compact high-redshift image if not already covered. Audit saved
options, reported zero backgrounds, fixed-center equality, finite profiles
and unchanged source hashes. Run 3,348 Huang2013 AutoProf outcomes only after
validation passes. Keep failures, without best-of selection.

After the full audit, rerun the Huang collector and figure exporter in new
analysis directories, accepting this complete AutoProf roster. Compare old
and new metrics, explicitly noting changed common support. Update scientific
reference and manuscript prose. Keep S4G prepared but not executed in this step.

The user-requested preprocessed-background comparison is distinct from a
complete automatic sky-estimation pipeline benchmark; retain that distinction
in the publication record and do not rewrite the separate Stage 4 timings.
