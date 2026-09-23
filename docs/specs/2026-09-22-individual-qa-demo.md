# Individual QA demonstration

Accepted by the user on 2026-09-23. The layout is now used for scientific
atlas and individual QA exports; the module keeps its historical name.
New measurements/QA use the fixed 2-pixel radial cut documented in
`2026-09-23-two-pixel-evaluation.md`. Earlier demo products remain historical.

Approved scope: two Huang2013 inputs, IC1459/wide_z050 and
NGC4742/wide_z050 (AutoProf primary failure). No refitting, source-data writes,
bulk export, or default adoption. Use the corrected zero-background science
selection, including all Isoster afterburner arms.

Cross-tool: reuse the existing comparison plot; unmasked data with labelled
inner/outer measurement ellipses; statistics table; common-aperture residual
maps with identical color limits. Retain failed primary arms in the table,
without replacement. Runtime is parallel-campaign fit time, not controlled timing.

Cross-arm: six aligned radial panels and a table of status, fit time,
truth-relative RMS, signed flux bias and radial reach. Use the designated
default as the difference reference, never a score winner. Include skipped
and failed arms. Preserve native uncertainty/stop-state semantics.

Validate each successful arm's rebuilt common-aperture metrics against the
saved scientific table; hash inputs before/after; render PNG/PDF and inspect.
Only a new output directory is writable. No architectural/default changes.

## Review revision: metric matrix and residual thumbnails

Replace the cross-arm table with four independently scaled metric columns
(fit time, truth RMS, signed flux bias, radial reach); arms form the rows.
Print actual values in black. Grey cells explicitly identify failed, skipped
or missing outcomes; preserve flags in captions. Add a residual thumbnail
aligned with each row, sharing pixel coordinates, science aperture and one
residual color scale across all successful arms of the same input.
Cross-tool table edges align with the combined visible panel bounds.
Move explanatory residual footnotes to separate caption files. Still demo-only.

## Final layout review

Use compact Inner/Outer labels without values. Move runtime/aperture notes
to captions and reduce the table-to-panel gap. Cross-arm row labels contain
only arm names; captions retain all statuses/flags. Align matrix top/bottom
with the touching, shared-x radial stack; bring metric colorbars and residual
thumbnails closer and omit the residual colorbar. Add an arm legend and
remove the cross-arm I=0 line. Red indicates lower time/RMS/absolute bias
or greater reach; blue indicates the opposite. Printed bias remains signed.
Constant metric columns are neutral; coverage is not a quality verdict.
