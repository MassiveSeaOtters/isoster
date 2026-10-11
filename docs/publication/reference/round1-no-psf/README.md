# Round-1 reports (mock images without PSF convolution)

The mock images used by every campaign and analysis before 2026-10-10 were generated
without PSF convolution, because the mock generator silently skipped the convolution
step (found and fixed in the
[Isoster vs AutoProf gap review](../2026-10-09-isoster-autoprof-gap-review.md)).
Those images have cusp-like centres that no real image has, so every accuracy number,
win fraction and ranking measured on them is superseded by the round-2 campaigns on
PSF-convolved images:
[small sample](../2026-10-10-round2-small-sample-campaign.md) and
[full Huang2013 sample](../2026-10-10-round2-huang2013-full-campaign.md).

The reports are kept here, unchanged apart from link paths, because parts of them still
hold. Do not quote their numbers in the manuscript.

| Report | Status |
|---|---|
| `2026-08-29-single-band-round1-log.md` | Record of what round 1 ran and where. History only. |
| `2026-08-29-single-band-concurrency.md` | Orchestrator concurrency test. Not an accuracy result; still valid. |
| `2026-09-10-huang2013-analysis-gate.md` | Analysis gate for round 1. Superseded. |
| `2026-09-11-autoprof-pa-correction-log.md` | AutoProf position-angle convention repair. The correction carries over and is in the round-2 AutoProf adapter. |
| `2026-09-12-huang2013-scientific-analysis.md` | Round-1 scientific analysis. Numbers superseded. |
| `2026-09-21-autoprof-zero-background.md` | Zero-background preparation for AutoProf. The decision carries over: round 2 uses the zero-background arm roster. |
| `2026-09-23-two-pixel-analysis.md` | Defines the 2-pixel inner cut and the common evaluation support. The definitions carry over unchanged to round 2; the numbers do not. |
| `2026-09-23-autoprof-center-extent.md` | AutoProf centre and extent behaviour. Qualitative notes carry over; numbers superseded. |
| `2026-09-23-harmonic-demonstration.md` | Bounded harmonic reconstruction demonstration. A method demonstration measured on round-1 images; repeat on round-2 images before quoting. |
| `2026-09-23-huang-final-analysis.md` | Round-1 final reconstruction analysis over 837 inputs. Superseded by the round-2 full campaign. |
| `2026-09-30-fixed-center-cases.md` | Fixed-centre investigation on selected cases. Qualitative conclusion carries over; numbers superseded. |

What still stands from round 1 without qualification: the timing benchmark
(`../../three-way-timing-benchmark-reference.md` and `docs/technical/`), which does not
depend on the PSF.

The round-1 campaign data remain on the external disk under
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/`. Whether to keep them
is a separate decision.
