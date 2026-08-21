<!-- Excised from the public docs at docs/technical/1.4.4-batch-robustness.md .
     Manuscript-only: this experiment has not been performed. -->

## §1.4.4.8 Proposed demonstration: a panel of failure-mode SGA-2020 galaxies

> **Status: proposed, not performed.** The panel below has not been
> selected, the galaxies have no identifiers yet, and the fits have
> not been run. Every statement about what the figure shows is a
> specification of the intended experiment, not a result. It is
> retained as a specification because the recovery-rate claim this
> section rests on needs exactly this evidence and does not yet
> have it.

To make the robustness gain concrete on real data we would use a
small
panel of galaxies drawn from the SGA-2020 catalogue, curated in
`sga_isoster/data/demo/` (an external companion repository, not
part of the ISOSTER source tree). The panel is selected by an
explicit
criterion: each galaxy is one where a defined *baseline* fit fails —
either by returning only the central pixel (anchor failure) or by
terminating after the first few isophotes with cascading stop
codes — but where visual inspection of the image confirms that the
galaxy is well-resolved with a clean photometric structure.

The baseline needs stating explicitly, because "the ISOSTER default"
is not the same thing as "the §1.4.4 primitives disabled". The
shipped default already carries three first-isophote retries, and
the gradient-SNR damping and step clipping of §1.4.4.4–§1.4.4.5 have
no off switch at all. A comparison against the shipped default would
therefore measure only the *increment* from the opt-in primitives —
a legitimate experiment, but not the one the surrounding text
describes. The ablation baseline is instead
`max_retry_first_isophote=0`, `permissive_geometry=False`,
`integrator='mean'`, `sigma_bg=None`, with the two always-on
safeguards left in place because they cannot be removed by
configuration. That baseline is what "the classical recovery policy"
means here, and it should be named in the figure caption rather than
left to the reader to infer. The panel would
therefore *not* be a random sample; it would be a cherry-picked set
of failures selected because each has a specific cause that one of
the §1.4.4 primitives is designed to address. Whether the primitive
in fact addresses it is what the figure asks.

**[FIG X.4.4 — to be generated. One row per panel galaxy. For each
galaxy: left subpanel shows the LegacySurvey $r$-band image with the
*ablation-baseline* fit's converged isophotes overlaid in dim red —
for a galaxy selected on anchor failure, typically just the central
pixel or a few early isophotes; right subpanel shows the same image
with the *single-primitive* arm's isophotes overlaid in solid blue,
with whatever SMA range it in fact reaches. The baseline is the
configuration defined above, not the shipped default, which already
carries three retries. Galaxies would be chosen so
that each targets one primitive: an anchor failure addressed by
`max_retry_first_isophote`; an inter-isophote cascade addressed by
`permissive_geometry`; an LSB-tail sigma-clipping anomaly addressed
by `integrator='adaptive'` plus `lsb_sma_threshold`; a
marginal-gradient regime addressed by the `sigma_bg` floor. Data source: `sga_isoster/data/demo/`
(external companion repository);
specific galaxy IDs to be selected at figure-commission time from
the SGA-2020 entries flagged in the curated demo subset.]**

The argument the figure is designed to make is a series of pairwise
comparisons rather than a single dramatic improvement. Each row would
show the *same galaxy*, fit by the *same algorithm* with the *same
data*, differing only in the §1.4.4 configuration: the ablation
baseline documenting what went wrong, and one arm per primitive
showing what each changes on its own.

The single-primitive arms are what licenses an attribution, and an
earlier version of this section lacked them. Turning on retries,
permissive propagation, adaptive integration and the `sigma_bg` floor
together and observing a recovery establishes only that *something*
in that set helped. Because the galaxies would be selected one per
targeted primitive, the natural failure of a single combined arm is
that a recovery gets credited to the primitive the galaxy was chosen
for, whether or not that is what did the work. A leave-one-out
arrangement answers the same question and is equally acceptable; a
single combined arm is not. Whether each failure is resolved at all
is the question the experiment asks, not a premise of it: an arm
where the targeted primitive does *not* recover the galaxy — or where
a different one does — is a legitimate and informative outcome that
belongs in the figure.

The reproducibility recipe is:

```python
from isoster import fit_image
from isoster.config import IsosterConfig

# Sketch, not runnable code: `load_galaxy` and
# `sma_at_surface_brightness` stand for whatever the caller's data
# access and profile inspection provide. Everything referring to
# ISOSTER itself is a real API.
#
# Ablation baseline: opt-in primitives off. The gradient-SNR damping
# and step clipping cannot be disabled and are present in every arm.
BASELINE = dict(
    sma0=10.0, maxsma=200.0,
    max_retry_first_isophote=0,
    permissive_geometry=False,
    integrator="mean",
    sigma_bg=None,
)

# One arm per primitive, each turning on exactly one thing relative to
# the baseline. Enabling all four at once would recover some galaxies
# but could not say which primitive did it.
def arms_for(image, sky_rms):
    """Baseline plus one single-primitive arm per §1.4.4 feature.

    ``sky_rms`` and the integrator transition are per-image quantities:
    the sky noise is measured from this galaxy's own blank-sky pixels,
    and the transition SMA is placed where this profile enters the
    noise-dominated regime (§1.4.4.3). Neither is a constant.
    """
    transition = sma_at_surface_brightness(image, level=3.0 * sky_rms)
    return {
        "baseline":   dict(BASELINE),
        "retry":      dict(BASELINE, max_retry_first_isophote=5),
        "permissive": dict(BASELINE, permissive_geometry=True),
        "adaptive":   dict(BASELINE, integrator="adaptive",
                           lsb_sma_threshold=transition),
        "sigma_bg":   dict(BASELINE, sigma_bg=sky_rms),
        "all":        dict(BASELINE, max_retry_first_isophote=5,
                           permissive_geometry=True, integrator="adaptive",
                           lsb_sma_threshold=transition, sigma_bg=sky_rms),
    }

# Run every arm on every panel galaxy. The "all" arm shows what a
# survey pipeline would actually get; the single-primitive arms are
# what license an attribution.
for galaxy_id in panel:
    image, mask, sky_rms = load_galaxy(galaxy_id)
    results = {
        name: fit_image(image, mask=mask, config=IsosterConfig(**kwargs))
        for name, kwargs in arms_for(image, sky_rms).items()
    }
    # results["retry"]["first_isophote_failure"] and
    # results["retry"]["first_isophote_retry_log"] document the recovery
    # trail for whichever galaxy was selected on anchor failure.
```

A closing observation. The classical algorithm's failure modes on
the panel galaxies are *not* algorithmic failures in the strict
sense; the algorithm is correctly identifying that the fit
attempt under the supplied starting conditions is unrecoverable.
The §1.4.4 primitives do not change the algorithm's correctness
criterion; they change the *recovery policy* when a single fit
attempt fails. The right way to read the figure is therefore as a
a measurement of how far ISOSTER's recovery policy gets on the
recoverable-failure subset of a real survey sample — *not* as a
demonstration that ISOSTER's
algorithm is fundamentally more capable than the classical one on
the same galaxy. The algorithmic capability is unchanged; the
operational robustness around it is what the §1.4.4 primitives
contribute, and that is what survey pipelines need.
