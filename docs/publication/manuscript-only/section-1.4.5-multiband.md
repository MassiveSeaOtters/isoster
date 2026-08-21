<!-- Excised from the public docs at docs/technical/1.4.5-multiband.md .
     Manuscript-only: this experiment has not been performed. -->

## §1.4.5.9 Proposed demonstration: band-dependent morphology

> **Status: proposed, not performed.** Nothing in this subsection
> has been run, and no numbers below are measurements. The mock it
> describes does not exist in the repository, and the claims about
> what the three workflows would produce are predictions from the
> argument of §1.4.5.1, not results. The demonstration that *has*
> been performed is §1.4.5.10, whose galaxy is deliberately
> different: it plants one common geometry in every band and so
> tests precision, not bias. The experiment below is the one that
> would test bias, and it remains to be built. It is retained here
> as a specification rather than deleted because the gap it fills is
> the one an attentive referee will ask about.

To make the joint-vs-forced difference visible on a controlled
input we would use a three-band synthetic galaxy generated through
the MockGal renderer. The galaxy is a two-component model: a Sersic
$n_\mathrm{Sersic} = 4.0$ bulge with effective semi-major axis
$R_e^\mathrm{bulge} = 25\,\mathrm{px}$, ellipticity
$\varepsilon^\mathrm{bulge} = 0.20$, and a position angle
$\mathrm{PA}^\mathrm{bulge} = 30^\circ$; plus a Sersic
$n_\mathrm{Sersic} = 1.0$ exponential disk with effective semi-major
axis $R_e^\mathrm{disk} = 70\,\mathrm{px}$, ellipticity
$\varepsilon^\mathrm{disk} = 0.55$, and position angle
$\mathrm{PA}^\mathrm{disk} = 0^\circ$ (i.e. mis-aligned with the
bulge by $30^\circ$). The two components have different bulge-to-
total ratios in the three bands by construction: $B/T = 0.7$ in
the redder band, $B/T = 0.5$ in the central band, $B/T = 0.3$ in
the bluer band. The synthetic color gradient between the bands
arises from the spatially-varying bulge-to-disk weighting; this
is the canonical color-gradient morphology and the canonical
forced-photometry failure mode.

The image grid is $512 \times 512\,\mathrm{px}$ per band, with
matched PSF (FWHM $= 2\,\mathrm{px}$) and matched sky noise
($\mu_\mathrm{sky} = 26.5\,\mathrm{mag\,arcsec^{-2}}$). The bands
are constructed so the total flux is comparable across bands.

**"Ground truth" needs an operational definition here, and this is
the part that makes the experiment harder than it looks.** A
two-component model with misaligned components has an analytic 2-D
intensity field, but it does *not* have an analytic sequence of
elliptical isophotes: its true isophotes are not ellipses at all, so
there is no closed-form "true $I_b(a)$" to compare against. The
comparison therefore has to be made against an explicitly defined
reference, and that reference has to estimate *the same quantity
ISOSTER reports*. ISOSTER's `intens` is not an enclosed-aperture
flux: it is a ring statistic — the intercept of the harmonic fit
along one ellipse, which for the default mean integrator on a
complete ring equals the mean intensity along that ellipse. A
reference defined by area integration inside an aperture would
therefore be a different quantity, and the comparison would confound
the workflows' geometry choices with a curve-of-growth-versus-ring
mismatch.

Three things must be pinned down, and pinning them down is most of
the work of building this mock:

1. **Which ellipse.** For each band and each semi-major axis, the
   reference ellipse is obtained by least-squares fitting an ellipse
   to the analytic isophotal contour of that band's noise-free
   intensity field, by numerical root-finding on the model rather
   than by fitting the rendered image. This is where each band's
   "own" geometry is defined, and it must not come from any fitter
   under test.
2. **Which angular basis and weighting.** The reference intensity is
   the *normalized angular mean* of the analytic field around that
   ellipse — $\frac{1}{2\pi}\int_0^{2\pi} I(\theta)\,\mathrm{d}\theta$,
   the quantity the harmonic intercept estimates — taken with uniform
   weight in the same angular coordinate the fit under test used:
   $\phi$ for a default fit, $\psi$ under eccentric-anomaly sampling.
   Note this is an average, not an unnormalized path integral: an
   arc-length-weighted line integral would be a third quantity again,
   and would not match the intercept. The $\phi$ and $\psi$ averages
   differ on a flattened ellipse for exactly the reasons §1.4.1.1
   gives, so the choice cannot be left implicit.
3. **Ring intensity, not enclosed flux.** The comparison is against
   $I_b(a)$ as a ring statistic. If the curve of growth is wanted as
   well, it is a separate reference requiring its own definition,
   and it should be reported as a separate panel rather than folded
   into the same residual.

That reference is noise-free, fitter-independent, and estimates what
ISOSTER estimates. Skipping this construction and calling the
renderer's radial profile "the truth" would compare the fitters
against a quantity none of them is computing.

**[FIG X.4.5 — to be generated. Three-panel figure. Panel A:
false-color $g$-$r$-$i$ composite of the input mock with the
joint-fit isophotes overlaid in white. Panel B: per-band intensity
profile $I_b(a)$ for the three bands, showing four lines per band —
each band's own reference profile as constructed above (solid black),
the joint free-fit profile (solid blue), the forced-photometry
profile using the central-band geometry (dashed red), and the
per-band independent free-fit profile (dotted green). The reference
is the ring-intensity construction defined in this subsection, *not*
the renderer's radial profile; the two are different quantities and
using the latter would invalidate the comparison. Panel C: the residual
$\Delta I / I$ of each of the three workflows against each band's
own truth, plotted against SMA.

The questions the panels are to answer, stated as questions because
the answers are not known: how large is each workflow's departure
from a given band's own truth, and how does it vary with radius as
the two components' relative weight changes; does the joint fit's
compromise sit between the bands in proportion to their geometric
leverage, as the weighting of §1.4.5.2 predicts; and is the joint
departure smaller in magnitude than the forced one, which adopts a
single band's geometry outright. Note that the joint fit is *not*
expected to reproduce any individual band's truth on this mock —
§1.4.5.1 explains why no shared-geometry method can — so a panel
showing it doing so would indicate an error in the mock, not a
success. Data source: to be generated;
bulge-plus-disk-with-color-gradient mock, recipe below.]**

The argument the experiment is designed to test is narrow. All
three workflows would be applied to the *same* synthetic image, so
that the only difference is which geometry is used for each band's
intensity extraction.

What it can and cannot establish deserves stating in advance,
because the obvious summary — "joint is unbiased, forced is biased"
— is not available from a mock in which the bands genuinely have
different shapes. On such a mock there is no single aperture
sequence that is correct for every band, so *every* shared-geometry
method returns a compromise, the joint fit included. What the
experiment can show is the *size and direction* of each method's
departure from each band's own truth: whether the joint compromise
sits between the bands in proportion to their information content,
as the weighting of §1.4.5.2 predicts, and whether it is smaller
than the departure forced photometry incurs by adopting one band's
geometry wholesale. Those are the honest questions, and they are
worth answering. "The joint fit recovers the truth" is not one of
them, because on this mock the three workflows are not estimating
the same quantity — see §1.4.5.1.

The reproducibility recipe is:

```python
from isoster.multiband import fit_image_multiband
from isoster.multiband.config_mb import IsosterConfigMB

# All three bands share the same mask (no loose validity needed)
images        = [image_g, image_r, image_i]
variance_maps = [var_g,   var_r,   var_i]

config = IsosterConfigMB(
    sma0=10.0,
    maxsma=180.0,
    bands=["g", "r", "i"],
    reference_band="r",  # required field
    multiband_higher_harmonics="independent",  # per-band a_n / b_n
    harmonic_orders=[3, 4],
)

results_joint = fit_image_multiband(
    images,
    masks=mask,
    config=config,
    variance_maps=variance_maps,
)
```

The output dictionary contains the per-SMA shared geometry
$(x_0, y_0, \varepsilon, \mathrm{PA})$, the per-band intensities
$I_{g}$, $I_{r}$, $I_{i}$ (and their uncertainties), and the
per-band higher-harmonic coefficients for each band and each
order. Under `'independent'` (the mode used here) the stored
coefficients are already Bender-normalized per band; under the
cross-band modes (`'shared'`, `'simultaneous_in_loop'`,
`'simultaneous_original'`) the stored `a<n>_<b>` / `b<n>_<b>`
columns carry *raw* coefficients and the per-band Bender
normalization of §1.4.5.5 is applied at plotting time — band-distinct
raw values that normalize to one shared shape under `'shared'`,
identical raw values under the two `'simultaneous_*'` modes. The
`harmonics_shared` flag in the result dictionary tells a consumer
which convention the stored values follow.
The corresponding forced-photometry call would be a
sequence of single-band `fit_image` calls with the central-band
result passed as `template` to the other two; the per-band
independent-fit call would be three independent `fit_image` calls
with the same `IsosterConfig` for all three.

A closing observation. The joint multi-band free fit is *not* a
replacement for forced photometry in all cases. Forced photometry
remains the right choice when the user *explicitly wants* to
report band-to-band intensities at a fixed geometry — for example,
when comparing aperture photometry across bands using a
predetermined elliptical aperture from a reference catalogue. The
joint free fit is the right choice when the user wants every band
measured on one aperture sequence that the data as a whole
determined — for example, when measuring colour gradients, age maps,
or stellar-population gradients, all of which are differences
between bands and so depend on the apertures matching. It does not
deliver each band's *natural* isophotal structure, and could not:
one shared geometry cannot simultaneously be every band's own. The
workflow that does deliver that is independent per-band fitting, at
the cost of the geometry mismatch §1.4.5.10 quantifies. ISOSTER
provides all three because they answer different questions. The
choice between them is a science decision, not a technical one.

