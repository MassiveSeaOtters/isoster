# Introduction literature review

Date: 2026-08-11  
Purpose: source map and claim audit for `draft/introduction.md`

## Narrative selected for the introduction

1. Isophotal profiles are scientifically useful measurements of galaxy
   structure, not just a plotting convenience.
2. Non-parametric isophotes and parametric two-dimensional decompositions are
   complementary.
3. ISOSTER belongs to the Carter--Jedrzejewski--Ciambur algorithmic lineage.
4. Deep and multi-band imaging make speed, uncertainty handling, outer-profile
   stability, and information sharing across bands increasingly important.
5. `photutils.isophote` and AutoProf establish the modern software context.
6. ISOSTER's contribution is the integrated combination of a fast compatible
   core, controlled extensions, and reproducible diagnostics. The draft does
   not claim that each individual ingredient is unprecedented.

## Search record

NASA ADS searches used the `/v1/search/query` endpoint with explicit fields
`bibcode,title,author,year,doi,identifier,citation_count,abstract`. Searches
were run on 2026-08-11.

- Foundations:
  `property:refereed AND (title:isophot* OR abs:"isophote fitting" OR
  abs:"surface photometry") AND (abs:galaxy OR abs:galaxies) AND
  year:[1975 TO 2005]`
- Modern tools:
  `property:refereed AND (title:(AutoProf OR Isofit OR photutils) OR
  abs:("isophote fitting" AND (pipeline OR software))) AND
  year:[2005 TO 2026]`
- Isophote science:
  `property:refereed AND (title:isophot* OR abs:"isophote shape") AND
  abs:(boxy OR disky OR peanut OR bar OR morphology) AND
  year:[1975 TO 2026]`
- Parametric and multi-band tools:
  `property:refereed AND (title:(GALFIT OR GALFITM OR IMFIT OR PROFIT OR
  "multi-band galaxy fitting") OR abs:("multi-band" AND "galaxy fitting"))
  AND year:[2000 TO 2026]`
- Recent isophote methods:
  `identifier:(arXiv:2606.20370 OR arXiv:2407.12983) OR
  (title:isophot* AND year:[2022 TO 2026] AND property:refereed)`

The arXiv public API was queried once, respecting its single-connection and
three-second rules:

`all:"isophote fitting" AND cat:astro-ph.GA`, sorted by submission date,
50 results maximum. It returned nine records. AutoProf was the only general
isophote-fitting pipeline in this deliberately narrow query. Recent results
included specialist applications and methods for bars, lens multipoles,
lopsidedness, and dwarf-galaxy isophote statistics; this is evidence of the
method's continuing scientific use, not proof that no other general package
exists.

## Claim-level source map

| Draft claim | Primary support | What the source establishes | Use boundary |
|---|---|---|---|
| Fourier deviations from ellipses encode boxy/disky structure and correlate with galaxy properties | Carter (1978); Bender et al. (1988, 1989) | Establishes the Fourier/isophote-shape framework and observed correlations | Do not imply a unique causal interpretation for every coefficient |
| Radial isophotal profiles reveal embedded components, bars, lopsidedness, and structural diversity | Ferrarese et al. (2006); Kormendy et al. (2009); Li et al. (2011) | Large observational studies using surface-brightness and geometric profiles | Examples span different samples and should not be presented as one homogeneous result |
| The standard iterative fitting lineage is Jedrzejewski (1987) | Jedrzejewski (1987) | Ellipse fitting plus harmonic deviations in CCD surface photometry | The paper is an observational study containing the algorithm, not a standalone software paper |
| Eccentric anomaly improves highly flattened/non-elliptical harmonic descriptions | Ciambur (2015) | Introduces the formalism and `Isofit`/`Cmodel`; demonstrates edge-on and peanut/X structures | Avoid saying polar-angle fitting is universally inappropriate |
| Parametric fitting is powerful but model-conditional | Peng et al. (2010); Erwin (2015); Robotham et al. (2017) | Documents flexible multi-component 2-D fitting and its statistical choices | The complementarity argument is a synthesis, not a criticism of parametric fitting |
| Joint use of multi-band data stabilizes structural inference | Häußler et al. (2013) | GALFITM/GALAPAGOS-2 improves parametric measurements, especially in low-S/N bands | This is parametric prior art; it does not establish ISOSTER's non-parametric algorithm |
| Modern data reach extremely faint diffuse structure | Abraham & van Dokkum (2014); Duc et al. (2015); Trujillo & Fliri (2016) | Demonstrates ultra-low-surface-brightness instrumentation and science at/below 30 mag arcsec^-2 | Depth numbers depend on band, aperture, significance convention, and reduction |
| AutoProf is an automated modern comparator and reports >2 mag arcsec^-2 gain over comparison methods | Stone et al. (2021) | Directly stated and tested in the AutoProf paper | Attribute the number to its images and comparison setup; do not generalize universally |
| Isophote measurements remain relevant for upcoming large samples | Watkins et al. (2026) | Uses isophotal quantities for dwarf galaxies and argues for high-dimensional large-sample analysis | Useful as a current application/caveat, but not necessary in the first draft |

## Core references and links

- Carter (1978), *The structure of the isophotes of elliptical galaxies*:
  [ADS](https://ui.adsabs.harvard.edu/abs/1978MNRAS.182..797C),
  [DOI](https://doi.org/10.1093/mnras/182.4.797)
- Jedrzejewski (1987), *CCD surface photometry of elliptical galaxies I*:
  [ADS](https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J),
  [DOI](https://doi.org/10.1093/mnras/226.4.747)
- Bender et al. (1988, 1989), isophote-shape data and correlations:
  [Paper I](https://ui.adsabs.harvard.edu/abs/1988A&AS...74..385B),
  [Paper II](https://ui.adsabs.harvard.edu/abs/1989A&A...217...35B)
- Ferrarese et al. (2006), ACS Virgo isophotal analysis:
  [ADS](https://ui.adsabs.harvard.edu/abs/2006ApJS..164..334F)
- Kormendy et al. (2009), elliptical and spheroidal structure:
  [ADS](https://ui.adsabs.harvard.edu/abs/2009ApJS..182..216K)
- Li et al. (2011), Carnegie-Irvine isophotal analysis:
  [ADS](https://ui.adsabs.harvard.edu/abs/2011ApJS..197...22L)
- Peng et al. (2010), GALFIT 3:
  [ADS](https://ui.adsabs.harvard.edu/abs/2010AJ....139.2097P)
- Häußler et al. (2013), MegaMorph/GALFITM:
  [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430..330H)
- Abraham & van Dokkum (2014), Dragonfly:
  [ADS](https://ui.adsabs.harvard.edu/abs/2014PASP..126...55A)
- Erwin (2015), IMFIT:
  [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...799..226E)
- Ciambur (2015), `Isofit`/`Cmodel`:
  [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...810..120C),
  [local source text](../reference/isofit_ciambur_2015.txt)
- Duc et al. (2015), deep imaging of early-type galaxies:
  [ADS](https://ui.adsabs.harvard.edu/abs/2015MNRAS.446..120D)
- Trujillo & Fliri (2016), imaging beyond 31 mag arcsec^-2:
  [ADS](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..123T)
- Robotham et al. (2017), ProFit:
  [ADS](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.1513R)
- Stone et al. (2021), AutoProf:
  [ADS](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1870S),
  [local source text](../reference/autoprof_stone_2021.txt)
- Bradley et al. (2025), Photutils 2.3.0:
  [software DOI](https://doi.org/10.5281/zenodo.17129028),
  [official isophote guide](https://photutils.readthedocs.io/en/2.3.0/user_guide/isophote.html)
- Watkins et al. (2026), dwarf-galaxy isophote analysis:
  [ADS](https://ui.adsabs.harvard.edu/abs/2026MNRAS.547ag472W)

## Wording that should remain conservative

- Say ISOSTER is a "faithful descendant" or "compatible implementation" of
  the Jedrzejewski family only where matched-configuration tests support the
  claim. Do not say outputs are byte-identical to `photutils.isophote`.
- Present non-parametric and parametric methods as complementary.
- Attribute every numerical low-surface-brightness depth to its measurement
  convention and dataset.
- Describe the novelty as an integrated combination unless the final paper
  includes a systematic feature audit establishing that a specific component
  has no prior implementation.
- Do not repeat the 10--15x headline in the introduction until the final
  benchmark table fixes the hardware, versions, image sizes, and configuration.
- Do not call IRAF unmaintained. That claim was not needed for the introduction
  and is sensitive to the distinction between the historical NOAO project and
  community-maintained IRAF distributions.

## Author decisions still needed

1. Target journal (ApJ, MNRAS, A&A, or another venue), which determines
   citation commands, section length, and software-paper conventions.
2. Whether the introduction should foreground low-surface-brightness science
   or survey throughput; the current draft gives them equal weight.
3. Whether to cite the 2026 Watkins et al. dwarf-galaxy paper as a current
   scientific application or keep the introduction focused on method lineage.
4. Whether the final abstract/introduction may quote a numerical speedup; this
   should wait for the frozen publication benchmark table.
5. Final paper section order, so the last paragraph can name section numbers.
