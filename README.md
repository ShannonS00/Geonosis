# Imaging Geonosis with the ELT

*A long time ago, in a galaxy far, far away… we pointed the Extremely Large
Telescope at it.*

This is a just-for-fun project, but the physics is done carefully. We build
the Star Wars Geonosis system (the droid foundry, the arena and the ring of
boulders Obi-Wan nearly died in) and then run two simulated ELT imaging
campaigns. They use [ScopeSim](https://scopesim.readthedocs.io) and the
official IRDB instrument packages (Armazones, ELT, METIS, MICADO).

1. **The system:** the star Ea, the ringed desert planet Geonosis, and its 15 moons.
2. **Campaign A: "Help me, METIS, you're my only hope."** ELT/METIS observes
   Geonosis from Earth. Ea is placed at 1.3 pc, the distance of Alpha Centauri.
3. **Campaign B: "This is where the fun begins."** The Geonosians build their
   own ELT with MICADO and turn it on their moons.

Everything runs locally. The only downloads are the Python packages and the
IRDB instrument packages. Astropy IERS auto-downloads are switched off, so
no data is sent anywhere.

## Quick start (punch it, Chewie)

```bash
uv sync                                         # Python 3.12+/3.13 environment
uv run python -c "import scopesim; scopesim.rc.__config__['!SIM.file.local_packages_path']='inst_pkgs'; scopesim.download_packages(['Armazones','ELT','METIS','MICADO'])"
uv run pytest -q tests                          # sanity checks
uv run python campaigns/01_build_system.py      # ~20 s
uv run python campaigns/02_earth_metis.py       # full run ~30-40 min (--quick: ~10 min)
uv run python campaigns/03_geonosis_micado.py   # full run ~30 min (--quick: ~8 min)
uv run python campaigns/05_easter_eggs.py       # ~10 min, after 03
uv run python campaigns/04_report.py            # local HTML report: outputs/report.html
```

That's well under the Kessel Run, even with the full settings. Run `04` last
so the report includes the easter eggs.

## What's in the hangar

```
geonosis/               the model (an importable package)
  system.py             star, planet, ring and 15 moons, plus the stability checks
  spectra.py            SEDs: star, rock/rust/ice reflectance, thermal, ring
  textures.py           procedural craters and albedo maps
  render.py             phase-lit renderer with cast shadows
  site.py               ephemerides for an observer standing on Geonosis
  scopesim_utils.py     ScopeSim setup, Geonosis atmosphere, PSF, detector numbers
  hci.py                Fourier-optics coronagraph + ADI/PCA pipeline for METIS
  easter_eggs.py        AT-ATs, a Death Star, Slave I, and space communist sharks
campaigns/01-05_*.py    runnable scripts, one per step (see Quick start)
tests/test_system.py    sanity checks on the model, renderers and photometry
inst_pkgs/              IRDB packages (Armazones, ELT, METIS, MICADO), ~94 MB, not in git
outputs/                everything the campaigns write (FITS frames are not in git)
  system/               parameter tables (CSV + Markdown), orbits, portraits, SEDs
  earth_metis/          epoch choice, coronagraph, ADI, contrast curves, reduced FITS
  geonosis_micado/      sky chart, night plan, Bahryn mosaic and ice map, gallery, FITS
  easter_eggs/          figures 12-16 plus their MICADO FITS frames
  report.html           all of the above on one self-contained page
```

Each campaign folder has PNG figures, FITS frames, CSV observing plans and a
`summary.json`.

## 1. The system (`geonosis/system.py`)

Every parameter is tagged CANON, DERIVED or INVENTED. The canon values come
from Wookieepedia:
- the star is called Ea, and Geonosis is its 2nd planet;
- diameter 11,370 km, rotation period 30 h, year 256 standard days;
- rocky rings;
- fifteen moons, four of them major, one of which is the icy moon Bahryn.

Canon is light on orbital mechanics, so we filled in the rest so that the
system stays together:

| | |
|---|---|
| Ea | G2V-like, 1 M☉, T_eff 5770 K, 1.3 pc, RA 17h30m, Dec −35° |
| Orbit | a = 0.789 AU (Kepler III with the canon year), e = 0.03 → 1.6× Earth's insolation, T_eq ≈ 291 K |
| Planet | ρ = 4.5 g/cm³ → 0.58 M⊕, 0.73 g surface gravity, obliquity 18° |
| Ring | 1.8–2.9 R_p, τ = 0.3, entirely inside the fluid Roche limit (2.97 R_p) |
| Major moons | Sun Fac (180 km), **Bahryn** (610 km, ice), Poggle (400 km, rusty), Karina (560 km) |
| Minor moons | 11 small rocks; I–III orbit inside/near synchronous orbit (40,875 km), and the outer captured ones are mostly retrograde |

There are 24 dynamical checks, and all of them pass. No moon gets torn apart
or wanders off:
- Roche limits;
- Hill-sphere stability limits (0.49 r_H for prograde orbits, 0.70 r_H for retrograde);
- Gladman's 2√3 mutual-Hill spacing for neighbouring regular moons.

The supporting modules are:
- `spectra.py`: SEDs. Blackbody star, regolith/rock/rust/ice reflectance with
  1.5 and 2.0 µm ice bands, thermal emission, and ring single scattering.
- `textures.py`: deterministic procedural crater fields and albedo maps that
  can be rendered at any scale.
- `render.py`: a phase-lit orthographic renderer with exact cast shadows, plus
  a planet + ring renderer with mutual shadows.
- `site.py`: the sky as seen from the Geonosian observatory, including moon
  positions, phases, eclipses and ring occultations.
- `scopesim_utils.py`: plugs all of this into ScopeSim. It sets up the
  instruments, swaps in the Geonosis atmosphere and PSF, and reads off
  detector constants.

## 2. Campaign A: METIS from Earth (`campaigns/02_earth_metis.py`)

*Can Earth's biggest eye find a planet next to a star 10⁷ times brighter?*

- **Epoch:** the code searches 2031–2032 for maximum elongation while Ea
  transits high at night, and picks 3 consecutive nights.
- **Photometry:** ScopeSim METIS `img_n` (N2) and `img_lm` (L′, M′) give the
  star and companion count rates, the thermal background and the detector
  constants.
- **Direct frame:** a non-coronagraphic ScopeSim frame shows that the star
  saturates even at the minimum DIT.
- **Coronagraph:** IRDB has no METIS coronagraph mode, so `geonosis/hci.py`
  builds METIS' ring-apodized vortex coronagraph (RAVC) from scratch with
  Fourier optics. See [How the coronagraph works](#how-the-coronagraph-works)
  below.
- **ADI:** pupil-tracking frame cubes use the real parallactic-angle track,
  photon and read noise, and DIT/NDIT set by the full well. They are reduced
  with classical ADI and PCA-ADI. The pipeline also produces S/N (Mawet et al.
  2014) and 5σ contrast curves with algorithmic throughput from injected fake
  planets.
- **The moons:** at 1.3 pc the whole planet + ring + moon system spans about
  2.5 mas. That is far below any ELT diffraction limit, so the moons cannot be
  imaged from Earth. These aren't the moons you're looking for.

### How the coronagraph works

*The problem: at 11 µm Ea is about 4 million times brighter than Geonosis
(contrast 2.6×10⁻⁷), and the planet is only 10 λ/D from the star. A
coronagraph is the Jedi mind trick that convinces the starlight to leave and
lets the planet's light through.*

The light passes through four optical planes. Each step is a Fourier
transform or a mask multiplication:

```
 ELT pupil  x  ring apodizer  --FFT-->  vortex phase mask  --IFFT-->  Lyot stop  --MFT-->  METIS detector
 (39 m, 28% hole, 6 spiders)            exp(2iθ), charge 2            (blocks the          (real pixel
                                                                       starlight)           scale)
```

1. **The pupil.** The ELT is modelled as a 39 m disc with a 28% central
   obscuration (R₀ = 0.28) and six 0.54 m spiders, sampled on 192×192 pixels
   and zero-padded 8× for the FFT.
2. **The vortex.** At the focal plane the light is multiplied by a phase
   ramp exp(2iθ). Its phase winds twice around the star, and that is all the
   mask does. There is a neat property: a charge-2 vortex takes a uniformly lit
   disc of radius a and, in the next pupil plane, sends all of its light
   *outside* the disc, with amplitude (a/r)². Inside the disc it is perfectly
   dark. With an unobstructed telescope, all of the starlight would end up
   outside the pupil, where a stop blocks it.
3. **The catch, and the ring.** The ELT has a hole in the middle. The hole
   acts like a "negative disc" of radius R₀, which leaks (R₀/r)² of starlight
   back into the pupil. The RAVC trick (Mawet et al. 2013) is to dim the
   outer part of the pupil, r > R₁, to an amplitude transmission t. The dimmed
   ring then leaks (1 − t)(R₁/r)² of the *opposite* sign. The two leaks cancel
   exactly when

   **t = 1 − R₀² / R₁²**

   The code tries ring radii from 0.36 to 0.64 and keeps the one that lets
   the most planet light through at 10 λ/D. That is **R₁ = 0.52**, so
   **t = 0.71**.
4. **The Lyot stop.** In the dark annulus between R₁ and the pupil edge, a
   mask keeps only 1.02 R₁ < r < 0.96, with spiders twice as wide. Everything
   outside this annulus is thrown away, and that includes the starlight.
5. **The detector.** A matrix Fourier transform puts the field straight onto
   the METIS pixel grid, normalised so that each pixel holds the fraction of
   all the light that entered the telescope. Each band is computed at 3
   wavelengths and averaged, which smears the speckles the way a real filter
   bandwidth does.

**Why the planet survives:** Geonosis is 0.61″ off-axis, so its light lands
on the vortex away from the centre. The phase ramp barely affects it, and it
passes the Lyot stop mostly intact. About **21% of its peak** makes it
through: the apodizer and the stop cost some light, but nothing like the
millions-fold suppression of the star.

**Why it isn't perfect:** a real telescope's wavefront is never flat, and
any phase error scatters starlight around the vortex as speckles. The model
has three error terms, each with its own job:

| Term | rms | Behaviour | Effect |
|---|---|---|---|
| Static NCPA | 40 nm | fixed in the pupil | speckles stay put, so ADI removes them |
| Quasi-static drift | 12 nm, τ = 40 min | slow random walk (AR(1)) | speckles that wander, the real limit on ADI |
| SCAO residuals | 170 nm | change every millisecond | average into a smooth halo; Strehl ≈ 0.99 at N2, 0.92 at L′ |

The SCAO screens have an AO-shaped spectrum: flat inside the control radius
of the 75-actuator system and a Kolmogorov f⁻¹¹ᐟ³ tail outside it. The halo
is the average of 20 random realisations, minus the coherent (Strehl-scaled)
core. Each simulated frame is the star (halo plus speckles from the current
slow wavefront) plus the planet PSF, thermal background and detector noise,
with photometry from ScopeSim. Figure `outputs/earth_metis/03_coronagraph_model.png`
shows the apodized pupil, the Lyot stop and the resulting N2 halo.

## 3. Campaign B: MICADO on Geonosis (`campaigns/03_geonosis_micado.py`)

*What if the Geonosians had built a telescope instead of a Death Star? (They
did both. See the easter eggs.)*

- **Site and night:** an ELT near the Petranaki arena at latitude +12°. The
  code picks the best of 60 Geonosian nights.
- **Ephemerides** for all 15 moons: altitude, distance, apparent size, phase,
  eclipse by Geonosis' shadow, occultation by the ring arc, and tracking
  rates. For comparison, the sidereal rate on Geonosis is 12″/s.
- **Atmosphere:** Armazones' skycalc model is replaced by an invented Geonosis
  atmosphere (dust extinction, ringshine sky glow and warm-air emission). The
  SCAO PSF is an AnisoCADO PSF with Ks Strehl 25%, for dusty seeing with the
  moon as its own wavefront reference.
- **Exposures:** the moons are extremely bright, so the code automatically
  picks the open/ND1/ND3 pupil filter and the DIT that keeps the peak at 70% of
  the H4RG's linear range.
- **Science blocks:**
  - a Bahryn terminator mosaic in Ks;
  - a 1.5 mas zoom;
  - a water-ice map from 1 − xH1/xH2 and 1 − xK1/xK2 band depths, compared with
    the model truth;
  - Karina, Poggle and Sun Fac tiles;
  - the full disc of the 5 km moon XI.

## Results

**From Earth (METIS, 3 nights, 18–20 July 2032, about 5.8 h per night on source):**
- Geonosis sits 0.61″ from Ea (about 10 λ/D at N2).
- At N2 the planet plus its ring has a contrast of 2.6×10⁻⁷, and the ring
  supplies about half the thermal flux. Geonosis is detected at S/N ≈ 5 (cADI)
  and S/N ≈ 4.4 (PCA-ADI). The 5σ limit at its separation is about 1.8×10⁻⁷, so
  N2 is background-limited and needs about 3 nights. **We found it.**
- In L′ (8×10⁻¹⁰) and M′ (4×10⁻⁹) it is not detected. The 5σ limits there are
  about 2.5×10⁻⁸, so reflected light is 30–100× out of reach.
- Without a coronagraph, Ea saturates METIS even at the 11 ms minimum DIT.
- The moons are unresolvable from Earth: the whole system is about 2.5 mas
  across, while λ/D is 59 mas at N2.

**From Geonosis (MICADO, night #43, 12.5 h of darkness):**
- **Sizes in the sky:** Bahryn appears 36′ across, bigger than our Moon from
  Earth. Sun Fac is 23′, Poggle 15′ and Karina 12′. Geonosian nights are busy.
- **Two odd moons:**
  - Moon III orbits close to the synchronous radius, so it hangs almost still
    at 75° altitude.
  - Moon XII slips behind the ring late in the night (22% transmission).
- **Resolution:** at 4 mas MICADO samples Bahryn at 2.2 m/pixel. The 3×3 Ks
  mosaic covers 13 km, which is less than 1% of the disc.
- **Brightness:** every target needs the ND3 filter. Without it the moons
  would saturate the H4RG detectors in under 0.02 s.
- **Ice map:** the narrow-band band-depth maps (1 − xH1/xH2, 1 − xK1/xK2) match
  the true ice distribution with Spearman ρ ≈ 0.97.
- **Small moon:** XI, a 5 km moon, is imaged as a full, half-lit disc 5″
  across at 8 m/pixel.

## Easter eggs (`geonosis/easter_eggs.py`, `campaigns/05_easter_eggs.py`)

The easter eggs are built as physical objects, using canon sizes where canon
gives them. They go through the same renderer and ScopeSim/MICADO pipeline as
the science targets, so the telescope has no idea it is being trolled. Run
`uv run python campaigns/05_easter_eggs.py` after campaign 03; its outputs go
to `outputs/easter_eggs/`.

- **Hoth on Bahryn.** Three AT-AT walkers (22.5 m tall) and a Viper probe
  droid stand on the icy moon, inside the science campaign's 1.5 mas zoom
  tile, so they also hide in the main moon gallery. They are 3-D ray-cast
  boxes that cast real shadows.
  - MICADO picks up the walkers through their 50 m shadows.
  - The 2 m probe droid stays below the 6 m Ks diffraction limit. It
    escapes detection, as it does in the film.
- **"That's no moon."** In canon, the first Death Star was built in orbit of
  Geonosis. DS-1 (160 km) orbits at 72,000 km and appears 8′ across.
  - The model has a panelled hull, an equatorial trench, the superlaser dish
    with tributary beams, and an unfinished southern hemisphere.
  - A MICADO tile on the focusing lens catches an Imperial Star Destroyer
    (1.6 km) passing in front.
- **The *Attack of the Clones* ring chase.** Slave I (21.5 m) drops a seismic
  charge on Obi-Wan's Delta-7 (8 m) among the ring boulders, 7,900 km from
  the site, at 15 cm per pixel.
  - Tracking that ring patch would take 100″/s, which no real ELT can do, so
    this one is fantasy mode.
- **Shark Side of the Moon** (The Asylum, 2022). The communist sharks' base
  "Krasnaya Akula" stands on Poggle, the rust-red moon, with a 1.5 × 1 km
  flag laid out on the regolith: a gold hammerhead shark crossed with a
  sickle, and a gold star, on a red field.
  - In the film the base is on the far side. Poggle is tidally locked, so
    its far side never faces Geonosis. The sharks built on the near side
    instead, near the sub-Geonosis point, so Geonosis hangs 82° up in their
    sky, 3.4° across.
  - "Dark side" therefore means the night side. The code searches 90 nights
    for a crescent Poggle with the base in the dark. It picks night #81:
    phase 109°, altitude 57°, and the sun 22° below the base's horizon.
  - The base is lit by Geonosis-shine, the local version of earthshine:
    gibbous Geonosis plus a nearly edge-on ring give 1.5×10⁻⁴ of full
    sunlight. That makes the regolith Ks ≈ 12.5 mag/arcsec², brighter than
    the 14.0 mag sky. The 19 halogen floodlights are modelled as 3000 K
    blackbodies in full-cutoff fixtures, so only their light pools show.
    The sharks care about light pollution.
  - MICADO resolves the flag emblem, the habitat domes and the scorched
    rocket pad at 3.6 m/pixel. The shark tanks are black because water
    absorbs strongly at 2 µm.
  - In Geonosis-shine the light comes from almost exactly behind the
    observer, so there are no shadows and the craters vanish, as on a full
    moon.
  - The base is a surface overlay on Poggle (`SharkBase` in
    `easter_eggs.py`), so any MICADO render of Poggle that covers it shows it.
- **FITS headers.** Every MICADO frame carries `OBSERVER = 'Poggle the Lesser'`,
  `PROGID = '1138.C-0066(A)'` and a Star Wars quote under
  `HIERARCH SITE QUOTE`. Run `fitsheader` on any frame to see them.

## Caveats (it's a trap!)

- The METIS coronagraph and wavefront model is a representative physical model,
  not the official METIS HCI simulator.
- The Geonosis atmosphere and AO performance are invented. So is Geonosis.
- Renders are orthographic, which is accurate below about 1°. Bahryn's 36′ disc
  is only used for context.
- Non-sidereal tracking and AO on extended targets are assumed to be perfect.
  Only a Sith deals in absolutes; we deal in approximations.
