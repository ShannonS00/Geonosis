"""Spectral energy distributions for Ea, Geonosis, its ring and its moons.

All spectra are ``synphot.SourceSpectrum`` objects in PHOTLAM, so they can be
handed straight to ScopeSim.  Reflectance spectra are simple, physically
motivated toy models (INVENTED): a red desert regolith for Geonosis, dark
rock, rusty rock, and water ice with its 1.5 and 2.0 micron absorption bands.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from synphot import SourceSpectrum, Empirical1D

from .system import GeonosisSystem, R_SUN, lambert_phase

_C = 2.99792458e8
_HC_K = 6.62607015e-34 * _C / 1.380649e-23

WAVE_UM = np.concatenate([np.linspace(0.30, 2.6, 2301), np.linspace(2.61, 30.0, 2740)])

# Reflectance shapes, normalised to 1 at 0.55 micron, then scaled by the
# body's visual geometric albedo.  (wavelength [um], relative reflectance)
_SHAPES = {
    "regolith": ([0.30, 0.45, 0.55, 0.80, 1.20, 1.60, 2.50, 5.0, 30.0],
                 [0.45, 0.62, 1.00, 1.30, 1.42, 1.48, 1.50, 1.45, 1.30]),
    "rock":     ([0.30, 0.55, 1.00, 2.50, 5.0, 30.0],
                 [0.90, 1.00, 1.08, 1.18, 1.20, 1.10]),
    "rust":     ([0.30, 0.45, 0.60, 1.00, 2.50, 5.0, 30.0],
                 [0.45, 0.60, 1.00, 1.40, 1.55, 1.50, 1.30]),
    "ice":      ([0.30, 0.50, 1.00, 1.30, 2.40, 2.80, 3.00, 3.6, 5.0, 30.0],
                 [0.95, 1.00, 0.97, 0.90, 0.80, 0.25, 0.05, 0.25, 0.20, 0.05]),
}
# Water-ice absorption bands: (centre um, depth, sigma um)
_ICE_BANDS = [(1.04, 0.08, 0.03), (1.25, 0.12, 0.04), (1.50, 0.45, 0.07),
              (1.65, 0.12, 0.012), (2.02, 0.62, 0.09)]


def reflectance(surface: str, albedo: float, wave_um=WAVE_UM) -> np.ndarray:
    """Geometric-albedo spectrum for a surface type."""
    key = {"ring": "rock"}.get(surface, surface)
    w, r = _SHAPES[key]
    refl = np.interp(wave_um, w, r)
    if key == "ice":
        for c, d, s in _ICE_BANDS:
            refl *= 1 - d * np.exp(-0.5 * ((wave_um - c) / s) ** 2)
    return np.clip(albedo * refl, 0, 1)


def _spec(wave_um, flux_photlam) -> SourceSpectrum:
    return SourceSpectrum(Empirical1D, points=wave_um * u.um,
                          lookup_table=np.asarray(flux_photlam) * u.ph / u.s / u.cm**2 / u.AA,
                          keep_neg=False)


def blackbody_intensity(temperature, wave_um=WAVE_UM) -> np.ndarray:
    """Planck photon intensity in PHOTLAM per steradian."""
    lam = np.asarray(wave_um) * 1e-6
    x = np.clip(_HC_K / (lam * temperature), None, 700)
    n = 2 * _C / lam**4 / np.expm1(x)          # ph s-1 m-2 m-1 sr-1
    return n * 1e-4 * 1e-10                     # -> ph s-1 cm-2 A-1 sr-1


def star_flux(system: GeonosisSystem, distance_m, wave_um=WAVE_UM) -> np.ndarray:
    """Flux density of Ea [PHOTLAM] at a distance (blackbody photosphere)."""
    s = system.star
    return np.pi * blackbody_intensity(s.teff, wave_um) * (s.radius / distance_m) ** 2


def star_spectrum(system: GeonosisSystem, distance_m=None) -> SourceSpectrum:
    d = system.star.distance if distance_m is None else distance_m
    return _spec(WAVE_UM, star_flux(system, d))


# --------------------------------------------------------------------------
# Geonosis + ring as seen from Earth
# --------------------------------------------------------------------------
def earth_view_geometry(system: GeonosisSystem, t: float) -> dict:
    """Separation, phase angle and ring opening angles at time t [s]."""
    r_orb = system.planet_position(t)
    r_sky = system.orbit_to_sky @ r_orb
    r = np.linalg.norm(r_sky)
    to_sun = -r_sky / r
    cos_alpha = to_sun[2]                       # Earth is along +z
    n_ring = system.orbit_to_sky @ system.eq_to_orbit @ np.array([0, 0, 1.0])
    mu = n_ring[2]                              # cos(emission angle)
    mu0 = n_ring @ to_sun                       # cos(incidence angle)
    d = system.star.distance
    return dict(
        offset_east_arcsec=r_sky[0] / d * 206264.806,
        offset_north_arcsec=r_sky[1] / d * 206264.806,
        separation_arcsec=np.hypot(r_sky[0], r_sky[1]) / d * 206264.806,
        star_distance_m=r,
        phase_angle=np.arccos(np.clip(cos_alpha, -1, 1)),
        ring_mu=mu, ring_mu0=mu0, ring_lit_side_visible=bool(mu * mu0 > 0),
        ring_normal_sky=n_ring,
    )


def ring_reflectance_factor(mu, mu0, tau, alpha):
    """Ring I/F per unit particle reflectivity (single scattering)."""
    mu, mu0 = abs(mu), abs(mu0)
    if mu < 1e-4 or mu0 < 1e-4:
        return 0.0
    slab = 1 - np.exp(-tau * (1 / mu + 1 / mu0))
    return 2 * mu0 / (mu0 + mu) * slab * lambert_phase(alpha)


def planet_components(system: GeonosisSystem, t: float, wave_um=WAVE_UM) -> dict:
    """Flux [PHOTLAM] at Earth of each component of the Geonosis system."""
    g = earth_view_geometry(system, t)
    p, ring = system.planet, system.ring
    d, r, alpha = system.star.distance, g["star_distance_m"], g["phase_angle"]
    insolation = star_flux(system, r, wave_um)          # at Geonosis
    Rp = p.radius

    refl_planet = (insolation * reflectance("regolith", p.geometric_albedo, wave_um)
                   * lambert_phase(alpha) * (Rp / d) ** 2)
    # day / night hemispheres (INVENTED temperatures around T_eq = 291 K)
    day_frac = (1 + np.cos(alpha)) / 2
    therm_planet = np.pi * (Rp / d) ** 2 * (
        day_frac * blackbody_intensity(320.0, wave_um)
        + (1 - day_frac) * blackbody_intensity(255.0, wave_um))

    area = np.pi * ((ring.outer * Rp) ** 2 - (ring.inner * Rp) ** 2)
    mu, mu0 = g["ring_mu"], g["ring_mu0"]
    f_ring = ring_reflectance_factor(mu, mu0, ring.tau, alpha)
    if not g["ring_lit_side_visible"]:
        f_ring *= 0.3                                    # diffuse transmission
    refl_ring = (insolation / np.pi * reflectance("ring", ring.albedo, wave_um)
                 * f_ring * area * abs(mu) / d ** 2)
    t_ring = system.equilibrium_temperature(albedo=ring.albedo)
    therm_ring = (0.9 * blackbody_intensity(t_ring, wave_um)
                  * (1 - np.exp(-ring.tau / max(abs(mu), 1e-3)))
                  * area * abs(mu) / d ** 2)

    moons = np.zeros_like(wave_um)
    for m in system.moons:
        moons += (insolation * reflectance(m.surface, m.albedo, wave_um)
                  * lambert_phase(alpha) * (m.radius / d) ** 2)

    return dict(geometry=g,
                star=star_flux(system, d, wave_um),
                planet_reflected=refl_planet, planet_thermal=therm_planet,
                ring_reflected=refl_ring, ring_thermal=therm_ring,
                moons_reflected=moons)


def companion_spectrum(system: GeonosisSystem, t: float) -> SourceSpectrum:
    """Total spectrum of the unresolved planet + ring + moons point source."""
    c = planet_components(system, t)
    total = sum(c[k] for k in ("planet_reflected", "planet_thermal", "ring_reflected",
                               "ring_thermal", "moons_reflected"))
    return _spec(WAVE_UM, total)


# --------------------------------------------------------------------------
# Surfaces seen from Geonosis
# --------------------------------------------------------------------------
def surface_brightness_spectrum(system: GeonosisSystem, t: float, surface: str,
                                pixel_scale_arcsec: float) -> SourceSpectrum:
    """Flux per source pixel [PHOTLAM] of a surface with reflectance shape
    ``surface`` and unit albedo-weight (image value 1 = normal albedo 1 at
    normal illumination).  I = (I/F) F_sun / pi per steradian."""
    r = np.linalg.norm(system.planet_position(t))
    omega = (pixel_scale_arcsec / 206264.806) ** 2
    flux = star_flux(system, r) / np.pi * reflectance(surface, 1.0) * omega
    return _spec(WAVE_UM, flux)


def planetshine_irradiance(system: GeonosisSystem, moon_pos_eq, t: float,
                           wave_um=WAVE_UM) -> dict:
    """Geonosis-shine at a moon: irradiance from the planet and from the
    ring, relative to the direct sunlight at normal incidence, i.e. the
    planet's flux F = F_sun p(lambda) Phi(alpha) (Rp/d)^2 plus the ring's
    single-scattering flux (as in :func:`planet_components`).  Mutual
    planet/ring shadowing and occultation are ignored."""
    p, ring = system.planet, system.ring
    Rp = p.radius
    r = np.asarray(moon_pos_eq, float).ravel()
    d = np.linalg.norm(r)
    rhat = r / d
    s = system.sun_direction_eq(t).ravel()
    alpha = np.arccos(np.clip(s @ rhat, -1, 1))          # Ea-Geonosis-moon angle
    planet = reflectance("regolith", p.geometric_albedo, wave_um) * lambert_phase(alpha) * (Rp / d) ** 2
    mu, mu0 = abs(rhat[2]), abs(s[2])
    f_ring = ring_reflectance_factor(mu, mu0, ring.tau, alpha)
    if rhat[2] * s[2] < 0:                               # moon above the unlit face
        f_ring *= 0.3
    area = np.pi * ((ring.outer * Rp) ** 2 - (ring.inner * Rp) ** 2)
    ring_part = reflectance("ring", ring.albedo, wave_um) / np.pi * f_ring * area * mu / d**2
    return dict(planet=planet, ring=ring_part, total=planet + ring_part,
                phase_angle=alpha, distance=d, to_planet=-rhat)


def planetshine_surface_spectrum(system: GeonosisSystem, t: float, surface: str,
                                 pixel_scale_arcsec: float, shine: np.ndarray) -> SourceSpectrum:
    """As :func:`surface_brightness_spectrum`, but lit by Geonosis-shine
    (``shine`` = relative irradiance spectrum from :func:`planetshine_irradiance`)."""
    r = np.linalg.norm(system.planet_position(t))
    omega = (pixel_scale_arcsec / 206264.806) ** 2
    flux = star_flux(system, r) / np.pi * reflectance(surface, 1.0) * shine * omega
    return _spec(WAVE_UM, flux)


def lamp_surface_spectrum(power_w: float, temperature: float, surface: str,
                          pixel_scale_arcsec: float) -> SourceSpectrum:
    """Flux per source pixel [PHOTLAM] of a Lambertian surface lit by an
    incandescent lamp radiating ``power_w`` as a blackbody, per unit image
    weight albedo x g, where g [m^-2] is the fraction of the lamp's photons
    landing per square metre of ground."""
    wave_aa = WAVE_UM * 1e4
    nph = blackbody_intensity(temperature)                         # photon shape
    energy = nph * 6.62607015e-34 * _C / (WAVE_UM * 1e-6)          # energy shape
    emitted = power_w * nph / np.trapezoid(energy, wave_aa)        # ph s-1 A-1
    omega = (pixel_scale_arcsec / 206264.806) ** 2
    flux = emitted * 1e-4 / np.pi * reflectance(surface, 1.0) * omega
    return _spec(WAVE_UM, flux)


def band_mean(spec: SourceSpectrum, lo_um=1.99, hi_um=2.31) -> float:
    """Mean photon flux of a spectrum across a band (default Ks)."""
    w = np.linspace(lo_um, hi_um, 200) * u.um
    return float(np.mean(spec(w).value))


def sky_emission_photlam_arcsec2(wave_um, sunlight_photlam, k_band_ab_mag) -> np.ndarray:
    """Scale a sunlight-shaped spectrum to a K-band surface brightness."""
    k = np.argmin(np.abs(wave_um - 2.15))
    lam_cm = 2.15e-4
    h = 6.62607015e-27
    fnu = 3631e-23 * 10 ** (-0.4 * k_band_ab_mag)        # erg/s/cm2/Hz
    photlam = fnu / (h * lam_cm) / 1e8                    # ph/s/cm2/A
    return sunlight_photlam / sunlight_photlam[k] * photlam
