"""Sanity tests for the Geonosis model, renderers and photometry."""

import astropy.units as u
import numpy as np
import pytest

from geonosis.system import GeonosisSystem, AU, DAY, PC, RAD2MAS, lambert_phase
from geonosis import spectra as sp
from geonosis.site import GeonosisObserver
from geonosis.render import View, render_moon, tidally_locked_frame
from geonosis.textures import MoonSurface

SYS = GeonosisSystem()


def test_canon_values():
    p = SYS.planet
    assert 2 * p.radius == pytest.approx(11_370e3)
    assert p.rotation_period == pytest.approx(30 * 3600)
    assert len(SYS.moons) == 15
    assert len(SYS.major_moons) == 4
    bahryn = SYS.moon("Bahryn")
    assert bahryn.surface == "ice" and bahryn.major


def test_kepler_period_matches_canon_year():
    assert SYS.planet_orbit.period == pytest.approx(256 * DAY, rel=1e-9)
    assert SYS.semi_major_axis_au == pytest.approx(0.789, abs=0.002)


def test_orbit_returns_to_start_after_one_period():
    p0 = SYS.planet_position(0.0)
    p1 = SYS.planet_position(SYS.planet_orbit.period)
    assert np.allclose(p0, p1, atol=1.0)


def test_all_stability_checks_pass():
    assert all(c["ok"] for c in SYS.stability_report())


def test_equilibrium_temperature_desert_like():
    assert 280 < SYS.equilibrium_temperature() < 300


def test_angular_scales_by_hand():
    # separation ~ a / d ; 0.789 AU at 1.3 pc = 0.607 arcsec
    assert SYS.planet_orbit.a / AU / 1.30 == pytest.approx(0.607, abs=0.002)
    # planet angular diameter at 1.3 pc ~ 0.06 mas
    assert 2 * SYS.planet.radius / SYS.star.distance * RAD2MAS == pytest.approx(0.0585, rel=0.01)


def test_star_flux_matches_sun_at_10pc():
    f = sp.star_flux(SYS, 10 * PC, np.array([0.55]))[0]
    flam = f * 6.626e-27 * 2.998e10 / 0.55e-4          # erg/s/cm2/A
    v = -2.5 * np.log10(flam / 3.63e-9)
    assert v == pytest.approx(4.83, abs=0.1)


def test_lambert_phase_limits():
    assert lambert_phase(0.0) == pytest.approx(1.0)
    assert lambert_phase(np.pi) == pytest.approx(0.0, abs=1e-12)


def test_ice_bands_present():
    w = np.array([1.40, 1.50, 1.80, 2.02, 2.25])
    r = sp.reflectance("ice", 0.6, w)
    assert r[1] < 0.75 * r[0] and r[3] < 0.6 * r[2]


def test_n_band_contrast_order_of_magnitude():
    c = sp.planet_components(SYS, 64 * DAY)
    m = (sp.WAVE_UM > 10.0) & (sp.WAVE_UM < 12.4)
    total = sum(c[k][m].sum() for k in ("planet_thermal", "ring_thermal", "planet_reflected",
                                         "ring_reflected", "moons_reflected"))
    contrast = total / c["star"][m].sum()
    assert 1e-7 < contrast < 1e-6


def test_render_full_disc_flux_conservation():
    """A Lambertian-free (pure Lommel-Seeliger-ish) full disc at zero phase
    should integrate to ~ pi R^2 x albedo x mean shading."""
    m = SYS.moon("Karina")
    t = 0.0
    r = SYS.moon_position_eq(m, t)
    obs = -r / np.linalg.norm(r)
    view = View(distance=np.linalg.norm(r), to_observer=obs, to_sun=obs,
                up_hint=np.array([0, 0, 1.0]), body_axes=tidally_locked_frame(SYS, m, t))
    n = 300
    ang = 2 * m.radius / view.distance * 206264.806
    pix = 1.2 * ang / n
    out = render_moon(MoonSurface(m), view, n, n, pix, shadows=False, lunar_lambert=1.0)
    # pure Lommel-Seeliger at zero phase: I/F = 1 everywhere on the disc
    disc_area_pix = np.pi * (ang / 2 / pix) ** 2
    assert out["disc"].sum() == pytest.approx(disc_area_pix, rel=0.02)
    lit = out["reflect"][out["disc"]]
    assert np.median(lit) == pytest.approx(1.0, abs=0.05)


def test_site_ephemeris_basic():
    o = GeonosisObserver(SYS)
    assert o.sidereal_rate() == pytest.approx(12.0)
    e = o.moon_ephemeris(SYS.moon("Bahryn"), np.linspace(0, 30 * 3600, 50))
    # Bahryn looks bigger than Earth's Moon (~1865") from Geonosis
    assert e["ang_diam_arcsec"].min() > 1865
    assert e["alt"].max() > 0 > e["alt"].min()


def test_hci_ravc_rejects_on_axis_light():
    from geonosis.hci import RAVC
    cor = RAVC(n_pup=128, pad=8, ring_radius=0.52)
    z = np.zeros((128, 128))
    off = cor.image(z, 11.2e-6, 6.79, 101, coronagraph=False)
    on = cor.image(z, 11.2e-6, 6.79, 101, coronagraph=True)
    assert on.max() / off.max() < 1e-3


def test_easter_egg_ray_box_intersection():
    from geonosis.easter_eggs import _slab
    origins = np.array([[0.0, 5.0], [0.0, 0.0], [0.0, 0.0]])      # under the box / beside it
    t0, t1, axis = _slab(origins, np.array([0, 0, 1.0]), np.array([-1, -1, 2.0]), np.array([1, 1, 4.0]))
    hit = t1 >= np.maximum(t0, 0)
    assert hit.tolist() == [True, False]
    assert t0[0] == pytest.approx(2.0) and t1[0] == pytest.approx(4.0) and axis[0] == 2


def test_easter_eggs_do_not_change_the_moon_count():
    from geonosis.easter_eggs import death_star
    ds = death_star(SYS)
    assert len(SYS.moons) == 15 and ds not in SYS.moons
    assert ds.radius == pytest.approx(80e3)


def test_shark_flag_regions():
    from geonosis.easter_eggs import shark_flag, FLAG_FIELD, FLAG_GOLD, FLAG_BLACK
    x = np.array([0.20, 1.40, 0.005, 0.80, -0.1])     # star, plain field, hem, shark body, outside
    y = np.array([0.80, 0.20, 0.500, 0.48, 0.5])
    assert shark_flag(x, y).tolist() == [FLAG_GOLD, FLAG_FIELD, FLAG_BLACK, FLAG_GOLD, 0]


def test_shark_base_only_paints_poggle_near_the_base():
    from geonosis.easter_eggs import shark_base_for
    pog = SYS.moon("Poggle")
    base = shark_base_for(pog)
    assert base is not None and shark_base_for(SYS.moon("Bahryn")) is None
    surf = MoonSurface(pog)
    la0, lo0 = np.radians(base.lat_deg), np.radians(base.lon_deg)
    hq = base.hq
    lat = np.array([la0 + hq[1] / pog.radius, -0.5])            # HQ dome / far away
    lon = np.array([lo0 + hq[0] / (pog.radius * np.cos(la0)), 1.0])
    alb = surf.evaluate(lat, lon, r_min=5.0)["albedo"]
    ks = sp.reflectance("rust", 1.0, np.array([2.15]))[0]
    assert alb[0] * ks == pytest.approx(0.80)                   # painted HQ dome
    assert alb[1] < 0.5                                         # untouched regolith


def test_geonosis_shine_order_of_magnitude():
    pog = SYS.moon("Poggle")
    t = 3.7 * DAY
    shine = sp.planetshine_irradiance(SYS, SYS.moon_position_eq(pog, t), t)
    k = np.argmin(np.abs(sp.WAVE_UM - 2.15))
    upper = 0.4 * (SYS.planet.radius / shine["distance"]) ** 2  # full phase, p_Ks < 0.4
    assert 0 < shine["planet"][k] < upper
    assert shine["ring"][k] < shine["planet"][k]                # ring seen almost edge-on


def test_lamp_spectrum_conserves_energy():
    lamp = sp.lamp_surface_spectrum(1.0, 3000.0, "rock", 206264.806)   # 1 W, 1 sr pixel
    wave = sp.WAVE_UM
    photlam = lamp(wave * u.um).value
    # undo albedo / pi / cm^2 and integrate photon energy: back to 1 W
    refl = sp.reflectance("rock", 1.0)
    nph = photlam / refl * np.pi * 1e4                          # ph s-1 A-1 per W
    energy = np.trapezoid(nph * 6.62607015e-34 * 2.99792458e8 / (wave * 1e-6), wave * 1e4)
    assert energy == pytest.approx(1.0, rel=0.02)
