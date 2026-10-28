"""Easter-egg campaign: a few extra MICADO blocks in the free part of the
Geonosian night (between the science blocks of 03_geonosis_micado.py).

  * "That's no moon": the Death Star under construction, with an Imperial
    Star Destroyer passing in front of the superlaser dish;
  * the Attack of the Clones ring chase: Slave I and Obi-Wan's Delta-7
    starfighter among the ring boulders, seismic charge included;
  * Hoth on Bahryn: AT-AT walkers and a Viper probe droid, hidden in the
    science campaign's 1.5 mas zoom tile;
  * Shark Side of the Moon: the communist sharks' base and their giant flag
    on the night side of crescent Poggle, lit by Geonosis-shine (a separate
    night: the base is only in the dark when Poggle is a crescent);
  * sky chart with the eggs marked.
Outputs: outputs/easter_eggs/.  Run after 03_geonosis_micado.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Rectangle                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c3", ROOT / "campaigns" / "03_geonosis_micado.py")
c3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c3)

from geonosis import easter_eggs as ee                             # noqa: E402
from geonosis.render import camera_basis, render_moon, moon_weight_maps, image_hdu  # noqa: E402
from geonosis.textures import MoonSurface                          # noqa: E402
from geonosis.system import HOUR, DAY, RAD2ARCSEC                   # noqa: E402
from geonosis import spectra as sp                                 # noqa: E402
from scopesim import Source                                        # noqa: E402

OUT = ROOT / "outputs" / "easter_eggs"
OUT.mkdir(parents=True, exist_ok=True)
c3.OUT = OUT                                   # FITS frames land in easter_eggs/fits
(OUT / "fits").mkdir(exist_ok=True)
SYS, OBS = c3.SYS, c3.OBS
PIX = c3.PIX["IMG_4mas"]
WINDOW = c3.WINDOW


def night_and_gap():
    """Night from the science campaign and its largest unscheduled gap."""
    s = json.loads((ROOT / "outputs" / "geonosis_micado" / "summary.json").read_text())
    t_dusk = s["t_dusk_days"] * DAY
    edges = sorted((b["start_h"], b["end_h"]) for b in s["blocks"])
    gaps, last = [], 0.0
    for a, b in edges:
        gaps.append((last, a))
        last = max(last, b)
    gaps.append((last, s["night_hours"]))
    g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
    zoom = next(b for b in s["blocks"] if b["block"] == "Bahryn zoom 1.5mas")
    return t_dusk, (g0, g1), zoom


def ships_sun_cam(view):
    xc, yc, zc = camera_basis(view.to_observer, view.up_hint)
    s = view.to_sun
    return np.array([s @ xc, s @ yc, s @ zc]), (xc, yc, zc)


def source_from_weights(w, t):
    spec = sp.surface_brightness_spectrum(SYS, t, "rock", PIX)
    return Source(image_hdu=image_hdu(w, PIX), spectra=[spec])


def show(ax, img, title, pix_m, n=None, pct=(1, 99.7)):
    v = np.percentile(img, pct)
    ax.imshow(img, origin="lower", cmap="gray", vmin=v[0], vmax=v[1])
    ax.set_title(title, fontsize=9)
    c3.scale_bar(ax, pix_m, img.shape[0] if n is None else n)
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------
# "That's no moon"
# --------------------------------------------------------------------------
def death_star_block(t):
    # choose the orbital phase that puts DS-1 high, sunlit and dish-on at t
    best, best_score = None, -np.inf
    for m0 in range(0, 360, 2):
        ds = ee.death_star(SYS, m0)
        e = OBS.moon_ephemeris(ds, t)
        if not e["observable"][0] or e["alt"][0] < 35:
            continue
        view = c3.view_for(ds, t)
        dish = view.body_axes @ np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
        lit, facing = dish @ view.to_sun, dish @ view.to_observer
        if lit < 0.25 or facing < 0.3:
            continue
        score = e["alt"][0] + 40 * lit + 40 * facing
        if score > best_score:
            best, best_score = m0, score
    ds = ee.death_star(SYS, best)
    e = OBS.moon_ephemeris(ds, t)
    view = c3.view_for(ds, t)
    surf = ee.DeathStarSurface(ds)
    sun_cam, (xc, yc, zc) = ships_sun_cam(view)
    R_as = ds.radius / view.distance * RAD2ARCSEC
    # tile centred on the superlaser focusing lens
    dish = ds.radius * (view.body_axes @ np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))]))
    centre = np.array([dish @ xc, dish @ yc]) / view.distance * RAD2ARCSEC
    full, r = c3.render_with_margin(surf, view, WINDOW, PIX, centre)
    w = full["albedo"] * full["reflect"]
    # an Imperial Star Destroyer cruising past, ~20 km in front of DS-1
    ee.paint_ship(w, full["pix_m"], "Star Destroyer", 380.0, -640.0, 160.0, sun_cam)
    margin = (w.shape[0] - WINDOW) // 2
    truth = w[margin:margin + WINDOW, margin:margin + WINDOW]
    obs = c3.observe(source_from_weights(w, t), t, e["alt"][0], "Ks", label="death_star_Ks")
    print(f"   Death Star: alt {e['alt'][0]:.0f} deg, {2 * R_as:.0f}\" across, "
          f"ND={obs['pupil']} DIT={obs['dit']:.1f}s x{obs['ndit']}")

    ctx_n = 800
    ctx_pix = 2.3 * R_as / ctx_n
    ctx = render_moon(surf, view, ctx_n, ctx_n, ctx_pix)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(ctx["albedo"] * ctx["reflect"], origin="lower", cmap="gray")
    fx = (centre[0] - PIX * WINDOW / 2) / ctx_pix + ctx_n / 2
    fy = (centre[1] - PIX * WINDOW / 2) / ctx_pix + ctx_n / 2
    ax[0].add_patch(Rectangle((fx, fy), WINDOW * PIX / ctx_pix, WINDOW * PIX / ctx_pix,
                              fill=False, color="red", lw=1.5))
    ax[0].set_title(f"\"That's no moon.\" DS-1 under construction, {e['distance'][0] / 1e3:,.0f} km away, "
                    f"{2 * R_as / 60:.1f}' across\n(red: MICADO field on the superlaser focusing lens)",
                    fontsize=9)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    show(ax[1], truth, "model: focusing lens + Imperial Star Destroyer (1.6 km)", full["pix_m"])
    show(ax[2], obs["calib"], f"MICADO Ks, {full['pix_m']:.1f} m/pixel, {obs['pupil']}, "
                              f"DIT {obs['dit']:.1f}s x {obs['ndit']}", full["pix_m"])
    fig.tight_layout()
    fig.savefig(OUT / "12_death_star.png", dpi=120)
    plt.close(fig)
    return dict(mean_anomaly0_deg=best, alt_deg=float(e["alt"][0]), az_deg=float(e["az"][0]),
                distance_km=float(e["distance"][0]) / 1e3, diameter_arcmin=2 * R_as / 60,
                phase_deg=float(e["phase_angle"][0]), m_per_pixel=full["pix_m"],
                nd=obs["pupil"], dit_s=obs["dit"], ndit=obs["ndit"])


# --------------------------------------------------------------------------
# The ring chase
# --------------------------------------------------------------------------
def ring_chase_block(t):
    Rp = SYS.planet.radius
    r_site, east, north, up = (v.ravel() for v in OBS.site_vectors(t))
    s = SYS.sun_direction_eq(t).ravel()
    psi = np.radians(np.arange(0, 360, 0.5))
    rr = 2.35 * Rp
    pts = rr * np.vstack([np.cos(psi), np.sin(psi), np.zeros_like(psi)])
    d = pts - r_site[:, None]
    dn = d / np.linalg.norm(d, axis=0)
    alt = np.degrees(np.arcsin(dn.T @ up))
    along = s @ pts
    lit = ~((along < 0) & (np.linalg.norm(pts - np.outer(s, along), axis=0) < Rp))
    k = int(np.argmax(np.where(lit, alt, -99)))
    p, dk = pts[:, k], d[:, k]
    D = np.linalg.norm(dk)
    view = c3.View(distance=D, to_observer=-dk / D, to_sun=s, up_hint=up, body_axes=np.eye(3))
    sun_cam, _ = ships_sun_cam(view)
    mu = abs(dk[2] / D)
    pix_m = PIX / RAD2ARCSEC * D
    # apparent motion of the ring patch (circular orbit) relative to the site
    v_orb = np.sqrt(SYS.planet.gm / rr) * np.cross([0, 0, 1.0], p / rr)
    v_site = np.cross([0, 0, 2 * np.pi / SYS.planet.rotation_period], r_site)
    v_rel = v_orb - v_site
    v_t = v_rel - (v_rel @ (dk / D)) * dk / D
    rate = np.linalg.norm(v_t) / D * RAD2ARCSEC
    margin = 256
    n = WINDOW + 2 * margin
    w = ee.render_ring_chase(n, pix_m, sun_cam, SYS.ring.tau / max(mu, 0.05))
    truth = w[margin:margin + WINDOW, margin:margin + WINDOW]
    obs = c3.observe(source_from_weights(w, t), t, alt[k], "Ks", label="ring_chase_Ks")
    print(f"   Ring chase: alt {alt[k]:.0f} deg, {D / 1e3:,.0f} km, {pix_m:.2f} m/pixel, "
          f"ring patch moves {rate:.0f}\"/s; ND={obs['pupil']} DIT={obs['dit']:.2f}s x{obs['ndit']}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    show(ax[0], truth, f"model: ring boulders at {D / 1e3:,.0f} km, Slave I, Delta-7, seismic charge", pix_m)
    show(ax[1], obs["calib"], f"MICADO Ks, {pix_m * 100:.0f} cm/pixel, {obs['pupil']}, "
                              f"DIT {obs['dit']:.2f}s x {obs['ndit']}", pix_m)
    # zoom on the two ships
    c = WINDOW // 2
    zx0, zx1 = int(c - 70 / pix_m), int(c + 70 / pix_m)
    zy0, zy1 = int(c - 40 / pix_m), int(c + 40 / pix_m)
    show(ax[2], obs["calib"][zy0:zy1, zx0:zx1], "zoom: Slave I (upper right) has just dropped a\n"
         "seismic charge on Obi-Wan's Delta-7 (lower left)", pix_m, n=zy1 - zy0)
    fig.suptitle(f"Attack of the Clones in the rings of Geonosis. (Tracking the ring at {rate:.0f}\"/s is "
                 "far beyond any real ELT, so this one is fantasy mode.)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "13_ring_chase.png", dpi=120)
    plt.close(fig)
    return dict(alt_deg=float(alt[k]), az_deg=float(np.degrees(np.arctan2(dn[:, k] @ east, dn[:, k] @ north)) % 360),
                distance_km=D / 1e3, m_per_pixel=pix_m, track_rate_arcsec_s=rate,
                slant_tau=SYS.ring.tau / mu, nd=obs["pupil"], dit_s=obs["dit"], ndit=obs["ndit"])


# --------------------------------------------------------------------------
# Hoth on Bahryn
# --------------------------------------------------------------------------
def hoth_block(t_dusk, zoom):
    t = t_dusk + (zoom["start_h"] + 0.25) * HOUR
    g = c3.single_tile("Bahryn", t, zoom["alt_deg"], mode="IMG_1.5mas", depth=0.6)
    r, obs = g["render"], g["obs"]
    lod_m = 2.15e-6 / 39.0 * g["view"].distance          # Ks diffraction limit on Bahryn
    margin = min(int(round(1.0 / c3.PIX["IMG_1.5mas"])), 512)
    props = {k: (v[0] - margin, v[1] - margin) for k, v in r["props"].items()}
    truth = r["albedo"] * r["reflect"]
    walkers = np.concatenate([props[k][0] for k in props if k.startswith("AT-AT")]), \
        np.concatenate([props[k][1] for k in props if k.startswith("AT-AT")])
    droid = props["Viper probe droid"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 11))
    show(ax[0, 0], truth, "Bahryn 1.5 mas zoom tile (science campaign): model", r["pix_m"])
    show(ax[1, 0], obs["calib"], "MICADO Ks, as in the moon gallery", r["pix_m"])
    for col, (rows, cols, half, name) in enumerate(((walkers[0], walkers[1], 140, "AT-AT walkers"),
                                                    (droid[0], droid[1], 45, "Viper probe droid"))):
        cy, cx = int(np.mean(rows)), int(np.mean(cols))
        sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        show(ax[0, col + 1], truth[sl], f"{name}: model ({r['pix_m']:.2f} m/pixel)", r["pix_m"], n=2 * half)
        show(ax[1, col + 1], obs["calib"][sl], f"{name}: MICADO (lambda/D = {lod_m:.0f} m at Ks)", r["pix_m"],
             n=2 * half)
        for a in (ax[0, 0], ax[1, 0]):
            a.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, color="red", lw=1))
    fig.suptitle("Hoth on Bahryn: imperial walkers and a probe droid on the icy moon. "
                 "MICADO sees their shadows long before it resolves the walkers.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "14_hoth_on_bahryn.png", dpi=110)
    plt.close(fig)
    return dict(m_per_pixel=r["pix_m"], walker_pixels=int(len(walkers[0])), droid_pixels=int(len(droid[0])))


# --------------------------------------------------------------------------
# Shark Side of the Moon
# --------------------------------------------------------------------------
def _anchor_normal(view, base):
    la, lo = np.radians(base.lat_deg), np.radians(base.lon_deg)
    return view.body_axes @ np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def shark_night(moon, base, n_nights=90):
    """Highest moment, over ``n_nights`` Geonosian nights, when Poggle is a
    crescent and the base lies in the night (sun > 10 deg below its
    horizon) and faces the site."""
    t, best = 0.0, None
    for k in range(n_nights):
        a, b = OBS.night_window(t)
        t = b + HOUR
        tt = np.arange(a, b, 600.0)
        e = OBS.moon_ephemeris(moon, tt)
        ok = e["observable"] & (e["phase_angle"] > 100) & (e["phase_angle"] < 155) & (e["alt"] > 35)
        for j in np.flatnonzero(ok):
            view = c3.view_for(moon, tt[j])
            n_a = _anchor_normal(view, base)
            sun_el = np.degrees(np.arcsin(n_a @ view.to_sun))
            if sun_el < -10 and n_a @ view.to_observer > 0.8 and (best is None or e["alt"][j] > best[2]):
                best = (k, tt[j], e["alt"][j], a)
    return best


def ab_mag_arcsec2(spec, weight, pix):
    """Ks AB surface brightness of a source-pixel weight on a template."""
    lam_cm = 2.15e-4
    flam = sp.band_mean(spec) * weight * 6.62607015e-27 * 2.99792458e10 / lam_cm   # erg s-1 cm-2 A-1
    fnu = flam * (lam_cm * 1e8) ** 2 / 2.99792458e18
    return -2.5 * np.log10(fnu / pix**2) - 48.6


def shark_block():
    m = SYS.moon("Poggle")
    base = ee.shark_base_for(m)
    k_night, t, alt, t_dusk_k = shark_night(m, base)
    e = OBS.moon_ephemeris(m, t)
    v0 = c3.view_for(m, t)
    # MICADO's derotator is set to put Poggle's north up
    view = c3.View(distance=v0.distance, to_observer=v0.to_observer, to_sun=v0.to_sun,
                   up_hint=v0.body_axes[:, 2], body_axes=v0.body_axes)
    xc, yc, zc = camera_basis(view.to_observer, view.up_hint)
    n_a = _anchor_normal(view, base)
    R = m.radius
    centre = np.array([n_a @ xc, n_a @ yc]) * R / view.distance * RAD2ARCSEC
    shine = sp.planetshine_irradiance(SYS, SYS.moon_position_eq(m, t), t)
    surf = MoonSurface(m)
    full, r = c3.render_with_margin(surf, view, WINDOW, PIX, centre, secondary_light=shine["to_planet"])

    def lamps(rend):
        e_, n_ = base.local_xy(rend["lat"], rend["lon"], R)
        return np.where(rend["disc"], base.lamp_factor(e_, n_), 0.0)

    specs = {"sun": sp.surface_brightness_spectrum(SYS, t, m.surface, PIX),
             "shine": sp.planetshine_surface_spectrum(SYS, t, m.surface, PIX, shine["total"]),
             "lamp": sp.lamp_surface_spectrum(1.0, base.lamp_temperature, m.surface, PIX)}
    weights = {"sun": full["albedo"] * full["reflect"], "shine": full["albedo"] * full["reflect2"],
               "lamp": full["albedo"] * lamps(full)}
    src = None
    for key, w in weights.items():
        if np.any(w > 0):
            s_ = Source(image_hdu=image_hdu(w, PIX), spectra=[specs[key]])
            src = s_ if src is None else src + s_
    obs = c3.observe(src, t, alt, "Ks", t_total=120.0, label="shark_base_Ks")
    # Ks brightness of each light source relative to sunlight, for the model panels
    ks = {key: sp.band_mean(sp_) / sp.band_mean(specs["sun"]) for key, sp_ in specs.items()}
    truth = r["albedo"] * (r["reflect"] + ks["shine"] * r["reflect2"] + ks["lamp"] * lamps(r))
    sun_el = np.degrees(np.arcsin(n_a @ view.to_sun))
    geo_el = np.degrees(np.arcsin(n_a @ shine["to_planet"]))
    geo_diam = 2 * np.degrees(np.arcsin(SYS.planet.radius / shine["distance"]))
    ks_i = np.argmin(np.abs(sp.WAVE_UM - 2.15))
    shine_ks = float(shine["total"][ks_i])
    regolith = ab_mag_arcsec2(specs["shine"], m.albedo, PIX)
    print(f"   Shark base: night #{k_night}, Poggle alt {alt:.0f} deg, phase {e['phase_angle'][0]:.0f} deg; "
          f"sun {-sun_el:.0f} deg below the base's horizon; Geonosis-shine = {shine_ks:.1e} x sunlight; "
          f"ND={obs['pupil']} DIT={obs['dit']:.1f}s x{obs['ndit']}")

    # ---- figure
    pix_m = r["pix_m"]
    e_r, n_r = base.local_xy(r["lat"], r["lon"], R)

    def zoom_box(pe, pn, half_m):
        j = np.unravel_index(np.argmin((e_r - pe) ** 2 + (n_r - pn) ** 2), e_r.shape)
        h = int(half_m / pix_m)
        y0, x0 = max(j[0] - h, 0), max(j[1] - h, 0)
        return slice(y0, min(j[0] + h, WINDOW)), slice(x0, min(j[1] + h, WINDOW))

    R_as = R / view.distance * RAD2ARCSEC
    ctx_n = 800
    ctx_pix = 2.3 * R_as / ctx_n
    ctx = render_moon(surf, view, ctx_n, ctx_n, ctx_pix, secondary_light=shine["to_planet"])
    ctx_img = ctx["albedo"] * (ctx["reflect"] + ks["shine"] * ctx["reflect2"])
    fig, ax = plt.subplots(2, 3, figsize=(18, 12.4))
    # stretched like an earthshine photo: the crescent saturates, the night side shows
    ax[0, 0].imshow(np.arcsinh(ctx_img / (0.5 * ks["shine"] * m.albedo)), origin="lower", cmap="gray",
                    vmin=0, vmax=np.arcsinh(40.0))
    fx = (centre[0] - PIX * WINDOW / 2) / ctx_pix + ctx_n / 2
    fy = (centre[1] - PIX * WINDOW / 2) / ctx_pix + ctx_n / 2
    ax[0, 0].add_patch(Rectangle((fx - 4, fy - 4), WINDOW * PIX / ctx_pix + 8, WINDOW * PIX / ctx_pix + 8,
                                 fill=False, color="red", lw=1.5))
    ax[0, 0].set_title(f"Poggle, phase {e['phase_angle'][0]:.0f} deg: sunlit crescent + night side in "
                       f"Geonosis-shine\n(asinh stretch, crescent saturated; red: MICADO field on the base, "
                       f"{e['distance'][0] / 1e3:,.0f} km away)", fontsize=9)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].imshow(ee.shark_flag_rgb(), origin="lower")
    ax[0, 1].set_title(f"The Red Shark flag (design, visible colours): "
                       f"{base.flag_width / 1e3:.1f} x {base.flag_width / 1.5e3:.1f} km\n"
                       "gold hammerhead and sickle, gold star, red field", fontsize=9)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    show(ax[0, 2], truth, "model Ks: base 'Krasnaya Akula' in Geonosis-shine + floodlights", pix_m, pct=(0.5, 99.0))
    show(ax[1, 0], obs["calib"], f"MICADO Ks, {pix_m:.1f} m/pixel, {obs['pupil']}, "
                                 f"DIT {obs['dit']:.1f}s x {obs['ndit']}", pix_m, pct=(0.5, 99.0))
    sl = zoom_box(*base.flag_centre, 0.62 * base.flag_width)
    show(ax[1, 1], obs["calib"][sl], "MICADO zoom: the flag", pix_m, n=sl[0].stop - sl[0].start, pct=(0.5, 99.5))
    sl = zoom_box(base.hq[0] - 150, base.hq[1] - 250, 950)
    show(ax[1, 2], obs["calib"][sl], "MICADO zoom: HQ dome, habitats, shark tanks (black: water), "
                                     "landing pad", pix_m, n=sl[0].stop - sl[0].start, pct=(0.5, 99.5))
    fig.suptitle(f"Shark Side of the Moon: the communist sharks' base on the night side of Poggle. "
                 f"The sun is {-sun_el:.0f} deg below their horizon; Geonosis hangs {geo_el:.0f} deg up, "
                 f"{geo_diam:.1f} deg across, and lights the regolith at {shine_ks:.1e} x sunlight "
                 f"(Ks {regolith:.1f} mag/arcsec2, sky 14.0).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(OUT / "16_shark_side_of_the_moon.png", dpi=110)
    plt.close(fig)
    return dict(night_index=k_night, hours_after_dusk=(t - t_dusk_k) / HOUR, alt_deg=float(alt),
                az_deg=float(e["az"][0]), distance_km=float(e["distance"][0]) / 1e3,
                phase_deg=float(e["phase_angle"][0]), base_lat_deg=base.lat_deg, base_lon_deg=base.lon_deg,
                sun_elevation_at_base_deg=float(sun_el), geonosis_elevation_at_base_deg=float(geo_el),
                geonosis_diameter_deg=float(geo_diam),
                geonosis_illuminated_fraction=float((1 + np.cos(shine["phase_angle"])) / 2),
                shine_vs_sunlight_ks=shine_ks,
                ring_share_of_shine_ks=float(shine["ring"][ks_i] / shine["total"][ks_i]),
                regolith_ks_ab_mag_arcsec2=float(regolith), flag_km=[base.flag_width / 1e3, base.flag_width / 1.5e3],
                n_floodlights=len(base.lamps), m_per_pixel=pix_m,
                nd=obs["pupil"], dit_s=obs["dit"], ndit=obs["ndit"])


def main():
    t_dusk, (g0, g1), zoom = night_and_gap()
    print(f"free gap in the night plan: {g0:.2f}-{g1:.2f} h after dusk")
    t_ds = t_dusk + (g0 + 0.35 * (g1 - g0)) * HOUR
    t_ring = t_dusk + (g0 + 0.75 * (g1 - g0)) * HOUR
    res = {"death_star": death_star_block(t_ds), "ring_chase": ring_chase_block(t_ring),
           "hoth_on_bahryn": hoth_block(t_dusk, zoom), "shark_side_of_the_moon": shark_block()}
    ds, rc = res["death_star"], res["ring_chase"]
    c3.sky_chart(t_ds, OUT / "15_sky_chart_easter_eggs.png", extras=[
        dict(alt=ds["alt_deg"], az=ds["az_deg"], label="That's no moon", color="#ff5555", marker="o", size=90),
        dict(alt=rc["alt_deg"], az=rc["az_deg"], label="Slave I chase", color="#55ff88", marker="*", size=120),
    ])
    res["blocks"] = [
        dict(block="DS-1 'That's no moon'", start_h=round(g0 + 0.25 * (g1 - g0), 2),
             end_h=round(g0 + 0.45 * (g1 - g0), 2), mode="SCAO + IMG_4mas", filters="Ks"),
        dict(block="Ring chase", start_h=round(g0 + 0.65 * (g1 - g0), 2),
             end_h=round(g0 + 0.85 * (g1 - g0), 2), mode="SCAO + IMG_4mas", filters="Ks"),
        dict(block="Shark Side of the Moon (separate night "
                   f"#{res['shark_side_of_the_moon']['night_index']})",
             start_h=round(res["shark_side_of_the_moon"]["hours_after_dusk"], 2),
             end_h=round(res["shark_side_of_the_moon"]["hours_after_dusk"] + 0.5, 2),
             mode="SCAO + IMG_4mas", filters="Ks"),
    ]
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
