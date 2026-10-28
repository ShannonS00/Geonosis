"""Campaign B: an ELT with MICADO on the surface of Geonosis, imaging its moons.

Steps
  1. Find the best Geonosian night (major moons high, gibbous, not eclipsed).
  2. Ephemerides of all 15 moons from the Petranaki site; sky chart incl. ring.
  3. Night plan: targets, modes, filters, ND, DIT/NDIT, tracking rates.
  4. Simulate MICADO (SCAO, Geonosis atmosphere + PSF) observations:
       - Bahryn terminator mosaic (Ks, 4 mas) + 1.5 mas zoom
       - Bahryn water-ice map (xH1/xH2, xK1/xK2 narrow bands)
       - Karina, Poggle, Sun Fac terminator tiles
       - full disc of the small moon XI
Outputs go to outputs/geonosis_micado/.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402
from astropy.io import fits                                      # noqa: E402
from scipy import ndimage, stats                                 # noqa: E402

from geonosis.system import GeonosisSystem, HOUR, DAY            # noqa: E402
from scopesim import Source                                      # noqa: E402
from geonosis.site import GeonosisObserver                       # noqa: E402
from geonosis.render import (View, camera_basis, render_moon,    # noqa: E402
                             tidally_locked_frame, moon_weight_maps, image_hdu)
from geonosis.textures import MoonSurface                        # noqa: E402
from geonosis import spectra as sp                               # noqa: E402
from geonosis import scopesim_utils as su                        # noqa: E402
from geonosis import easter_eggs                                 # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "geonosis_micado"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fits").mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv

SYS = GeonosisSystem()
OBS = GeonosisObserver(SYS)
WINDOW = 1024 if QUICK else 2048         # detector window [pixels]
MINDIT = 1.3                              # s, assumed H4RG full-frame minimum DIT
FULL_WELL = 1.0e5                         # e-, linear range of the H4RG (IRDB curve)
FILL = 0.7
STREHL_KS = 0.25                          # INVENTED: dusty Geonosis seeing, SCAO on the moon itself
PIX = {"IMG_4mas": 0.004, "IMG_1.5mas": 0.0015}
ND_TRANSMISSION = {"open": 1.0, "ND1": 0.1, "ND3": 1e-3}   # IRDB TC_filter_ND*.dat


# --------------------------------------------------------------------------
# 1. choose the night
# --------------------------------------------------------------------------
def best_night(n_nights=60):
    t, rows = 0.0, []
    for k in range(n_nights):
        a, b = OBS.night_window(t)
        t = b + HOUR
        tt = np.arange(a, b, 900.0)
        score, hrs = 0.0, {}
        for m in SYS.major_moons:
            e = OBS.moon_ephemeris(m, tt)
            good = e["observable"] & (e["phase_angle"] > 25) & (e["phase_angle"] < 95) & (e["alt"] > 35)
            hrs[m.name] = good.sum() * 0.25
            score += hrs[m.name] * (3 if m.name == "Bahryn" else 1)
        rows.append((score, k, a, b, hrs))
    rows.sort(key=lambda r: -r[0])
    return rows[0]


# --------------------------------------------------------------------------
# 2. figures: sky chart and ephemerides
# --------------------------------------------------------------------------
def sky_chart(t, fname, extras=()):
    """Polar alt-az chart of the moons and ring arc.  ``extras``: extra
    markers, dicts with alt, az, label, size, color, marker."""
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("#0b0b1a")
    ring = OBS.ring_altaz(t)
    for edge in ("inner", "outer"):
        r = ring[edge]
        up = r["alt"] > 0
        ax.scatter(np.radians(r["az"][up & r["lit"]]), 90 - r["alt"][up & r["lit"]], s=1.5,
                   color="#d9a066")
        ax.scatter(np.radians(r["az"][up & ~r["lit"]]), 90 - r["alt"][up & ~r["lit"]], s=1.5,
                   color="#4a3a2a")
    for m in SYS.moons:
        e = OBS.moon_ephemeris(m, t)
        if e["alt"][0] <= 0:
            continue
        size = np.clip(e["ang_diam_arcsec"][0] / 25, 6, 120)
        col = "#bfe3ff" if m.surface == "ice" else ("#c8743c" if m.surface == "rust" else "#b0b0b0")
        if e["eclipsed"][0]:
            col = "#333333"
        ax.scatter(np.radians(e["az"][0]), 90 - e["alt"][0], s=size, color=col, zorder=3,
                   edgecolor="w", linewidth=0.3)
        ax.annotate(m.label, (np.radians(e["az"][0]), 90 - e["alt"][0]), color="w", fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    for x in extras:
        ax.scatter(np.radians(x["az"]), 90 - x["alt"], s=x.get("size", 60), color=x.get("color", "r"),
                   marker=x.get("marker", "*"), zorder=4, edgecolor="w", linewidth=0.4)
        ax.annotate(x["label"], (np.radians(x["az"]), 90 - x["alt"]), color=x.get("color", "r"),
                    fontsize=9, fontweight="bold", xytext=(6, -12), textcoords="offset points")
    alt_s, az_s = OBS.sun_altaz(t)
    ax.set_rlim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["60", "30", "0"], color="0.6")
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.grid(color="0.35", alpha=0.5)
    ax.set_title(f"Sky above the Petranaki ELT (lat +12 deg), Ea altitude {alt_s[0]:.0f} deg\n"
                 "ring arc: lit (tan) / in Geonosis' shadow (brown); marker size ~ apparent size",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def ephemeris_table(t_dusk, t_dawn):
    tt = np.linspace(t_dusk, t_dawn, 61)
    rows = []
    for m in SYS.moons:
        e = OBS.moon_ephemeris(m, tt)
        up = e["observable"]
        k = int(np.argmax(np.where(up, e["alt"], -99)))
        rows.append(dict(
            moon=m.label, radius_km=m.radius / 1e3, surface=m.surface,
            hours_observable=round(up.mean() * (t_dawn - t_dusk) / HOUR, 1),
            max_alt_deg=round(float(e["alt"][k]), 1),
            distance_km=round(float(e["distance"][k]) / 1e3),
            ang_diam_arcsec=round(float(e["ang_diam_arcsec"][k]), 1),
            phase_deg=round(float(e["phase_angle"][k])),
            m_per_4mas_pixel=round(float(e["m_per_mas"][k]) * 4, 2),
            rate_inertial_arcsec_s=round(float(e["rate_inertial"][k]), 2),
            rate_vs_sidereal=round(float(e["rate_inertial"][k]) / OBS.sidereal_rate(), 2),
            eclipsed_any=bool(e["eclipsed"].any()),
            min_ring_transmission=round(float(e["ring_transmission"].min()), 2)))
    return rows


def fig_timeline(t_dusk, t_dawn, blocks):
    tt = np.linspace(t_dusk, t_dawn, 300)
    fig, ax = plt.subplots(figsize=(11, 5))
    names = sorted({b["target"] for b in blocks}, key=lambda n: SYS.moon(n).a)
    for name in names:
        e = OBS.moon_ephemeris(SYS.moon(name), tt)
        ax.plot((tt - t_dusk) / HOUR, e["alt"], label=name)
    for b in blocks:
        ax.axvspan(b["start_h"], b["end_h"], color="0.85", alpha=0.6)
        ax.text((b["start_h"] + b["end_h"]) / 2, 92, b["block"], rotation=90, fontsize=7,
                ha="center", va="bottom")
    ax.axhline(20, color="r", ls=":", lw=0.8)
    ax.set_ylim(0, 120)
    ax.set_xlabel("hours after dusk (Geonosian night)")
    ax.set_ylabel("altitude [deg]")
    ax.set_title("Night plan: target altitudes and scheduled MICADO blocks")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "08_night_plan.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. observation machinery
# --------------------------------------------------------------------------
def view_for(moon, t):
    d, r_m, (r_site, e, n, up) = OBS.target_vector(moon, t)
    d = d.ravel()
    return View(distance=float(np.linalg.norm(d)), to_observer=-d / np.linalg.norm(d),
                to_sun=SYS.sun_direction_eq(t).ravel(), up_hint=up.ravel(),
                body_axes=tidally_locked_frame(SYS, moon, t))


def terminator_offset(moon, view, depth=0.85):
    """Tile centre (arcsec) on the lit side close to the terminator."""
    xc, yc, zc = camera_basis(view.to_observer, view.up_hint)
    sxy = np.array([view.to_sun @ xc, view.to_sun @ yc])
    sxy /= np.linalg.norm(sxy)
    R_as = moon.radius / view.distance * 206264.806
    return -sxy * R_as * np.cos(view.phase_angle) * depth, R_as


def render_with_margin(surf, view, window, pix, centre, **render_kw):
    """Render the scene 1 arcsec beyond the detector window, so that light
    from outside the field is scattered in by the PSF halo (as in reality).
    Returns (full render for the ScopeSim source, render cropped to the window)."""
    margin = min(int(round(1.0 / pix)), 512)
    n = window + 2 * margin
    full = render_moon(surf, view, n, n, pix, center_arcsec=tuple(centre), **render_kw)
    crop = {k: (v[margin:margin + window, margin:margin + window]
                if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in full.items()}
    return full, crop


def build_source(render, surface_type, t, pix):
    w = moon_weight_maps(render, surface_type)
    src = None
    for comp, img in w.items():
        if not np.any(img > 0):
            continue
        spec = sp.surface_brightness_spectrum(SYS, t, comp, pix)
        s_ = Source(image_hdu=image_hdu(img, pix), spectra=[spec])
        src = s_ if src is None else src + s_
    return src


def observe(src, t, alt, fw2, mode="IMG_4mas", window=WINDOW, t_total=60.0, label=""):
    """Choose ND + DIT so the peak stays at 70 % of full well, then read out."""
    airmass = 1 / np.sin(np.radians(max(alt, 5)))
    opt = su.micado_train(fw2=fw2, pupil="open", mode=mode, window=window, system=SYS, t=t,
                          airmass=airmass, strehl_ks=STREHL_KS)
    rate_open = su.rate_map(opt, src)
    # ND filters are grey: pick the weakest one that allows DIT >= MINDIT
    for pupil in ("open", "ND1", "ND3"):
        dit = FILL * FULL_WELL / (rate_open.max() * ND_TRANSMISSION[pupil])
        if dit >= MINDIT:
            break
    if pupil != "open":
        opt["pupil_wheel"].change_filter(pupil)
        rate = su.rate_map(opt, src)
        dit = FILL * FULL_WELL / rate.max()
    else:
        rate = rate_open
    dit = float(min(dit, t_total))
    ndit = max(1, int(round(t_total / dit)))
    hdul = opt.readout(dit=dit, ndit=ndit)[0]
    data = hdul[1].data.astype(float)
    if label:
        for key, val in easter_eggs.fits_cards(label).items():
            hdul[0].header[key] = val
        hdul.writeto(OUT / "fits" / f"{label}.fits", overwrite=True)
    # rate-calibrated image [e-/s] with the ND attenuation undone, so tiles
    # taken with different ND filters share one scale.  The Geonosis sky
    # (~14 mag/arcsec^2) is ~1e-4 of the moons' surface brightness: ignored.
    calib = data / (dit * ndit) / ND_TRANSMISSION[pupil]
    return dict(data=data, calib=calib, rate=rate, dit=dit, ndit=ndit, pupil=pupil,
                peak_rate=float(rate.max()), airmass=airmass)


# --------------------------------------------------------------------------
# 4. science blocks
# --------------------------------------------------------------------------
def scale_bar(ax, pix_m, npix, color="w"):
    target = npix * pix_m / 5
    nice = 10 ** np.floor(np.log10(target))
    for f in (5, 2, 1):
        if f * nice <= target:
            length = f * nice
            break
    lp = length / pix_m
    ax.plot([0.05 * npix, 0.05 * npix + lp], [0.06 * npix] * 2, color=color, lw=3)
    lab = f"{length / 1e3:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(0.05 * npix + lp / 2, 0.09 * npix, lab, color=color, ha="center", fontsize=9)


def bahryn_mosaic(t, alt, n_side):
    m = SYS.moon("Bahryn")
    view = view_for(m, t)
    surf = MoonSurface(m)
    pix = PIX["IMG_4mas"]
    centre, R_as = terminator_offset(m, view)
    step = WINDOW * pix * 0.95                         # 5 % overlap
    tiles = []
    for iy in range(n_side):
        for ix in range(n_side):
            off = centre + np.array([(ix - (n_side - 1) / 2) * step, (iy - (n_side - 1) / 2) * step])
            full, r = render_with_margin(surf, view, WINDOW, pix, off)
            src = build_source(full, "ice", t, pix)
            obs = observe(src, t, alt, "Ks", label=f"bahryn_mosaic_{iy}{ix}_Ks")
            tiles.append(dict(off=off, render=r, obs=obs))
            print(f"     tile {iy}{ix}: ND={obs['pupil']} DIT={obs['dit']:.1f}s x{obs['ndit']}")
    # assemble
    span = int(round(step / pix * (n_side - 1))) + WINDOW
    canvas = np.zeros((span, span))
    truth = np.zeros((span, span))
    weight = np.zeros((span, span))
    o0 = centre - (n_side - 1) / 2 * step
    for tl in tiles:
        x0 = int(round((tl["off"][0] - o0[0]) / pix))
        y0 = int(round((tl["off"][1] - o0[1]) / pix))
        canvas[y0:y0 + WINDOW, x0:x0 + WINDOW] += tl["obs"]["calib"]
        truth[y0:y0 + WINDOW, x0:x0 + WINDOW] += tl["render"]["albedo"] * tl["render"]["reflect"]
        weight[y0:y0 + WINDOW, x0:x0 + WINDOW] += 1
    canvas /= np.maximum(weight, 1)
    truth /= np.maximum(weight, 1)
    fits.PrimaryHDU(canvas.astype(np.float32)).writeto(OUT / "fits" / "bahryn_mosaic_Ks.fits", overwrite=True)

    # context render (full disc) with mosaic footprint
    ctx_n = 700
    ctx_pix = 2.3 * R_as / ctx_n
    ctx = render_moon(surf, view, ctx_n, ctx_n, ctx_pix, shadows=True)
    pix_m = tiles[0]["render"]["pix_m"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(ctx["albedo"] * ctx["reflect"], origin="lower", cmap="gray")
    fx = (o0[0] - pix * WINDOW / 2) / ctx_pix + ctx_n / 2
    fy = (o0[1] - pix * WINDOW / 2) / ctx_pix + ctx_n / 2
    ax[0].add_patch(Rectangle((fx, fy), span * pix / ctx_pix, span * pix / ctx_pix,
                              fill=False, color="red", lw=1.5))
    ax[0].set_title(f"Bahryn from Petranaki: {2 * R_as / 60:.0f}' across, phase {np.degrees(view.phase_angle):.0f} deg\n"
                    "(red: MICADO mosaic footprint, barely visible!)")
    ax[1].imshow(truth, origin="lower", cmap="gray")
    ax[1].set_title("model surface (truth) at 4 mas")
    scale_bar(ax[1], pix_m, span)
    v = np.percentile(canvas, [1, 99.7])
    ax[2].imshow(canvas, origin="lower", cmap="gray", vmin=v[0], vmax=v[1])
    ax[2].set_title(f"MICADO Ks {n_side}x{n_side} mosaic (SCAO, Strehl {STREHL_KS:.0%}), {pix_m:.1f} m/pixel")
    scale_bar(ax[2], pix_m, span)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "09_bahryn_mosaic.png", dpi=120)
    plt.close(fig)
    return tiles, dict(step_arcsec=step, n_tiles=n_side**2, pix_m=pix_m, span_km=span * pix_m / 1e3)


def bahryn_ice_map(t, alt):
    """Water-ice band depths from MICADO narrow-band filters."""
    m = SYS.moon("Bahryn")
    view = view_for(m, t)
    surf = MoonSurface(m)
    pix = PIX["IMG_4mas"]
    centre, _ = terminator_offset(m, view, depth=0.55)
    full, r = render_with_margin(surf, view, WINDOW, pix, centre)
    src = build_source(full, "ice", t, pix)
    obs = {}
    for f in ("xH1", "xH2", "xK1", "xK2"):
        obs[f] = observe(src, t, alt, f, t_total=120.0, label=f"bahryn_ice_{f}")
        print(f"     {f}: ND={obs[f]['pupil']} DIT={obs[f]['dit']:.1f}s x{obs[f]['ndit']}")
    # calibrate band ratios with the pure-rock (band-free) colour, measured
    # through the same optical trains on a uniform rock source
    ref = {}
    flat = np.ones((64, 64))
    for f in obs:
        rs = Source(image_hdu=image_hdu(flat, pix), spectra=[sp.surface_brightness_spectrum(SYS, t, "rock", pix)])
        opt = su.micado_train(fw2=f, pupil="open", window=64, system=SYS, t=t,
                              airmass=obs[f]["airmass"], strehl_ks=None)
        rate = su.rate_map(opt, rs)
        ref[f] = np.median(rate[16:48, 16:48])
    def norm(f):
        return obs[f]["calib"] / ref[f]

    # well-lit pixels only: band ratios are meaningless in (and next to) shadows
    lit = r["reflect"] > 0.3 * np.percentile(r["reflect"], 90)
    lit = ndimage.binary_erosion(lit, iterations=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        d15 = np.where(lit, 1 - norm("xH1") / norm("xH2"), np.nan)
        d20 = np.where(lit, 1 - norm("xK1") / norm("xK2"), np.nan)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(obs["xK2"]["calib"], origin="lower", cmap="gray")
    ax[0].set_title("MICADO xK2 (2.24 um continuum)")
    ax[1].imshow(np.where(lit, r["ice"], np.nan), origin="lower", cmap="Blues",
                 vmin=np.nanpercentile(r["ice"][lit], 2), vmax=np.nanpercentile(r["ice"][lit], 98))
    ax[1].set_title("model ice fraction (truth)")
    im = ax[2].imshow(d15, origin="lower", cmap="Blues", vmin=np.nanpercentile(d15, 2),
                      vmax=np.nanpercentile(d15, 98))
    ax[2].set_title("1.5 um band depth: 1 - xH1/xH2")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    im = ax[3].imshow(d20, origin="lower", cmap="Blues", vmin=np.nanpercentile(d20, 2),
                      vmax=np.nanpercentile(d20, 98))
    ax[3].set_title("2.0 um band depth: 1 - xK1/xK2")
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    scale_bar(ax[0], r["pix_m"], WINDOW)
    fig.suptitle("Bahryn water-ice mapping with MICADO narrow-band filters")
    fig.tight_layout()
    fig.savefig(OUT / "10_bahryn_ice_map.png", dpi=120)
    plt.close(fig)
    # compare at the resolution MICADO actually delivers (PSF ~ 3 pixels)
    truth_s = ndimage.gaussian_filter(r["ice"], 3)
    corr = {}
    for key, d in (("1p5", d15), ("2p0", d20)):
        ok = np.isfinite(d) & lit
        lo, hi = np.percentile(d[ok], [1, 99])
        ok &= (d > lo) & (d < hi)
        corr[key] = float(stats.spearmanr(d[ok], truth_s[ok])[0])
    return obs, dict(spearman_1p5um_vs_truth=corr["1p5"], spearman_2p0um_vs_truth=corr["2p0"],
                     median_depth_1p5=float(np.nanmedian(d15)),
                     median_depth_2p0=float(np.nanmedian(d20)))


def single_tile(name, t, alt, fw2="Ks", mode="IMG_4mas", depth=0.85, full_disc=False,
                window=None, t_total=60.0):
    m = SYS.moon(name)
    view = view_for(m, t)
    surf = MoonSurface(m)
    pix = PIX[mode]
    window = window or WINDOW
    centre = (0.0, 0.0) if full_disc else tuple(terminator_offset(m, view, depth)[0])
    full, r = render_with_margin(surf, view, window, pix, centre)
    src = build_source(full, m.surface, t, pix)
    obs = observe(src, t, alt, fw2, mode=mode, window=window, t_total=t_total,
                  label=f"{name.replace(' ', '_')}_{mode}_{fw2}")
    print(f"     {name} {mode} {fw2}: ND={obs['pupil']} DIT={obs['dit']:.2f}s x{obs['ndit']}")
    return dict(render=r, obs=obs, view=view)


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("1) choosing the night ...")
    score, k, t_dusk, t_dawn, hrs = best_night()
    print(f"   Geonosian night #{k} (t = {t_dusk / DAY:.2f} d), {(t_dawn - t_dusk) / HOUR:.1f} h dark; {hrs}")
    t_mid = 0.5 * (t_dusk + t_dawn)

    print("2) ephemerides + sky chart ...")
    sky_chart(t_dusk + 4 * HOUR, OUT / "07_sky_chart.png")
    eph = ephemeris_table(t_dusk, t_dawn)
    with open(OUT / "moon_ephemerides.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eph[0]))
        w.writeheader()
        w.writerows(eph)

    # 3) Night plan: put each block where its target is highest & observable
    plan = [  # (block, target, duration_h)
        ("Bahryn mosaic Ks", "Bahryn", 1.0),
        ("Bahryn ice map", "Bahryn", 1.0),
        ("Bahryn zoom 1.5mas", "Bahryn", 0.5),
        ("Karina Ks", "Karina", 0.5),
        ("Poggle Ks", "Poggle", 0.5),
        ("Sun Fac Ks", "Sun Fac", 0.5),
        ("XI full disc H+Ks", "XI", 0.75),
    ]
    tt = np.arange(t_dusk, t_dawn, 300.0)
    busy = np.zeros(len(tt), bool)
    blocks = []
    for block, target, dur in sorted(plan, key=lambda p: -p[2]):
        e = OBS.moon_ephemeris(SYS.moon(target), tt)
        nslot = int(round(dur * HOUR / 300.0))
        best, best_alt = None, -1
        for s0 in range(0, len(tt) - nslot):
            sl = slice(s0, s0 + nslot)
            if busy[sl].any() or not e["observable"][sl].all():
                continue
            a = e["alt"][sl].mean()
            if a > best_alt:
                best, best_alt = s0, a
        if best is None:
            print(f"   ! could not schedule {block}")
            continue
        busy[best:best + nslot] = True
        tm = tt[best + nslot // 2]
        em = OBS.moon_ephemeris(SYS.moon(target), tm)
        blocks.append(dict(block=block, target=target,
                           start_h=round((tt[best] - t_dusk) / HOUR, 2),
                           end_h=round((tt[best] + nslot * 300 - t_dusk) / HOUR, 2),
                           t_mid=float(tm), alt_deg=round(float(em["alt"][0]), 1),
                           phase_deg=round(float(em["phase_angle"][0])),
                           distance_km=round(float(em["distance"][0]) / 1e3),
                           ang_diam_arcsec=round(float(em["ang_diam_arcsec"][0]), 1),
                           track_rate_arcsec_s=round(float(em["rate_inertial"][0]), 2),
                           m_per_pixel_4mas=round(float(em["m_per_mas"][0]) * 4, 2)))
    blocks.sort(key=lambda b: b["start_h"])
    fig_timeline(t_dusk, t_dawn, blocks)
    B = {b["block"]: b for b in blocks}

    print("4) simulating MICADO observations ...")
    results = {}
    b = B["Bahryn mosaic Ks"]
    print("   Bahryn mosaic")
    mosaic_tiles, results["mosaic"] = bahryn_mosaic(b["t_mid"], b["alt_deg"], 2 if QUICK else 3)
    b = B["Bahryn ice map"]
    print("   Bahryn ice map")
    ice_obs, results["ice"] = bahryn_ice_map(b["t_mid"], b["alt_deg"])

    gallery = {}
    b = B["Bahryn zoom 1.5mas"]
    gallery["Bahryn, 1.5 mas zoom (Ks)"] = single_tile("Bahryn", b["t_mid"], b["alt_deg"],
                                                        mode="IMG_1.5mas", depth=0.6)
    for name in ("Karina", "Poggle", "Sun Fac"):
        b = B[f"{name} Ks"]
        gallery[f"{name} (Ks)"] = single_tile(name, b["t_mid"], b["alt_deg"])
    b = B["XI full disc H+Ks"]
    gallery["XI full disc (H)"] = single_tile("XI", b["t_mid"], b["alt_deg"], fw2="H",
                                              full_disc=True, t_total=300.0)
    gallery["XI full disc (Ks)"] = single_tile("XI", b["t_mid"], b["alt_deg"], fw2="Ks",
                                               full_disc=True, t_total=300.0)

    fig, axes = plt.subplots(2, len(gallery), figsize=(4.2 * len(gallery), 8.6))
    for col, (title, g) in enumerate(gallery.items()):
        tr = g["render"]["albedo"] * g["render"]["reflect"]
        n = tr.shape[0]
        axes[0, col].imshow(tr, origin="lower", cmap="gray")
        axes[0, col].set_title(f"{title}\nmodel truth", fontsize=9)
        c = g["obs"]["calib"]
        v = np.percentile(c, [1, 99.8])
        axes[1, col].imshow(c, origin="lower", cmap="gray", vmin=v[0], vmax=v[1])
        axes[1, col].set_title(f"MICADO: {g['obs']['pupil']}, DIT {g['obs']['dit']:.1f}s x {g['obs']['ndit']}",
                               fontsize=9)
        for row in (0, 1):
            scale_bar(axes[row, col], g["render"]["pix_m"], n)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("MICADO on Geonosis: moon gallery (top: model surface, bottom: simulated observation)")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "11_moon_gallery.png", dpi=110)
    plt.close(fig)

    # night plan with exposure settings
    expo = {"Bahryn mosaic Ks": mosaic_tiles[len(mosaic_tiles) // 2]["obs"],
            "Bahryn ice map": ice_obs["xK1"],
            "Bahryn zoom 1.5mas": gallery["Bahryn, 1.5 mas zoom (Ks)"]["obs"],
            "Karina Ks": gallery["Karina (Ks)"]["obs"], "Poggle Ks": gallery["Poggle (Ks)"]["obs"],
            "Sun Fac Ks": gallery["Sun Fac (Ks)"]["obs"], "XI full disc H+Ks": gallery["XI full disc (Ks)"]["obs"]}
    mode_of = {"Bahryn zoom 1.5mas": "IMG_1.5mas"}
    filt_of = {"Bahryn mosaic Ks": "Ks", "Bahryn ice map": "xH1,xH2,xK1,xK2", "XI full disc H+Ks": "H,Ks"}
    for bl in blocks:
        o = expo.get(bl["block"])
        bl["mode"] = "SCAO + " + mode_of.get(bl["block"], "IMG_4mas")
        bl["filters"] = filt_of.get(bl["block"], "Ks")
        if o is not None:
            bl["nd_filter"], bl["dit_s"], bl["ndit"] = o["pupil"], round(o["dit"], 2), o["ndit"]
        bl.pop("t_mid")
    with open(OUT / "night_plan.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(k for bl in blocks for k in bl))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(blocks)

    summary = dict(night_index=k, t_dusk_days=t_dusk / DAY, night_hours=(t_dawn - t_dusk) / HOUR,
                   sidereal_rate_arcsec_s=OBS.sidereal_rate(), blocks=blocks, results=results,
                   strehl_ks=STREHL_KS, runtime_s=time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
