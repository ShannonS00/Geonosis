"""Campaign A: imaging Geonosis from Earth with ELT/METIS (Ea at 1.3 pc).

Steps
  1. Pick the epoch: maximum elongation while Ea is high at night (Armazones).
  2. Calibrate photometry with ScopeSim/IRDB (METIS img_n N2, img_lm L'/M'):
     star + companion count rates, thermal background, detector constants.
     Also make a direct (non-coronagraphic) ScopeSim frame for comparison.
  3. Build pupil-tracking ADI cubes with the RAVC Fourier-optics model.
  4. Reduce with classical ADI and PCA-ADI; S/N, contrast curves.
  5. Show that the ring and moons are unresolved.
Outputs go to outputs/earth_metis/.
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
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Ellipse, Circle                   # noqa: E402
from scipy import ndimage                                         # noqa: E402
from astropy.io import fits                                       # noqa: E402

from astropy.utils import iers                                    # noqa: E402
iers.conf.auto_download = False          # stay fully local / offline
iers.conf.auto_max_age = None
import astropy.units as u                                         # noqa: E402
from astropy.time import Time                                     # noqa: E402
from astropy.coordinates import (EarthLocation, SkyCoord, AltAz,  # noqa: E402
                                 get_sun)

from geonosis.system import GeonosisSystem, DAY, RAD2MAS, PC      # noqa: E402
from geonosis import spectra as sp                                # noqa: E402
from geonosis import scopesim_utils as su                         # noqa: E402
from geonosis import hci                                          # noqa: E402
from scopesim import Source                                       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "earth_metis"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv

ARMAZONES = EarthLocation(lat=-24.5892 * u.deg, lon=-70.1918 * u.deg, height=3046 * u.m)
SYS = GeonosisSystem()
EPOCH = Time(SYS.epoch_iso, scale="utc")
TARGET = SkyCoord(SYS.star.ra_deg * u.deg, SYS.star.dec_deg * u.deg)

BANDS = {
    #       mode      filter  lambda_c  rel. bandwidth  pixel [mas]
    "N2": dict(mode="img_n", lam=11.20e-6, bw=0.19, pix=6.79),
    "Lp": dict(mode="img_lm", lam=3.82e-6, bw=0.15, pix=5.47),
    "Mp": dict(mode="img_lm", lam=4.76e-6, bw=0.11, pix=5.47),
}


def t_of(time_: Time) -> float:
    return (time_ - EPOCH).to_value(u.s)


# --------------------------------------------------------------------------
# 1. Epoch selection
# --------------------------------------------------------------------------
def night_samples(date: Time, step_min=6):
    """Times through the local night (UT ~ 23h..11h covers Chilean nights)."""
    start = Time(date.datetime.strftime("%Y-%m-%dT22:00:00"), scale="utc")
    return start + np.arange(0, 14 * 60, step_min) * u.min


def plan_epoch():
    days = EPOCH + np.arange(0, 2 * 365) * u.day
    t = np.array([t_of(d) for d in days])
    sep = np.array([sp.earth_view_geometry(SYS, ti)["separation_arcsec"] for ti in t])
    hours = np.zeros(len(days))
    for k, d in enumerate(days[::3]):                       # every 3rd day is enough
        ts = night_samples(d)
        frame = AltAz(obstime=ts, location=ARMAZONES)
        sun_alt = get_sun(ts).transform_to(frame).alt.deg
        tgt_alt = TARGET.transform_to(frame).alt.deg
        hours[3 * k:3 * k + 3] = np.sum((sun_alt < -18) & (tgt_alt > 50)) * 0.1
    score = sep * np.minimum(hours, 4.0)
    best = int(np.argmax(score))
    nights = [days[best + k] for k in (-1, 0, 1)]
    return dict(days=days, sep=sep, hours=hours, best=best, nights=nights)


def night_sequence(date: Time, frame_min, min_alt=50.0):
    """Frame mid-times, parallactic angles and airmass for one night."""
    ts = night_samples(date, step_min=1)
    frame = AltAz(obstime=ts, location=ARMAZONES)
    sun_alt = get_sun(ts).transform_to(frame).alt.deg
    aa = TARGET.transform_to(frame)
    ok = (sun_alt < -18) & (aa.alt.deg > min_alt)
    ts, aa = ts[ok], aa[ok]
    ts, aa = ts[::frame_min], aa[::frame_min]
    lst = ts.sidereal_time("apparent", longitude=ARMAZONES.lon)
    ha = (lst - TARGET.ra).wrap_at(180 * u.deg).rad
    lat, dec = ARMAZONES.lat.rad, TARGET.dec.rad
    pa = np.degrees(np.arctan2(np.sin(ha), np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha)))
    pa = np.unwrap(np.radians(pa))
    return ts, np.degrees(pa), 1 / np.cos(np.radians(90 - aa.alt.deg))


# --------------------------------------------------------------------------
# 2. ScopeSim photometric calibration
# --------------------------------------------------------------------------
def calibrate(band, t_mid):
    b = BANDS[band]
    opt = su.metis_train(b["mode"], band)
    det = su.detector_params(opt)
    star = Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[sp.star_spectrum(SYS)])
    comp = Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[sp.companion_spectrum(SYS, t_mid)])
    a = su.rate_map(opt, star)
    bg = float(np.median(a[:200, :200]))
    star_rate = float(a.sum() - bg * a.size)
    c = su.rate_map(opt, comp)
    bgc = float(np.median(c[:200, :200]))
    comp_rate = float(c.sum() - bgc * c.size)
    res = dict(band=band, star_rate=star_rate, companion_rate=comp_rate,
               contrast=comp_rate / star_rate, background_per_pix=bg, **det)

    # Direct (non-coronagraphic) frame of star + Geonosis at the minimum DIT
    g = sp.earth_view_geometry(SYS, t_mid)
    both = Source(x=[0, g["offset_east_arcsec"] * -1], y=[0, g["offset_north_arcsec"]],
                  ref=[0, 1], weight=[1, 1],
                  spectra=[sp.star_spectrum(SYS), sp.companion_spectrum(SYS, t_mid)])
    opt.cmds["!OBS.auto_exposure.fill_frac"] = 0.75
    for eff in opt.optics_manager.all_effects:
        if type(eff).__name__ == "AutoExposure":
            eff.include = False
    opt.observe(both, update=True)
    # dit/ndit must be passed to readout(): METIS mode properties override cmds
    hdul = opt.readout(dit=det["mindit"], ndit=1)[0]
    raw = hdul[1].data.astype(float)
    res["direct_frame_saturated"] = bool(raw.max() * det["gain"] >= 0.95 * det["full_well"])
    return res, raw


# --------------------------------------------------------------------------
# 3. ADI cube simulation
# --------------------------------------------------------------------------
def simulate_band(band, cal, nights, frame_min, cor, rng):
    b = BANDS[band]
    pix = b["pix"]
    lams = b["lam"] * (1 + b["bw"] * np.array([-1 / 3, 0, 1 / 3]))
    lod_mas = b["lam"] / hci.D_ELT * RAD2MAS
    # field of view: planet separation + 5 lambda/D margin on each side
    sep0 = sp.earth_view_geometry(SYS, t_of(nights[1]))["separation_arcsec"] * 1e3
    npix = int(2 * ((sep0 + 5 * lod_mas) / pix) // 2 * 2 + 1)
    wf = hci.WavefrontModel(cor.n_pup, cor.dx, seed=7)
    zero = np.zeros((cor.n_pup, cor.n_pup))

    # static ingredients per wavelength
    halos, strehls, noncoro_peak = [], [], []
    for lam in lams:
        h, s_ = hci.smooth_ao_halo(cor, wf, lam, pix, npix, n_real=6 if QUICK else 20)
        halos.append(h)
        strehls.append(s_)
        noncoro_peak.append(cor.image(zero, lam, pix, npix, coronagraph=False).max())
    halo = np.mean(halos, axis=0)
    strehl = float(np.mean(strehls))
    peak_ref = float(np.mean(noncoro_peak))          # unocculted star peak fraction

    # off-axis planet PSF (through the coronagraph) at its separation
    t_mid = t_of(nights[1])
    g = sp.earth_view_geometry(SYS, t_mid)
    sep_mas = g["separation_arcsec"] * 1e3
    psf_off = np.mean([cor.image(zero, lam, pix, npix, coronagraph=True,
                                 tilt_mas=(sep_mas, 0.0), center_mas=(sep_mas, 0.0))
                       for lam in lams], axis=0) * strehl
    psf_off_unocc = np.mean([cor.image(zero, lam, pix, npix, coronagraph=False)
                             for lam in lams], axis=0) * strehl
    ap_r = 0.5 * lod_mas / pix * 1.0                  # aperture radius = FWHM/2
    c0 = npix // 2
    psf_flux_ap = hci.aperture_sum(psf_off, c0, c0, ap_r)
    throughput_coro = psf_off.max() / psf_off_unocc.max()

    # DIT: fill 75 % of full well in the brightest pixel beyond 2 lambda/D
    yy, xx = np.indices((npix, npix))
    rr = np.hypot(xx - c0, yy - c0) * pix
    cube_rates_test = cal["star_rate"] * (halo + strehl * cor.image(
        2 * np.pi * wf.opd() / b["lam"], b["lam"], pix, npix))
    peak_rate = cube_rates_test[rr > 2 * lod_mas].max() + cal["background_per_pix"] + cal["dark_current"]
    dit = max(cal["mindit"], 0.75 * cal["full_well"] / peak_rate)
    t_frame = frame_min * 60.0
    ndit = int(round(t_frame / dit))

    cubes, pas, times, truth = [], [], [], []
    for night in nights:
        ts, pa, am = night_sequence(night, frame_min)
        if QUICK:
            ts, pa, am = ts[::3], pa[::3], am[::3]
        frames = []
        for k, (tk, pak) in enumerate(zip(ts, pa)):
            wf.step(frame_min if k else 60.0)            # decorrelate between nights
            star_img = np.mean([cor.image(2 * np.pi * wf.opd() / lam, lam, pix, npix)
                                for lam in lams], axis=0) * strehl + halo
            gk = sp.earth_view_geometry(SYS, t_of(tk))
            # sky offset (east, north) -> detector, rotating with parallactic angle
            # (east is to the left on the sky: x_det = -east before rotation)
            th = np.radians(pak)
            ex, ny_ = -gk["offset_east_arcsec"] * 1e3, gk["offset_north_arcsec"] * 1e3
            xd = (ex * np.cos(th) - ny_ * np.sin(th)) / pix
            yd = (ex * np.sin(th) + ny_ * np.cos(th)) / pix
            # psf_off was computed for a source at (sep, 0) but sampled on a
            # grid centred on it (MFT centre_mas), so a plain shift places it
            planet = hci.shift_psf(psf_off, xd, yd)
            expected = ((cal["star_rate"] * star_img + cal["companion_rate"] * planet
                         + cal["background_per_pix"] + cal["dark_current"]) * t_frame)
            sigma = np.sqrt(expected + ndit * cal["readout_noise"] ** 2)
            frame = expected + rng.normal(size=expected.shape) * sigma
            # sky + dark subtraction (perfect background model, as from chopping/
            # sky frames): remove the constant pedestal
            frame -= (cal["background_per_pix"] + cal["dark_current"]) * t_frame
            frames.append(frame)
            truth.append((xd, yd))
        cubes.append(np.array(frames))
        pas.append(pa)
        times.append(ts)
    meta = dict(lod_mas=lod_mas, pix=pix, npix=npix, strehl=strehl, peak_ref=peak_ref,
                psf_flux_ap=psf_flux_ap, ap_r=ap_r, dit=dit, ndit=ndit,
                t_frame=t_frame, throughput_coro=throughput_coro,
                sep_mas=sep_mas, psf_off=psf_off, halo=halo)
    return cubes, pas, times, meta


# --------------------------------------------------------------------------
# 4. Reduction
# --------------------------------------------------------------------------
def reduce_band(band, cubes, pas, meta, cal):
    npix, pix, lod = meta["npix"], meta["pix"], meta["lod_mas"]
    c0 = npix // 2
    fwhm = lod / pix
    inner = int(2 * fwhm)
    results = {}
    finals = {"cADI": [], "PCA": []}
    for cube, pa in zip(cubes, pas):
        finals["cADI"].append(hci.cadi(cube, pa))
        finals["PCA"].append(hci.pca_adi(cube, pa, ncomp=min(8, len(cube) - 2),
                                         mask_inner_pix=inner))
    # Planet position in the derotated frame (sky orientation at PA = 0)
    t_mid_geom = sp.earth_view_geometry(SYS, t_of(TIMES_MID[0]))
    xs = c0 - t_mid_geom["offset_east_arcsec"] * 1e3 / pix
    ys = c0 + t_mid_geom["offset_north_arcsec"] * 1e3 / pix

    # algorithm throughput from noiseless fake-planet cubes (linear for cADI;
    # for PCA we reuse the same frames' basis implicitly via a fake injection)
    radii = np.arange(int(2 * fwhm), int(0.45 * npix), max(1, int(fwhm / 2)))
    for name, imgs in finals.items():
        img = np.mean(imgs, axis=0)
        snr, flux = hci.snr_mawet(img, xs, ys, fwhm)
        thr = algorithm_throughput(name, cubes[1], pas[1], meta, inner, radii)
        # contrast normalisation: a companion of contrast c gives an aperture
        # signal c * star_rate * psf_flux_ap * t_frame (psf_off already
        # includes the coronagraph's off-axis throughput)
        star_ap = cal["star_rate"] * meta["psf_flux_ap"] * meta["t_frame"]
        cc = hci.contrast_curve(img, fwhm, star_ap, lambda r: np.interp(r, radii, thr), radii)
        per_night = [hci.snr_mawet(im, xs, ys, fwhm)[0] for im in imgs]
        results[name] = dict(image=img, snr=float(snr), snr_per_night=[float(x) for x in per_night],
                             radii_mas=radii * pix, contrast5=cc, throughput=thr,
                             planet_xy=(xs, ys))
    return results


def algorithm_throughput(name, cube, pa, meta, inner, radii):
    """Inject noiseless fake planets in an empty cube, process, measure."""
    npix, pix = meta["npix"], meta["pix"]
    c0 = npix // 2
    fwhm = meta["lod_mas"] / pix
    out = []
    psf = meta["psf_off"]                 # centred on the array (see simulate_band)
    inj = hci.aperture_sum(psf, c0, c0, fwhm / 2)
    for r in radii:
        fake = np.zeros_like(cube)
        for k, a in enumerate(pa):
            th = np.radians(a + 200.0)                   # away from the real planet
            fake[k] = hci.shift_psf(psf, r * np.cos(th), r * np.sin(th))
        if name == "cADI":
            res = hci.cadi(fake + 0 * cube, pa)
        else:
            # project with the science cube's PCA basis (planet too faint to bias it)
            res = pca_with_basis(cube, fake, pa, ncomp=min(8, len(cube) - 2), inner=inner)
        x = c0 + r * np.cos(np.radians(200.0))
        y = c0 + r * np.sin(np.radians(200.0))
        out.append(max(hci.aperture_sum(res, x, y, fwhm / 2) / inj, 1e-3))
    return np.array(out)


def pca_with_basis(sci, fake, pa, ncomp, inner):
    nf, ny, nx = sci.shape
    yy, xx = np.indices((ny, nx))
    sel = np.hypot(xx - nx // 2, yy - ny // 2) >= inner
    M = sci[:, sel] - sci[:, sel].mean(axis=0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt[:ncomp]
    F = fake[:, sel] - fake[:, sel].mean(axis=0)
    R = F - (F @ V.T) @ V
    res = np.zeros_like(fake)
    res[:, sel] = R
    return np.mean(hci.rotate_frames(res, -np.asarray(pa)), axis=0)


# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
def fig_epoch(ep):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    dd = ep["days"].datetime
    ax1.plot(dd, ep["sep"], color="tab:orange", label="Ea-Geonosis separation")
    ax1.set_ylabel("separation [arcsec]")
    ax2 = ax1.twinx()
    ax2.plot(dd, ep["hours"], color="tab:blue", alpha=0.5, label="hours with alt>50 deg at night")
    ax2.set_ylabel("usable hours / night")
    for n in ep["nights"]:
        ax1.axvline(n.datetime, color="k", lw=0.8, ls="--")
    ax1.set_title("When to observe: maximum elongation while Ea is high at night (Armazones)")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "01_epoch_selection.png", dpi=130)
    plt.close(fig)


def fig_direct(raw, band, cal):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    n = raw.shape[0] // 2
    w = 160
    im = ax.imshow(raw[n - w:n + w, n - w:n + w], origin="lower", cmap="magma",
                   norm=matplotlib.colors.LogNorm())
    ax.set_title(f"ScopeSim METIS {band}, no coronagraph, DIT = {cal['mindit']} s\n"
                 f"star saturated: {cal['direct_frame_saturated']} - Geonosis invisible")
    plt.colorbar(im, ax=ax, label="ADU")
    fig.tight_layout()
    fig.savefig(OUT / f"02_direct_scopesim_{band}.png", dpi=130)
    plt.close(fig)


def fig_coronagraph(cor, meta_n2):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].imshow(cor.pupil * cor.apodizer, origin="lower", cmap="gray")
    ax[0].set_title(f"ELT pupil x RAVC apodizer\nR1 = {cor.ring_radius:.2f}, t_out = {cor.ring_t:.2f}")
    ax[1].imshow(cor.lyot, origin="lower", cmap="gray")
    ax[1].set_title("Lyot stop")
    npix, pix = meta_n2["npix"], meta_n2["pix"]
    ext = np.array([-1, 1, -1, 1]) * npix / 2 * pix / 1e3
    im = ax[2].imshow(np.log10(meta_n2["halo"] / meta_n2["peak_ref"] + 1e-10), origin="lower",
                      extent=ext, cmap="inferno", vmin=-8, vmax=-4)
    ax[2].set_title("N2 smooth AO halo after RAVC\n(log10 raw contrast)")
    ax[2].set_xlabel("arcsec")
    plt.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(OUT / "03_coronagraph_model.png", dpi=130)
    plt.close(fig)


def fig_adi(all_res, metas):
    fig, axes = plt.subplots(2, len(all_res), figsize=(5.2 * len(all_res), 10))
    axes = np.atleast_2d(axes)
    for col, (band, res) in enumerate(all_res.items()):
        m = metas[band]
        ext = np.array([-1, 1, -1, 1]) * m["npix"] / 2 * m["pix"] / 1e3
        for row, name in enumerate(("cADI", "PCA")):
            a = axes[row, col]
            img = res[name]["image"].copy()
            xs, ys = res[name]["planet_xy"]
            # mask the star-dominated core and stretch to the noise at the
            # planet's separation, so a 5-sigma source stands out
            yy, xx = np.indices(img.shape)
            c0 = m["npix"] // 2
            rr = np.hypot(xx - c0, yy - c0)
            rp = np.hypot(xs - c0, ys - c0)
            fwhm = m["lod_mas"] / m["pix"]
            ann = (np.abs(rr - rp) < 2 * fwhm) & (np.hypot(xx - xs, yy - ys) > 2 * fwhm)
            v = 4 * np.std(ndimage.gaussian_filter(img, fwhm / 2.355)[ann]) or 1.0
            img = ndimage.gaussian_filter(img, fwhm / 2.355)       # matched to the PSF core
            img[rr < 2 * fwhm] = np.nan
            a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
            px, py = (xs - m["npix"] // 2) * m["pix"] / 1e3, (ys - m["npix"] // 2) * m["pix"] / 1e3
            a.add_patch(Circle((px, py), 2.5 * m["lod_mas"] / 1e3, fill=False, color="lime", lw=1.2))
            a.plot(0, 0, "k+")
            a.set_title(f"{band} {name}: S/N = {res[name]['snr']:.1f}")
            a.set_xlabel("dRA [arcsec]  (E left)")
            a.invert_xaxis()
    fig.suptitle("METIS RAVC + ADI, 3 nights (PSF-smoothed, inner 2 $\\lambda$/D masked, "
                 "colour scale +-4 sigma at the planet's separation): Geonosis in the green circle")
    fig.tight_layout()
    fig.savefig(OUT / "04_adi_results.png", dpi=130)
    plt.close(fig)


def fig_contrast(all_res, cals, sep_arcsec):
    fig, ax = plt.subplots(figsize=(8, 5))
    for band, res in all_res.items():
        r = res["PCA"]
        ax.semilogy(r["radii_mas"] / 1e3, r["contrast5"], label=f"{band} PCA-ADI 5$\\sigma$")
        ax.plot(sep_arcsec, cals[band]["contrast"], "*", ms=14,
                label=f"Geonosis+ring in {band} ({cals[band]['contrast']:.1e})")
    ax.set_xlabel("separation [arcsec]")
    ax.set_ylabel("contrast")
    ax.set_title("5$\\sigma$ detection limits vs. Geonosis")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_contrast_curves.png", dpi=130)
    plt.close(fig)


def fig_unresolved(t):
    """True geometry of planet + ring + moons vs METIS/MICADO resolution."""
    d = SYS.star.distance
    R = SYS.orbit_to_sky @ SYS.eq_to_orbit
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, half in zip(ax, (2.6, 70.0)):
        rp = SYS.planet.radius / d * RAD2MAS
        a.add_patch(Circle((0, 0), rp, color="tab:orange"))
        n = R @ np.array([0, 0, 1.0])
        for rr in (SYS.ring.inner, SYS.ring.outer):
            ax_major = rr * rp
            a.add_patch(Ellipse((0, 0), 2 * ax_major, 2 * ax_major * abs(n[2]),
                                angle=np.degrees(np.arctan2(-n[0], n[1])) + 90,
                                fill=False, color="peru"))
        for m in SYS.moons:
            p = R @ SYS.moon_position_eq(m, t)
            a.plot(-p[0] / d * RAD2MAS, p[1] / d * RAD2MAS, "o",
                   ms=6 if m.major else 2.5, color="lightblue" if m.surface == "ice" else "0.5")
            if m.major and half < 10:
                a.annotate(m.name, (-p[0] / d * RAD2MAS, p[1] / d * RAD2MAS), fontsize=8,
                           xytext=(4, 4), textcoords="offset points")
        for lab, lod, col in (("METIS N2 $\\lambda/D$", 11.2e-6 / hci.D_ELT * RAD2MAS, "r"),
                              ("METIS L' $\\lambda/D$", 3.82e-6 / hci.D_ELT * RAD2MAS, "m"),
                              ("MICADO Ks $\\lambda/D$", 2.15e-6 / hci.D_ELT * RAD2MAS, "g")):
            a.add_patch(Circle((0, 0), lod / 2, fill=False, color=col, ls="--", label=lab))
        a.set_xlim(half, -half)
        a.set_ylim(-half, half)
        a.set_aspect("equal")
        a.set_xlabel("dRA [mas]")
        a.set_ylabel("dDec [mas]")
        a.set_facecolor("black")
    ax[0].set_title("True Geonosis system at 1.3 pc (zoom: +-2.6 mas)")
    ax[1].set_title("Same, with ELT diffraction limits (+-70 mas)")
    ax[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "06_moons_unresolved.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
TIMES_MID = []


def main():
    t0 = time.time()
    rng = np.random.default_rng(2031)
    print("1) choosing the epoch ...")
    ep = plan_epoch()
    nights = ep["nights"]
    TIMES_MID.append(nights[1] + 4 * u.hour)
    fig_epoch(ep)
    g = sp.earth_view_geometry(SYS, t_of(nights[1]))
    print(f"   nights: {[n.iso[:10] for n in nights]}, separation {g['separation_arcsec']:.3f}\" "
          f"phase angle {np.degrees(g['phase_angle']):.0f} deg")
    fig_unresolved(t_of(nights[1]))

    print("2) building the coronagraph ...")
    cor = hci.RAVC(n_pup=192, pad=8)

    cals, all_res, metas, ob_rows = {}, {}, {}, []
    frame_min = 2
    for band in BANDS:
        print(f"3) {band}: ScopeSim calibration ...")
        cal, raw = calibrate(band, t_of(nights[1]))
        cals[band] = cal
        if band == "N2":
            fig_direct(raw, band, cal)
        print(f"   star {cal['star_rate']:.2e} ph/s, Geonosis {cal['companion_rate']:.2e} ph/s "
              f"(contrast {cal['contrast']:.1e}), bg {cal['background_per_pix']:.2e} ph/s/px")
        print(f"   {band}: simulating ADI cubes ...")
        cubes, pas, times, meta = simulate_band(band, cal, nights, frame_min, cor, rng)
        metas[band] = meta
        print(f"   {band}: reducing ({sum(len(c) for c in cubes)} frames) ...")
        all_res[band] = reduce_band(band, cubes, pas, meta, cal)
        print(f"   {band}: S/N cADI {all_res[band]['cADI']['snr']:.1f}, "
              f"PCA {all_res[band]['PCA']['snr']:.1f}  [{time.time() - t0:.0f}s]")
        for night, pa, ts in zip(nights, pas, times):
            ob_rows.append(dict(
                night=night.iso[:10], band=band, mode=BANDS[band]["mode"].upper() + " + RAVC",
                start_ut=ts[0].iso[11:16], end_ut=ts[-1].iso[11:16],
                n_frames=len(ts), dit_s=round(meta["dit"], 4), ndit_per_frame=meta["ndit"],
                on_source_h=round(len(ts) * meta["t_frame"] / 3600, 2),
                field_rotation_deg=round(float(np.ptp(pa)), 1)))

    for band, res in all_res.items():                 # keep reduced images
        fits.HDUList([fits.PrimaryHDU()] + [
            fits.ImageHDU(res[n]["image"].astype(np.float32), name=n) for n in ("cADI", "PCA")
        ]).writeto(OUT / f"reduced_{band}.fits", overwrite=True)
    fig_coronagraph(cor, metas["N2"])
    fig_adi(all_res, metas)
    fig_contrast(all_res, cals, g["separation_arcsec"])

    with open(OUT / "observing_blocks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ob_rows[0]))
        w.writeheader()
        w.writerows(ob_rows)

    summary = dict(
        nights=[n.iso[:10] for n in nights],
        separation_arcsec=g["separation_arcsec"], phase_angle_deg=float(np.degrees(g["phase_angle"])),
        ravc=dict(ring_radius=cor.ring_radius, outer_transmission=cor.ring_t),
        bands={b: dict(
            contrast=cals[b]["contrast"], star_rate=cals[b]["star_rate"],
            companion_rate=cals[b]["companion_rate"], background_per_pix=cals[b]["background_per_pix"],
            direct_frame_saturated=cals[b]["direct_frame_saturated"],
            dit=metas[b]["dit"], strehl=metas[b]["strehl"], coronagraph_throughput=metas[b]["throughput_coro"],
            snr_cadi=all_res[b]["cADI"]["snr"], snr_pca=all_res[b]["PCA"]["snr"],
            snr_pca_per_night=all_res[b]["PCA"]["snr_per_night"],
            contrast5_at_planet=float(np.interp(g["separation_arcsec"] * 1e3,
                                                all_res[b]["PCA"]["radii_mas"],
                                                all_res[b]["PCA"]["contrast5"])))
            for b in BANDS},
        runtime_s=time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
