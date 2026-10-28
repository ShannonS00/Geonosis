"""Part 1: build the Geonosis system and document it.

Writes to outputs/system/:
  system_parameters.csv / .md   canon / derived / invented parameters
  moons.csv                     the 15 moons
  stability.csv                 dynamical sanity checks
  01_postcard.png               Geonosis + ring rendered from space
  02_orbits.png                 top-down map of the ring and moon orbits
  03_moon_portraits.png         the four major moons
  04_sed.png                    spectra of Ea and the Geonosis components
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Circle                              # noqa: E402

from geonosis.system import GeonosisSystem, DAY, HOUR              # noqa: E402
from geonosis.render import (View, render_moon, render_planet_and_ring,  # noqa: E402
                             tidally_locked_frame)
from geonosis.textures import MoonSurface                          # noqa: E402
from geonosis import spectra as sp                                 # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "system"
OUT.mkdir(parents=True, exist_ok=True)
SYS = GeonosisSystem()


def tables():
    rows = SYS.summary_table()
    with open(OUT / "system_parameters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "provenance"])
        w.writerows(rows)
    md = ["| parameter | value | provenance |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    (OUT / "system_parameters.md").write_text("\n".join(md) + "\n")

    with open(OUT / "moons.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["designation", "name", "radius_km", "surface", "albedo", "density_kg_m3",
                    "a_km", "e", "inc_deg", "period_h", "major", "provenance"])
        for m in SYS.moons:
            w.writerow([m.designation, m.name, m.radius / 1e3, m.surface, m.albedo, m.density,
                        m.a / 1e3, m.e, m.inc_deg, round(m.orbit.period / HOUR, 1), m.major,
                        m.provenance])

    checks = SYS.stability_report()
    with open(OUT / "stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    n_ok = sum(c["ok"] for c in checks)
    print(f"stability checks passed: {n_ok}/{len(checks)}")
    return checks


def postcard():
    t = 40 * DAY
    s = SYS.sun_direction_eq(t)
    # view from slightly above the ring plane, sun from the side (crescent-ish)
    side = np.cross(s, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    view = 0.55 * s + 0.75 * side + np.array([0, 0, 0.36])
    view /= np.linalg.norm(view)
    rgb = render_planet_and_ring(SYS, t, view, npix=1100, fov_rp=6.6)
    rng = np.random.default_rng(3)
    stars = np.zeros(rgb.shape[:2])
    idx = rng.integers(0, rgb.shape[0], size=(700, 2))
    stars[idx[:, 0], idx[:, 1]] = rng.uniform(0.2, 0.9, 700)
    empty = rgb.sum(axis=2) < 1e-3
    rgb[empty] = stars[empty][:, None]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="lower")
    ax.set_axis_off()
    ax.set_title("Geonosis and its rocky ring (rendered model)", color="w")
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(OUT / "01_postcard.png", dpi=120, facecolor="black")
    plt.close(fig)


def orbit_map():
    Rp = SYS.planet.radius
    K = 1e6                                       # plot unit: 1000 km (positions are in m)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, lim, title in ((axes[0], 650e6, "All 15 moons"), (axes[1], 130e6, "Inner system")):
        ax.set_facecolor("#0b0b1a")
        ax.add_patch(Circle((0, 0), Rp / K, color="#c8743c", zorder=3))
        for rr in np.linspace(SYS.ring.inner, SYS.ring.outer, 12):
            ax.add_patch(Circle((0, 0), rr * Rp / K, fill=False, color="#d9a066", lw=0.6, alpha=0.6))
        ph = np.linspace(0, 2 * np.pi, 400)
        for m in SYS.moons:
            tt = np.linspace(0, m.orbit.period, 400)
            p = m.orbit.position(tt)
            ls = "--" if m.inc_deg > 90 else "-"
            ax.plot(p[0] / K, p[1] / K, ls, lw=0.8,
                    color="#bfe3ff" if m.surface == "ice" else ("#e08a55" if m.surface == "rust" else "0.6"))
            p0 = m.orbit.position(0.0)
            if abs(p0[0]) < lim and abs(p0[1]) < lim:
                ax.plot(p0[0] / K, p0[1] / K, "o", ms=7 if m.major else 3, color="w", zorder=4)
                ax.annotate(m.label, (p0[0] / K, p0[1] / K), color="w", fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        sync = SYS.synchronous_orbit
        ax.plot(sync / K * np.cos(ph), sync / K * np.sin(ph), ":", color="lime", lw=0.8,
                label=f"synchronous orbit ({sync / 1e3:,.0f} km)")
        for frac, col, lab in ((0.49, "r", "prograde stability limit (0.49 r_H)"),
                               (0.70, "orange", "retrograde stability limit (0.70 r_H)")):
            r = frac * SYS.hill_radius / K
            ax.plot(r * np.cos(ph), r * np.sin(ph), ":", color=col, lw=0.8, label=lab)
        ax.set_xlim(-lim / K, lim / K)
        ax.set_ylim(-lim / K, lim / K)
        ax.set_aspect("equal")
        ax.set_xlabel("x [1000 km]")
        ax.set_ylabel("y [1000 km]")
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Geonosis satellite system (equatorial plane, t = 0; dashed = retrograde)")
    fig.tight_layout()
    fig.savefig(OUT / "02_orbits.png", dpi=120)
    plt.close(fig)


def portraits():
    t = 10 * DAY
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, m in zip(axes, SYS.major_moons):
        r = SYS.moon_position_eq(m, t)
        obs = -r / np.linalg.norm(r)                      # seen from Geonosis
        side = np.cross(obs, [0, 0, 1.0])
        side /= np.linalg.norm(side)
        sun = 0.5 * obs + 0.85 * side                      # ~60 deg phase for drama
        sun /= np.linalg.norm(sun)
        view = View(distance=np.linalg.norm(r), to_observer=obs, to_sun=sun,
                    up_hint=np.array([0, 0, 1.0]), body_axes=tidally_locked_frame(SYS, m, t))
        n = 600
        ang = 2 * m.radius / view.distance * 206264.806
        out = render_moon(MoonSurface(m), view, n, n, 1.08 * ang / n)
        img = out["albedo"] * out["reflect"]
        if m.surface == "rust":
            rgb = np.dstack([img * 1.0, img * 0.62, img * 0.42])
        elif m.surface == "ice":
            rgb = np.dstack([img * 0.85, img * 0.93, img * 1.0])
        else:
            rgb = np.dstack([img] * 3)
        ax.imshow(np.clip(rgb / np.percentile(img, 99.5), 0, 1), origin="lower")
        ax.set_title(f"{m.label}: R = {m.radius / 1e3:.0f} km, {m.surface}, albedo {m.albedo}\n"
                     f"a = {m.a / 1e3:,.0f} km ({m.provenance.split(';')[0]})", fontsize=9)
        ax.set_axis_off()
    fig.suptitle("The four major moons (rendered models, ~60 deg phase)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "03_moon_portraits.png", dpi=120)
    plt.close(fig)


def sed():
    t = 64 * DAY
    c = sp.planet_components(SYS, t)
    w = sp.WAVE_UM
    fig, ax = plt.subplots(figsize=(10, 5.5))
    lam_flux = lambda f: f * w                      # nu F_nu-like (photon flux x lambda)
    ax.loglog(w, lam_flux(c["star"]), color="gold", label="Ea (G2V, 1.3 pc)")
    for key, col, lab in (("planet_reflected", "tab:orange", "Geonosis reflected"),
                          ("planet_thermal", "tab:red", "Geonosis thermal"),
                          ("ring_reflected", "tan", "ring reflected"),
                          ("ring_thermal", "brown", "ring thermal"),
                          ("moons_reflected", "tab:blue", "15 moons reflected")):
        ax.loglog(w, lam_flux(c[key]), color=col, label=lab)
    for name, lo, hi in (("Ks", 2.0, 2.3), ("L'", 3.54, 4.1), ("M'", 4.5, 5.0), ("N2", 10.0, 12.4)):
        ax.axvspan(lo, hi, color="0.9")
        ax.text((lo * hi) ** 0.5, 1e-3, name, ha="center", fontsize=9)
    ax.set_ylim(1e-4, 1e9)
    ax.set_xlim(0.4, 25)
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("photon flux x lambda [arb. units]")
    ax.set_title(f"What Earth receives from the Geonosis system (phase angle "
                 f"{np.degrees(c['geometry']['phase_angle']):.0f} deg)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "04_sed.png", dpi=120)
    plt.close(fig)


def main():
    tables()
    postcard()
    orbit_map()
    portraits()
    sed()
    for row in SYS.summary_table():
        print(f"  {row[0]:24s} {row[1]:50s} [{row[2]}]")


if __name__ == "__main__":
    main()
