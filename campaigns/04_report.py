"""Collect all figures, tables and summaries into one self-contained local
HTML report: outputs/report.html (images embedded as base64, no network)."""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return f"<p class='missing'>missing: {html.escape(path.name)}</p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(caption)}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def table(path: Path, max_rows=40) -> str:
    if not path.exists():
        return f"<p class='missing'>missing: {html.escape(path.name)}</p>"
    with open(path) as f:
        rows = list(csv.reader(f))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                   for r in rows[1:max_rows + 1])
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    a = load(OUT / "earth_metis" / "summary.json")
    b = load(OUT / "geonosis_micado" / "summary.json")
    S, E, G = OUT / "system", OUT / "earth_metis", OUT / "geonosis_micado"
    X = OUT / "easter_eggs"
    egg = load(X / "summary.json")
    kpi_x = ""
    if egg:
        kpi_x = (f"<ul class='kpi'><li><b>{egg['death_star']['diameter_arcmin']:.1f}&prime;</b> Death Star in the sky</li>"
                 f"<li><b>{egg['ring_chase']['m_per_pixel'] * 100:.0f} cm</b> per pixel in the rings</li>"
                 f"<li><b>{egg['ring_chase']['track_rate_arcsec_s']:.0f}&Prime;/s</b> ring tracking (fantasy!)</li>"
                 f"<li><b>{egg['hoth_on_bahryn']['m_per_pixel']:.2f} m</b> per pixel on the walkers</li>")
        shark = egg.get("shark_side_of_the_moon")
        if shark:
            kpi_x += (f"<li><b>{shark['shine_vs_sunlight_ks']:.1e}</b> Geonosis-shine on the shark base</li>"
                      f"<li><b>{shark['flag_km'][0]:.1f} km</b> shark flag</li>")
        kpi_x += "</ul>"

    kpi_a = ""
    if a:
        n2 = a["bands"]["N2"]
        kpi_a = (f"<ul class='kpi'><li><b>{a['separation_arcsec']:.2f}&Prime;</b> separation</li>"
                 f"<li><b>{n2['contrast']:.1e}</b> N2 contrast</li>"
                 f"<li><b>S/N {n2['snr_pca']:.1f}</b> N2 PCA-ADI, 3 nights</li>"
                 f"<li><b>S/N {a['bands']['Lp']['snr_pca']:.1f}</b> in L&prime;</li></ul>")
    kpi_b = ""
    if b:
        m = b["results"].get("mosaic", {})
        ice = b["results"].get("ice", {})
        kpi_b = (f"<ul class='kpi'><li><b>{b['night_hours']:.1f} h</b> Geonosian night</li>"
                 f"<li><b>{m.get('pix_m', 0):.1f} m</b> per pixel on Bahryn</li>"
                 f"<li><b>{len(b['blocks'])}</b> science blocks</li>"
                 f"<li><b>&rho; = {ice.get('spearman_2p0um_vs_truth', 0):.2f}</b> 2 &micro;m ice map vs truth</li></ul>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geonosis with the ELT</title>
<style>
:root {{ --bg:#fbf8f3; --fg:#2a2118; --muted:#7a6a58; --accent:#b4572a; --card:#fff; --line:#e6dccd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16120e; --fg:#efe6da; --muted:#b3a28d;
  --accent:#e08a55; --card:#211b15; --line:#3a3027; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,sans-serif; margin:0; }}
main {{ max-width:1100px; margin:auto; padding:24px 16px 64px; }}
h1 {{ font-size:2rem; margin:.2em 0; }} h2 {{ color:var(--accent); margin-top:2.2em; }}
.lede {{ color:var(--muted); }}
figure {{ margin:18px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ color:var(--muted); font-size:.9rem; margin-top:6px; }}
.tbl {{ overflow-x:auto; }} table {{ border-collapse:collapse; font-size:.85rem; }}
th,td {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; white-space:nowrap; }}
.kpi {{ display:flex; flex-wrap:wrap; gap:10px; list-style:none; padding:0; }}
.kpi li {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }}
.kpi b {{ color:var(--accent); font-size:1.15rem; }}
.missing {{ color:#c33; }}
</style></head><body><main>
<h1>Imaging Geonosis with the ELT</h1>
<p class="lede">A physically consistent model of the Star Wars Geonosis system, imaged with simulated
ELT/METIS from Earth (star Ea placed at 1.3&nbsp;pc) and with ELT/MICADO from the surface of
Geonosis itself. Simulations use ScopeSim with the IRDB instrument packages, plus a Fourier-optics vortex
coronagraph model. Parameters are tagged CANON, DERIVED or INVENTED.</p>

<h2>1. The system</h2>
{table(S / "system_parameters.csv")}
{img(S / "01_postcard.png", "Geonosis and its rocky ring (model render, including the planet's shadow on the ring)")}
{img(S / "02_orbits.png", "Ring and the 15 moon orbits; dotted lines: synchronous orbit and Hill-stability limits")}
{img(S / "03_moon_portraits.png", "The four major moons (procedural surfaces)")}
{img(S / "04_sed.png", "Spectral energy distribution of Ea and the Geonosis components as seen from Earth")}
{table(S / "moons.csv")}

<h2>2. Campaign A: METIS from Earth</h2>
{kpi_a}
{img(E / "01_epoch_selection.png", "Epoch selection: maximum elongation with Ea high at night")}
{img(E / "02_direct_scopesim_N2.png", "ScopeSim METIS N2 frame without coronagraph")}
{img(E / "03_coronagraph_model.png", "Ring-apodized vortex coronagraph model on the ELT pupil")}
{img(E / "04_adi_results.png", "ADI-processed images, 3 nights (green: Geonosis)")}
{img(E / "05_contrast_curves.png", "5-sigma contrast curves vs. Geonosis' contrast")}
{img(E / "06_moons_unresolved.png", "Why the moons cannot be seen from Earth: the whole system is about 2.5 mas across")}
{table(E / "observing_blocks.csv")}

<h2>3. Campaign B: MICADO on Geonosis</h2>
{kpi_b}
{img(G / "07_sky_chart.png", "The sky above the Petranaki ELT: the ring arc and the moons")}
{img(G / "08_night_plan.png", "Night plan")}
{table(G / "night_plan.csv")}
{img(G / "09_bahryn_mosaic.png", "Bahryn: context, model truth and the MICADO Ks mosaic")}
{img(G / "10_bahryn_ice_map.png", "Water-ice band-depth maps from MICADO narrow-band filters")}
{img(G / "11_moon_gallery.png", "Moon gallery: model surface (top) vs simulated MICADO (bottom)")}
{table(G / "moon_ephemerides.csv")}

<h2>4. Easter eggs</h2>
<p class="lede">Hidden in the simulations and observed in the free part of the Geonosian night.
They go through the same renderer and ScopeSim MICADO pipeline as the science targets.
The MICADO FITS headers hide a few more.</p>
{kpi_x}
{img(X / "15_sky_chart_easter_eggs.png", "The sky with the easter eggs: the Death Star and the ring chase")}
{img(X / "12_death_star.png", "That's no moon: DS-1 under construction above Geonosis, with a Star Destroyer passing the superlaser lens")}
{img(X / "13_ring_chase.png", "Attack of the Clones in the rings: Slave I, Obi-Wan's Delta-7 and a seismic charge")}
{img(X / "14_hoth_on_bahryn.png", "Hoth on Bahryn: AT-AT walkers and a Viper probe droid in the science zoom tile")}
{img(X / "16_shark_side_of_the_moon.png", "Shark Side of the Moon: the communist sharks' base and their hammerhead-and-sickle flag on the night side of crescent Poggle, lit by Geonosis-shine")}
</main></body></html>"""
    (OUT / "report.html").write_text(doc)
    print(f"wrote {OUT / 'report.html'} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
