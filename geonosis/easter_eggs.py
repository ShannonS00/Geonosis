"""Star Wars easter eggs, built as physical objects that go through the same
rendering + ScopeSim pipeline as the science targets.

* Props on moons (3-D boxes, ray-cast with cast shadows):
    - three AT-AT walkers and a Viper probe droid on icy Bahryn (a Hoth
      homage), placed inside the campaign's 1.5 mas zoom tile;
* "That's no moon": the first Death Star under construction in orbit of
  Geonosis (canon: Rogue One / Rebels), plus an Imperial Star Destroyer;
* the Attack of the Clones chase: Slave I and Obi-Wan's Delta-7 starfighter
  among the ring boulders, with a seismic charge going off;
* Star Wars quotes hidden in the FITS headers.
Sizes are canon where canon gives them (AT-AT 22.5 m tall, Slave I 21.5 m,
Delta-7 8 m, Star Destroyer 1.6 km, Death Star ~160 km).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

from .system import G, Moon, Orbit
from .textures import value_noise


# ==========================================================================
# 3-D props standing on a moon surface
# ==========================================================================
def _slab(G_, d, lo, hi):
    """Vectorised ray / axis-aligned box intersection (slab method).
    G_: (3, N) ray origins, d: (3,) direction.  Returns t_enter, t_exit,
    exit_axis (N,)."""
    t0 = np.full(G_.shape[1], -np.inf)
    t1 = np.full(G_.shape[1], np.inf)
    exit_axis = np.zeros(G_.shape[1], int)
    for a in range(3):
        if abs(d[a]) < 1e-12:
            inside = (G_[a] >= lo[a]) & (G_[a] <= hi[a])
            t1 = np.where(inside, t1, -np.inf)
            continue
        ta = (lo[a] - G_[a]) / d[a]
        tb = (hi[a] - G_[a]) / d[a]
        tmin, tmax = np.minimum(ta, tb), np.maximum(ta, tb)
        t0 = np.maximum(t0, tmin)
        better = tmax < t1
        exit_axis = np.where(better, a, exit_axis)
        t1 = np.minimum(t1, tmax)
    return t0, t1, exit_axis


@dataclass
class Prop:
    """A group of boxes standing on a moon at (lat, lon), offset (east,
    north) metres from that anchor, rotated by ``heading_deg`` (prop x-axis
    measured from east towards north).  Boxes: (cx, cy, cz, sx, sy, sz) [m]."""

    name: str
    lat_deg: float
    lon_deg: float
    boxes: list
    offset_m: tuple = (0.0, 0.0)
    heading_deg: float = 0.0
    albedo: float = 0.3

    @property
    def height(self):
        return max(b[2] + b[5] / 2 for b in self.boxes)

    @property
    def reach(self):
        return max(np.hypot(abs(b[0]) + b[3] / 2, abs(b[1]) + b[4] / 2) for b in self.boxes)

    def apply(self, out, surface, view, X, Y, Z, h, n_c, cam, lunar_lambert):
        moon = surface.moon
        R = moon.radius
        xc, yc, zc = cam
        B = view.body_axes
        la, lo = np.radians(self.lat_deg), np.radians(self.lon_deg)
        up0 = B @ np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        e0 = np.cross(B[:, 2], up0)
        e0 /= np.linalg.norm(e0)
        n0 = np.cross(up0, e0)
        # ground level under the prop itself (anchor + offset)
        la_p = la + self.offset_m[1] / R
        lo_p = lo + self.offset_m[0] / (R * np.cos(la))
        h0 = surface.evaluate(np.array([la_p]), np.array([lo_p]),
                              r_min=max(2 * out["pix_m"], 1.0))["height"][0]
        hd = np.radians(self.heading_deg)
        ch, sh = np.cos(hd), np.sin(hd)

        def to_prop(e, n, u):                 # ENU -> prop frame
            return np.array([e * ch + n * sh, -e * sh + n * ch, u])

        # direction vectors (unit) in the prop frame
        def dir_prop(v):
            return to_prop(v @ e0, v @ n0, v @ up0)

        d_obs = dir_prop(zc)
        s = dir_prop(view.to_sun)
        if s[2] <= 0:                          # prop on the night side
            return
        # ground points of every pixel (moon-centred eq frame -> ENU -> prop)
        cos_e = np.array([xc @ e0, yc @ e0, zc @ e0])
        cos_n = np.array([xc @ n0, yc @ n0, zc @ n0])
        cos_u = np.array([xc @ up0, yc @ up0, zc @ up0])
        Rh = R + h
        Pe = Rh * (n_c[0] * cos_e[0] + n_c[1] * cos_e[1] + n_c[2] * cos_e[2]) - self.offset_m[0]
        Pn = Rh * (n_c[0] * cos_n[0] + n_c[1] * cos_n[1] + n_c[2] * cos_n[2]) - self.offset_m[1]
        Pu = Rh * (n_c[0] * cos_u[0] + n_c[1] * cos_u[1] + n_c[2] * cos_u[2]) - (R + h0)
        shadow_len = self.height / max(s[2] / np.hypot(s[0], s[1]), 0.02)
        ext = self.reach + min(shadow_len, 3000.0) + 5.0
        near = out["disc"] & (np.hypot(Pe, Pn) < ext)
        if not near.any():
            return
        idx = np.nonzero(near)
        G_ = to_prop(Pe[idx], Pn[idx], Pu[idx])

        # --- what does the observer see along each line of sight?
        t_best = np.full(G_.shape[1], -np.inf)
        normal = np.zeros_like(G_)
        for (cx, cy, cz, sx, sy, sz) in self.boxes:
            lo_ = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2])
            hi_ = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
            t0, t1, ax = _slab(G_, d_obs, lo_, hi_)
            hit = (t1 >= np.maximum(t0, 0)) & (t1 > t_best)
            t_best = np.where(hit, t1, t_best)
            nrm = np.zeros_like(G_)
            nrm[ax, np.arange(G_.shape[1])] = np.sign(d_obs[ax])
            normal = np.where(hit[None], nrm, normal)
        on_prop = np.isfinite(t_best)

        # --- shadows: from a point, does the ray towards the sun hit a box?
        def shadowed(origin):
            sh_ = np.zeros(origin.shape[1], bool)
            for (cx, cy, cz, sx, sy, sz) in self.boxes:
                lo_ = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2])
                hi_ = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
                t0, t1, _ = _slab(origin, s, lo_, hi_)
                sh_ |= t1 >= np.maximum(t0, 1e-3)
            return sh_

        rows, cols = idx
        # ground pixels in the prop's shadow
        ground = ~on_prop
        if ground.any():
            gs = shadowed(G_[:, ground])
            gr, gc = rows[ground][gs], cols[ground][gs]
            out["reflect"][gr, gc] = 0.0
            out["shadow"][gr, gc] = True
        # prop surfaces: Lambert shading of the visible face
        if on_prop.any():
            Q = G_[:, on_prop] + t_best[on_prop] * d_obs[:, None]
            nrm = normal[:, on_prop]
            mu0 = np.clip(s @ nrm, 0, None)
            lit = ~shadowed(Q + 1e-3 * nrm)
            pr, pc = rows[on_prop], cols[on_prop]
            out["reflect"][pr, pc] = mu0 * lit
            out["albedo"][pr, pc] = self.albedo
            out["ice"][pr, pc] = 0.0
        out.setdefault("props", {})[self.name] = (rows[on_prop], cols[on_prop])


def at_at():
    """All Terrain Armored Transport (canon: 22.5 m tall, ~20 m long)."""
    legs = [(x, y, 7.2, 1.3, 1.3, 13.6) for x in (-4.6, 4.6) for y in (-2.3, 2.3)]
    feet = [(x, y, 0.4, 2.6, 2.6, 0.8) for x in (-4.6, 4.6) for y in (-2.3, 2.3)]
    return [(0.0, 0.0, 17.0, 13.5, 5.6, 6.2),       # body
            (8.2, 0.0, 16.6, 3.0, 1.6, 1.6),        # neck
            (11.5, 0.0, 15.8, 5.0, 3.4, 3.4),       # head
            (14.4, 0.0, 15.0, 1.4, 0.4, 0.4)] + legs + feet


def probe_droid():
    """Imperial Viper probe droid hovering over the surface (~2.5 m)."""
    arms = [(0.5 * np.cos(a), 0.5 * np.sin(a), 1.4, 0.12, 0.12, 1.6)
            for a in np.radians([0, 72, 144, 216, 288])]
    return [(0.0, 0.0, 2.7, 1.2, 1.2, 1.0), (0.0, 0.0, 3.35, 0.15, 0.15, 0.9)] + arms


PROPS: dict[str, list[Prop]] = {}


def set_bahryn_anchor(lat_deg, lon_deg):
    PROPS["VI"] = [
        Prop("AT-AT 1", lat_deg, lon_deg, at_at(), offset_m=(-430.0, 10.0), heading_deg=100, albedo=0.35),
        Prop("AT-AT 2", lat_deg, lon_deg, at_at(), offset_m=(-375.0, 75.0), heading_deg=95, albedo=0.35),
        Prop("AT-AT 3", lat_deg, lon_deg, at_at(), offset_m=(-485.0, 105.0), heading_deg=105, albedo=0.35),
        Prop("Viper probe droid", lat_deg, lon_deg, probe_droid(), offset_m=(-250.0, -60.0),
             heading_deg=0, albedo=0.06),
    ]


# Anchor = centre of the campaign's Bahryn 1.5 mas zoom tile (night #43,
# 2.67 h after dusk, depth 0.6 -> sun 24 deg up), computed with
# geonosis.render.  The walkers march across the plain west of the centre.
set_bahryn_anchor(-6.8064, -27.4628)


def props_for(moon) -> list:
    return PROPS.get(getattr(moon, "designation", None), [])


# ==========================================================================
# "That's no moon": the Death Star under construction
# ==========================================================================
def death_star(system, mean_anomaly0_deg=0.0) -> Moon:
    """DS-1 in a low equatorial orbit (radius 80 km; Rogue One-era canon size
    ~120-160 km).  Not one of the 15 moons: it is a space station."""
    m = Moon("DS-1", "That's no moon", 80e3, 0.35, "rock", 300.0, 72_000e3, 0.0,
             0.4, 30.0, 0.0, mean_anomaly0_deg, major=False,
             provenance="CANON: built in orbit of Geonosis; INVENTED orbit")
    m.orbit = Orbit(a=m.a, e=0.0, inc=np.radians(m.inc_deg), node=np.radians(m.node_deg),
                    peri=0.0, mean_anomaly0=np.radians(mean_anomaly0_deg),
                    gm=G * system.planet.mass)
    return m


class DeathStarSurface:
    """Panelled hull, equatorial trench, superlaser dish (lat +30, facing
    Geonosis) and an unfinished southern hemisphere of open framework."""

    DISH_LAT, DISH_LON, DISH_RAD = np.radians(30.0), 0.0, 0.17   # dish angular radius [rad]

    def __init__(self, moon, seed=1977):
        self.moon, self.seed = moon, seed

    def dish_distance(self, lat, lon):
        c = (np.sin(lat) * np.sin(self.DISH_LAT)
             + np.cos(lat) * np.cos(self.DISH_LAT) * np.cos(lon - self.DISH_LON))
        return np.arccos(np.clip(c, -1, 1))

    def evaluate(self, lat, lon, r_min):
        R = self.moon.radius
        u, v = R * lon * np.cos(lat), R * lat
        cell = 1800.0                                   # hull panel size [m]
        iu, iv = np.floor(u / cell), np.floor(v / cell)
        from .textures import _uniform
        panel = _uniform(self.seed, iu, iv)
        fu, fv = (u / cell) % 1, (v / cell) % 1
        seam = (np.minimum(fu, 1 - fu) < 0.03) | (np.minimum(fv, 1 - fv) < 0.03)
        city = value_noise(u, v, 300.0, self.seed + 3, octaves=4)
        albedo = 0.30 + 0.10 * panel + 0.05 * city - 0.12 * seam
        height = 60.0 * panel - 25.0 * seam
        # equatorial trench
        trench = np.abs(v) < 700.0
        height = np.where(trench, -500.0, height)
        albedo = np.where(trench, 0.18, albedo)
        # superlaser dish: spherical bowl + focusing lens + 8 tributary beams
        dd = self.dish_distance(lat, lon)
        x = dd / self.DISH_RAD
        dish = x < 1
        height = np.where(dish, -0.05 * R * (1 - x**2) + 400 * np.exp(-(x / 0.05) ** 2), height)
        az = np.arctan2(np.cos(lat) * np.sin(lon - self.DISH_LON),
                        np.cos(self.DISH_LAT) * np.sin(lat)
                        - np.sin(self.DISH_LAT) * np.cos(lat) * np.cos(lon - self.DISH_LON))
        # eight tributary beams of constant physical width, converging on the lens
        beam_half = np.minimum(0.0035 / np.maximum(x, 1e-3), 0.5)
        beams = dish & (np.abs(((az * 8 / (2 * np.pi)) % 1) - 0.5) > 0.5 - beam_half) & (x > 0.06)
        ribs = dish & (np.abs(np.sin(x * 60 * np.pi)) > 0.92)
        lens = x < 0.05
        albedo = np.where(dish, 0.20 + 0.06 * ribs + 0.20 * beams, albedo)
        albedo = np.where(lens, 0.45 + 0.35 * np.exp(-(x / 0.015) ** 2), albedo)
        albedo = np.where(dish & (x >= 0.05) & (x < 0.065), 0.05, albedo)      # dark ring
        # under construction: south of -25 deg only the framework exists
        open_ = lat < np.radians(-25.0) + 0.05 * np.sin(3 * lon)
        frame = (np.minimum(fu, 1 - fu) < 0.06) | (np.minimum(fv, 1 - fv) < 0.06)
        height = np.where(open_ & ~frame, -3000.0, height)
        albedo = np.where(open_, np.where(frame, 0.45, 0.02), albedo)
        return dict(height=height, albedo=np.clip(albedo, 0.01, 1), ice=np.zeros_like(height))


# ==========================================================================
# 2-D silhouettes (spaceships) and the ring-chase scene
# ==========================================================================
def _ellipse(a, b, cx=0.0, cy=0.0, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


SHIPS = {
    # outlines in metres, x = forward.  Canon lengths: Slave I 21.5 m,
    # Delta-7 Aethersprite 8 m, Imperial-class Star Destroyer 1,600 m.
    "Slave I": [(_ellipse(10.75, 6.5), 0.32), (_ellipse(4.0, 4.0, -5.5, 7.0), 0.30),
                (_ellipse(4.0, 4.0, -5.5, -7.0), 0.30), (_ellipse(1.6, 1.6, 6.5, 0.0), 0.05)],
    "Delta-7": [(np.array([[4.0, 0], [-4.0, 1.95], [-3.0, 0], [-4.0, -1.95]]), 0.60),
                (_ellipse(0.7, 0.5, 0.8, 0.0), 0.05)],
    "Star Destroyer": [(np.array([[800, 0], [-800, 450], [-800, -450]]), 0.40),
                       (np.array([[-500, 60], [-760, 60], [-760, -60], [-500, -60]]), 0.40),
                       (np.array([[-640, 140], [-700, 140], [-700, -140], [-640, -140]]), 0.38)],
}


def paint_ship(canvas_w, pix_m, name, x_m, y_m, heading_deg, sun_cam, two_tone=True, ambient=0.0):
    """Paint a ship silhouette (albedo x I/F) into ``canvas_w`` in place.
    Flat hull facets tilted +-25 deg about the keel give the classic two-tone
    look under side lighting."""
    ny, nx = canvas_w.shape
    hd = np.radians(heading_deg)
    rot = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
    yy, xx = np.mgrid[0:ny, 0:nx]
    px = (xx - nx / 2) * pix_m - x_m
    py = (yy - ny / 2) * pix_m - y_m
    local = np.column_stack([px.ravel(), py.ravel()]) @ rot     # into ship frame
    for poly, alb in SHIPS[name]:
        inside = MplPath(poly).contains_points(local).reshape(ny, nx)
        if not inside.any():
            continue
        side = np.sign(local[:, 1].reshape(ny, nx)) if two_tone else np.ones((ny, nx))
        tilt = np.radians(25.0)
        # facet normals in camera frame: z towards camera, tilted sideways
        nxf = np.sin(tilt) * side * -np.sin(hd)
        nyf = np.sin(tilt) * side * np.cos(hd)
        mu0 = np.clip(sun_cam[0] * nxf + sun_cam[1] * nyf + sun_cam[2] * np.cos(tilt), 0.0, None) + ambient
        canvas_w[inside] = (alb * mu0)[inside]


def render_ring_chase(n, pix_m, sun_cam, tau_slant, seed=66):
    """Ring boulders (power-law sizes 0.4-10 m, albedo 0.1-0.2), Slave I,
    Obi-Wan's Delta-7 and a seismic charge that has just cleared a ring of
    debris.  Returns the weight map (albedo x I/F) and ship pixel boxes."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    size = n * pix_m
    rmin, rmax = 0.4, 10.0
    cover = 1 - np.exp(-tau_slant)
    mean_area = np.pi * 2 * rmin**2 * np.log(rmax / rmin) / (1 - (rmin / rmax) ** 2)
    nb = int(cover * size**2 / mean_area)
    r = (rmin**-2 - rng.uniform(size=nb) * (rmin**-2 - rmax**-2)) ** -0.5
    x = rng.uniform(-size / 2, size / 2, nb)
    y = rng.uniform(-size / 2, size / 2, nb)
    alb = rng.uniform(0.10, 0.20, nb)
    # seismic charge: boulders inside the blast are gone, a dust shell glows in sunlight
    bx, by, br = 0.0, 0.0, 16.0
    ships = [("Slave I", 42.0, 16.0, 20.0, 14.0), ("Delta-7", -42.0, -14.0, 25.0, 6.0)]
    keep = np.hypot(x - bx, y - by) > br + r
    for _, sx_, sy_, _, rad in ships:                 # flying through a gap
        keep &= np.hypot(x - sx_, y - sy_) > rad + r
    x, y, r, alb = x[keep], y[keep], r[keep], alb[keep]
    # unresolved boulders: deposit their (phase-averaged) flux
    small = r / pix_m < 3
    lam = max(sun_cam[2], 0) * 0.5 + 0.25                       # Lambert sphere, ~phase
    ix = np.clip(((x[small] + size / 2) / pix_m).astype(int), 0, n - 1)
    iy = np.clip(((y[small] + size / 2) / pix_m).astype(int), 0, n - 1)
    np.add.at(w, (iy, ix), alb[small] * lam * np.pi * (r[small] / pix_m) ** 2)
    # resolved boulders: shaded spheres
    for xi, yi, ri, ai in zip(x[~small], y[~small], r[~small], alb[~small]):
        rp = ri / pix_m
        cx, cy = (xi + size / 2) / pix_m, (yi + size / 2) / pix_m
        x0, x1 = int(max(cx - rp - 1, 0)), int(min(cx + rp + 2, n))
        y0, y1 = int(max(cy - rp - 1, 0)), int(min(cy + rp + 2, n))
        if x0 >= x1 or y0 >= y1:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        dx, dy = (gx - cx) / rp, (gy - cy) / rp
        rr = dx**2 + dy**2
        inside = rr < 1
        nz = np.sqrt(np.clip(1 - rr, 0, None))
        mu0 = np.clip(dx * sun_cam[0] + dy * sun_cam[1] + nz * sun_cam[2], 0, None)
        patch = w[y0:y1, x0:x1]
        patch[inside] = (ai * mu0)[inside]
    # dust shell of the seismic charge
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot((xx - n / 2) * pix_m - bx, (yy - n / 2) * pix_m - by)
    w += 0.35 * np.exp(-((rr - br) / 1.5) ** 2)
    # the chase: Slave I ahead, the Jedi starfighter dodging behind it.  Ships
    # in the ring are also lit by sunlight scattered off the boulders
    for name, sx_, sy_, hd, _ in ships:
        paint_ship(w, pix_m, name, sx_, sy_, hd, sun_cam, ambient=0.35)
    return w


# ==========================================================================
# Shark Side of the Moon (The Asylum, 2022): the communist sharks' base
# ==========================================================================
# In the film, Soviet shark-human hybrids hold a base on the far side of the
# Moon.  Poggle (the rust-red moon) is tidally locked, so its far side never
# faces Geonosis.  The sharks therefore built on the near side, under a
# Geonosis that hangs in their zenith, and laid out their flag on the regolith
# for all of Geonosis to see.  The base is "on the dark side" when Poggle is a
# crescent: then it lies in the night and is lit only by Geonosis-shine (and
# its own floodlights).  Everything here is INVENTED.

def _poly(*pts):
    return MplPath(np.array(pts, float))


def _mirror(right_side):
    """Closed outline from its right half (x >= 0), top to bottom."""
    left = [(-x, y) for x, y in reversed(right_side)]
    return _poly(*right_side, *left)


# Hammerhead shark seen from above, nose up, length 1 (y = 0 tail .. 1 head).
# Its cephalofoil makes it look like a hammer: crossed with a sickle it is
# the sharks' "hammerhead and sickle".
_HAMMERHEAD = _mirror([
    (0.00, 1.000), (0.10, 0.995), (0.27, 0.975), (0.33, 0.955), (0.345, 0.93),
    (0.33, 0.905), (0.27, 0.895), (0.10, 0.885), (0.075, 0.84), (0.075, 0.70),
    (0.090, 0.66), (0.240, 0.53), (0.235, 0.50), (0.075, 0.57), (0.060, 0.44),
    (0.050, 0.33), (0.100, 0.27), (0.045, 0.28), (0.030, 0.18), (0.020, 0.12),
    (0.120, -0.02), (0.075, -0.03), (0.000, 0.07)])
_HAMMERHEAD_EYES = [(0.315, 0.93), (-0.315, 0.93)]


def _star(cx, cy, r_out, r_in=None, rot=np.pi / 2):
    r_in = 0.382 * r_out if r_in is None else r_in
    a = rot + np.arange(10) * np.pi / 5
    r = np.where(np.arange(10) % 2 == 0, r_out, r_in)
    return _poly(*zip(cx + r * np.cos(a), cy + r * np.sin(a)))


# flag regions -> (visible-light RGB, Ks albedo).  The red field is
# iron-oxide paint and bright in the near-IR; the emblem is gold-coated
# Mylar; the border and the shark's eyes are carbon-black fabric.
FLAG_FIELD, FLAG_GOLD, FLAG_BLACK = 1, 2, 3
FLAG_COLOURS = {FLAG_FIELD: ((0.80, 0.08, 0.10), 0.35),
                FLAG_GOLD: ((1.00, 0.80, 0.18), 0.85),
                FLAG_BLACK: ((0.04, 0.04, 0.05), 0.05)}


def shark_flag(x, y):
    """Region code of the Red Shark flag at flag coordinates x in [0, 1.5]
    (hoist to fly), y in [0, 1] (bottom to top); 0 outside the flag.  A gold
    star and a gold hammerhead shark crossed with a sickle on a red field,
    with a black hem."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    code = np.zeros(x.shape, np.int8)
    inside = (x >= 0) & (x <= 1.5) & (y >= 0) & (y <= 1)
    hem = 0.025
    xi, yi = x[inside], y[inside]
    sub = np.full(xi.shape, FLAG_FIELD, np.int8)
    pts = np.column_stack([xi, yi])
    # sickle: crescent blade opening towards the upper left (towards the
    # shark's head), handle at the lower left
    sx, sy = xi - 0.80, yi - 0.50
    ang = np.degrees(np.arctan2(sy, sx)) % 360
    blade = (np.hypot(sx, sy) < 0.30) & (np.hypot(sx + 0.045, sy - 0.045) > 0.265) \
        & ~((ang > 105) & (ang < 205))
    p0 = np.array([0.80 + 0.28 * np.cos(np.radians(205)), 0.50 + 0.28 * np.sin(np.radians(205))])
    hd = np.array([-0.45, -0.89])
    hu = (xi - p0[0]) * hd[0] + (yi - p0[1]) * hd[1]
    hv = -(xi - p0[0]) * hd[1] + (yi - p0[1]) * hd[0]
    handle = (np.abs(hv) < 0.026) & (hu > -0.02) & (hu < 0.17)
    # hammerhead: length 0.80, tail lower right, head (the "hammer") upper left
    cx, cy, L, rot = 0.80, 0.48, 0.80, np.radians(40.0)
    c, s = np.cos(rot), np.sin(rot)

    def shark_frame(dx=0.0, dy=0.0):
        lx = ((xi - cx - dx) * c + (yi - cy - dy) * s) / L
        ly = (-(xi - cx - dx) * s + (yi - cy - dy) * c) / L + 0.5
        return lx, ly

    lx, ly = shark_frame()
    shark = _HAMMERHEAD.contains_points(np.column_stack([lx, ly]))
    # thin red outline separating the shark from the sickle
    halo = shark.copy()
    for a in np.radians(np.arange(0, 360, 45)):
        hx, hy = shark_frame(0.014 * np.cos(a), 0.014 * np.sin(a))
        halo |= _HAMMERHEAD.contains_points(np.column_stack([hx, hy]))
    eyes = np.zeros(xi.shape, bool)
    for ex, ey in _HAMMERHEAD_EYES:
        eyes |= np.hypot(lx - ex, ly - ey) < 0.02
    star = _star(0.20, 0.80, 0.11).contains_points(pts)
    sub[(blade | handle) & ~halo] = FLAG_GOLD
    sub[shark | star] = FLAG_GOLD
    sub[eyes & shark] = FLAG_BLACK
    sub[(xi < hem) | (xi > 1.5 - hem) | (yi < hem) | (yi > 1 - hem)] = FLAG_BLACK
    code[inside] = sub
    return code


def shark_flag_rgb(n=600):
    """The flag design in visible colours (for figures)."""
    y, x = np.mgrid[0:n, 0:int(1.5 * n)]
    code = shark_flag((x + 0.5) / n, (y + 0.5) / n)
    rgb = np.zeros(code.shape + (3,))
    for k, (col, _) in FLAG_COLOURS.items():
        rgb[code == k] = col
    return rgb


@dataclass
class SharkBase:
    """Base 'Krasnaya Akula' (Red Shark) on Poggle.  Layout in metres east /
    north of the anchor (lat, lon); albedos are Ks-band values."""

    lat_deg: float = 7.0
    lon_deg: float = -3.0
    flag_centre: tuple = (700.0, 350.0)
    flag_width: float = 1500.0                     # 3:2 flag, 1.5 km x 1 km
    hq: tuple = (-900.0, -450.0)
    tanks: tuple = ((-1400.0, -100.0), (-500.0, -1150.0))
    pad: tuple = (-1300.0, -1050.0)
    rocket: tuple = (100.0, -1100.0)
    extent: float = 2200.0                          # half-size of the footprint [m]
    lamp_temperature: float = 3000.0                # halogen floodlights [K]
    lamps: list = field(default_factory=list)       # (east, north, mast height m, power W)

    def __post_init__(self):
        if not self.lamps:
            fe, fn = self.flag_centre
            w, h = self.flag_width / 2 + 80, self.flag_width / 3 + 80
            # 50 kW stadium floodlights around the flag ...
            for k, (a, b) in enumerate([(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1),
                                        (0, 1), (-1, 1), (-1, 0)]):
                self.lamps.append((fe + a * w, fn + b * h, 60.0, 50e3))
            # ... 5 kW lamps along the road and around HQ
            for f in np.linspace(0.12, 0.88, 5):
                self.lamps.append((self.hq[0] + f * (fe - w - self.hq[0]),
                                   self.hq[1] + f * (fn - self.hq[1]), 20.0, 5e3))
            for a in np.radians(np.arange(30, 360, 60)):
                self.lamps.append((self.hq[0] + 220 * np.cos(a), self.hq[1] + 220 * np.sin(a), 20.0, 5e3))

    # -- geometry ---------------------------------------------------------
    def local_xy(self, lat, lon, radius):
        la0, lo0 = np.radians(self.lat_deg), np.radians(self.lon_deg)
        return radius * np.cos(la0) * (lon - lo0), radius * (lat - la0)

    def flag_coords(self, e, n):
        fe, fn = self.flag_centre
        s = self.flag_width / 1.5                    # metres per flag unit
        return (e - fe) / s + 0.75, (n - fn) / s + 0.5

    # -- surface ----------------------------------------------------------
    def paint(self, e, n):
        """Ks albedo override (NaN = untouched regolith).  The flag and the
        base lie on the terrain as it is."""
        alb = np.full(e.shape, np.nan)

        def disc(c, r):
            return np.hypot(e - c[0], n - c[1]) < r

        def line(p, q, half):
            p, q = np.asarray(p, float), np.asarray(q, float)
            d = q - p
            f = np.clip(((e - p[0]) * d[0] + (n - p[1]) * d[1]) / (d @ d), 0, 1)
            return np.hypot(e - p[0] - f * d[0], n - p[1] - f * d[1]) < half

        fe, fn = self.flag_centre
        hq = self.hq
        # roads (compacted regolith) and corridors
        road = line(hq, (fe - self.flag_width / 2, fn), 12) | line(hq, self.pad, 10) \
            | line(hq, self.rocket, 10) | line(hq, self.tanks[0], 8) | line(hq, self.tanks[1], 8)
        alb[road] = 0.45
        # rocket pad "Akula-1" with a scorched blast zone
        rr = np.hypot(e - self.rocket[0], n - self.rocket[1])
        scorch = (rr < 260) & (rr > 45)
        alb[scorch] = 0.28 - 0.22 * np.exp(-((rr[scorch] - 45) / 90.0))
        alb[rr <= 45] = 0.55
        # landing pad with a black ring
        rp = np.hypot(e - self.pad[0], n - self.pad[1])
        alb[rp < 90] = 0.55
        alb[(rp > 64) & (rp < 74)] = 0.05
        alb[(rp < 30)] = 0.05
        # shark tanks: glass domes over water (water is black at 2 um)
        for c in self.tanks:
            r = np.hypot(e - c[0], n - c[1])
            alb[r < 125] = 0.60
            alb[r < 112] = 0.02
        # HQ dome, black apron, ring of six habitat domes with corridors
        for a in np.radians(np.arange(0, 360, 60)):
            dome = (hq[0] + 330 * np.cos(a), hq[1] + 330 * np.sin(a))
            alb[line(hq, dome, 7)] = 0.60
            alb[disc(dome, 45)] = 0.75
        alb[disc(hq, 125) & ~disc(hq, 110)] = 0.06
        alb[disc(hq, 110)] = 0.80
        # the flag
        code = shark_flag(*self.flag_coords(e, n))
        for k, (_, a_ks) in FLAG_COLOURS.items():
            alb[code == k] = a_ks
        return alb

    def lamp_factor(self, e, n):
        """Ground irradiance from the floodlights, as photon fraction per m^2
        per watt (full-cutoff fixtures pointing straight down:
        E = P h / (2 pi r^3)); multiply by the lamp spectrum for 1 W."""
        g = np.zeros(e.shape)
        for le, ln, h, p in self.lamps:
            r2 = (e - le) ** 2 + (n - ln) ** 2 + h**2
            g += p * h / (2 * np.pi * r2**1.5)
        return g


SHARK_BASES = {"VII": SharkBase()}             # Poggle


def shark_base_for(moon):
    return SHARK_BASES.get(getattr(moon, "designation", None))


def apply_surface_overlays(moon, lat, lon, maps):
    """Hook called by MoonSurface.evaluate: paint the shark base on Poggle."""
    base = shark_base_for(moon)
    if base is None:
        return maps
    from .spectra import reflectance
    e, n = base.local_xy(lat, lon, moon.radius)
    near = (np.abs(e) < base.extent) & (np.abs(n) < base.extent)
    if not near.any():
        return maps
    alb = base.paint(e[near], n[near])
    ks_shape = float(reflectance(moon.surface, 1.0, np.array([2.15]))[0])
    a = maps["albedo"][near]
    painted = np.isfinite(alb)
    a[painted] = alb[painted] / ks_shape              # Ks albedo -> visual-scale weight
    maps["albedo"][near] = a
    return maps


# ==========================================================================
# FITS header easter eggs
# ==========================================================================
QUOTES = [
    "Help me, Obi-Wan Kenobi. You're my only hope.",
    "That's no moon. It's a space station.",
    "I have a bad feeling about this.",
    "The Force will be with you. Always.",
    "Begun, the Clone War has.",
    "These aren't the droids you're looking for.",
    "Stay on target...",
]


def fits_cards(label: str) -> dict:
    k = sum(map(ord, label)) % len(QUOTES)
    cards = {"OBSERVER": "Poggle the Lesser", "PROGID": "1138.C-0066(A)",
             "HIERARCH SITE QUOTE": QUOTES[k]}
    if label.startswith("shark"):
        cards["OBJECT"] = "Krasnaya Akula base"
        cards["HIERARCH SITE QUOTE"] = "Sharks of all oceans, unite!"
    return cards
