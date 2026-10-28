"""Deterministic procedural moon surfaces: crater topography + albedo maps.

Surfaces are evaluated in sinusoidal-projection coordinates
    u = R * lon * cos(lat),  v = R * lat        [m]
centred on the sub-Geonosis meridian (lon = 0).  The major moons are tidally
locked, so this projection is least distorted exactly on the face that an
observer on Geonosis sees.  Craters live on a hierarchy of hashed grid cells,
so any patch can be rendered at any resolution and adjacent mosaic tiles agree.
"""

from __future__ import annotations

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _hash(*ints) -> np.ndarray:
    """splitmix64-style integer hash of integer arrays -> uint64."""
    h = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for k in ints:
            h = h ^ (np.asarray(k).astype(np.int64).astype(np.uint64)
                     + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2)))
            h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(31))
    return h


def _uniform(*ints) -> np.ndarray:
    return (_hash(*ints) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def value_noise(u, v, scale, seed, octaves=5, persistence=0.55) -> np.ndarray:
    """Fractal (fBm) value noise in [-1, 1] approx; ``scale`` in metres."""
    total = np.zeros_like(u, dtype=float)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        s = scale / 2**o
        x, y = u / s, v / s
        i0, j0 = np.floor(x), np.floor(y)
        fx, fy = x - i0, y - j0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        c = [[_uniform(seed, o, i0 + a, j0 + b) for b in (0, 1)] for a in (0, 1)]
        val = ((c[0][0] * (1 - fx) + c[1][0] * fx) * (1 - fy)
               + (c[0][1] * (1 - fx) + c[1][1] * fx) * fy)
        total += amp * (2 * val - 1)
        norm += amp
        amp *= persistence
    return total / norm


class CraterField:
    """Hierarchical procedural crater population.

    Parameters
    ----------
    seed : int
    r_max : float   largest crater radius [m]
    r_min : float   smallest crater radius rendered by default [m]
    density : float probability that a grid cell hosts a crater
    depth_ratio : float  depth / diameter of fresh craters
    """

    def __init__(self, seed, r_max, r_min=5.0, density=0.55, depth_ratio=0.2):
        self.seed, self.r_max, self.r_min = seed, r_max, r_min
        self.density, self.depth_ratio = density, depth_ratio

    def levels(self, r_min=None):
        r_min = self.r_min if r_min is None else r_min
        out, r = [], self.r_max
        lev = 0
        while r >= r_min:
            out.append((lev, r))
            r /= 2.0
            lev += 1
        return out

    def evaluate(self, u, v, r_min=None):
        """Return (height [m], fresh-ejecta weight [0..1]) at coordinates."""
        h = np.zeros_like(u, dtype=float)
        ejecta = np.zeros_like(u, dtype=float)
        for lev, r_lev in self.levels(r_min):
            cell = 4.0 * r_lev                     # one crater per ~4 radii
            ci = np.floor(u / cell).astype(np.int64)
            cj = np.floor(v / cell).astype(np.int64)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    i, j = ci + di, cj + dj
                    # hash each distinct cell once, then gather per pixel
                    span = int(j.max() - j.min() + 1)
                    key = (i - i.min()) * span + (j - j.min())
                    ukey, inv = np.unique(key, return_inverse=True)
                    ui = ukey // span + i.min()
                    uj = ukey % span + j.min()
                    present_u = _uniform(self.seed, lev, ui, uj, 1) < self.density
                    if not present_u.any():
                        continue
                    present = present_u[inv]
                    cx = ((ui + _uniform(self.seed, lev, ui, uj, 2)) * cell)[inv]
                    cy = ((uj + _uniform(self.seed, lev, ui, uj, 3)) * cell)[inv]
                    # radius drawn within the octave (power-law-ish)
                    rad = (r_lev * (0.55 + 0.45 * _uniform(self.seed, lev, ui, uj, 4)))[inv]
                    age = _uniform(self.seed, lev, ui, uj, 5)[inv]   # 0 fresh .. 1 old
                    rho = np.hypot(u - cx, v - cy) / rad
                    near = present & (rho < 2.5)
                    if not near.any():
                        continue
                    depth = (self.depth_ratio * 2 * rad * (1 - 0.75 * age))[near]
                    rim = (0.04 * 2 * rad * (1 - 0.8 * age))[near]
                    rr = rho[near]
                    prof = np.where(rr < 1.0,
                                    depth * (rr**2 - 1) + rim,
                                    # tapered to exactly zero at rho = 2.5 (no seams)
                                    rim * np.clip(np.clip(rr, 1, None) ** -4 - 2.5**-4, 0, None)
                                    / (1 - 2.5**-4))
                    # older craters are flattened inside: blend towards smooth floor
                    a_n = age[near]
                    prof = np.where(rr < 0.45, np.maximum(prof, -depth * (0.8 - 0.6 * a_n) + rim), prof)
                    h[near] += prof
                    ej = np.where((rr > 0.9) & (rr < 2.2), np.exp(-((rr - 1.2) / 0.45) ** 2), 0)
                    ejecta[near] = np.maximum(ejecta[near], ej * (1 - a_n) ** 2)
        return h, ejecta


# Surface recipes for each moon type (INVENTED)
SURFACES = {
    "ice":  dict(depth_ratio=0.16, density=0.45, albedo_var=0.18, ejecta_boost=0.25,
                 ice_fraction=True),
    "rock": dict(depth_ratio=0.20, density=0.60, albedo_var=0.25, ejecta_boost=0.35,
                 ice_fraction=False),
    "rust": dict(depth_ratio=0.18, density=0.50, albedo_var=0.30, ejecta_boost=-0.15,
                 ice_fraction=False),
}


class MoonSurface:
    """Height, albedo and ice-fraction maps for one moon."""

    def __init__(self, moon, seed=None):
        self.moon = moon
        self.recipe = SURFACES[moon.surface]
        if seed is None:   # deterministic across runs (unlike Python's hash())
            seed = sum(ord(c) * 131**k for k, c in enumerate(moon.designation)) % 100_003
        self.seed = seed
        r_max = min(0.25 * moon.radius, 150e3)
        self.craters = CraterField(self.seed, r_max=r_max,
                                   density=self.recipe["density"],
                                   depth_ratio=self.recipe["depth_ratio"])

    def coords(self, lat, lon):
        R = self.moon.radius
        return R * lon * np.cos(lat), R * lat

    def evaluate(self, lat, lon, r_min):
        """Return dict(height [m], albedo [geometric], ice [0..1])."""
        u, v = self.coords(lat, lon)
        h, ej = self.craters.evaluate(u, v, r_min=r_min)
        R = self.moon.radius
        # regional albedo pattern + roughness
        broad = value_noise(u, v, 0.35 * R, self.seed + 7, octaves=4)
        fine = value_noise(u, v, max(20 * r_min, 0.002 * R), self.seed + 11, octaves=3)
        rough = value_noise(u, v, max(8 * r_min, 50.0), self.seed + 13, octaves=4)
        h = h + 0.004 * R * broad * 0.2 + 3 * r_min * rough * 0.15
        alb = self.moon.albedo * (1 + self.recipe["albedo_var"] * (0.8 * broad + 0.2 * fine)
                                  + self.recipe["ejecta_boost"] * ej)
        out = dict(height=h, albedo=np.clip(alb, 0.01, 1.0))
        if self.recipe["ice_fraction"]:
            # dark rocky lag deposits: regional lows of `broad` plus km-scale
            # patches (sublimation lag), brightened by fresh crater ejecta
            patch = value_noise(u, v, 2500.0, self.seed + 17, octaves=4)
            ice = np.clip(0.62 + 0.30 * broad + 0.55 * patch + 0.30 * ej, 0.05, 1.0)
            out["ice"] = ice
        else:
            out["ice"] = np.zeros_like(h)
        from .easter_eggs import apply_surface_overlays
        return apply_surface_overlays(self.moon, lat, lon, out)
