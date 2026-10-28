"""Renderers: phase-lit textured moons (with cast shadows) and the planet + ring.

Camera frame convention
-----------------------
x_c = image "right", y_c = image "up", z_c = from the target towards the
observer.  Arrays are indexed [row, col] = [y, x] and meant to be shown with
``origin="lower"``.  All renderers are orthographic, which is accurate for
targets smaller than about a degree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from scipy import ndimage

from .system import GeonosisSystem, Moon, lambert_phase
from .textures import MoonSurface, value_noise

ARCSEC = np.pi / 180 / 3600


def camera_basis(view_dir, up_hint=(0, 0, 1.0)):
    """Orthonormal camera basis; view_dir points from target to observer."""
    z = np.asarray(view_dir, float)
    z = z / np.linalg.norm(z)
    up = np.asarray(up_hint, float)
    if abs(up @ z) > 0.99:
        up = np.array([1.0, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def tidally_locked_frame(system: GeonosisSystem, moon: Moon, t: float) -> np.ndarray:
    """Body-fixed axes (columns) in the planet-equatorial frame: x towards
    Geonosis (sub-planet point at lon=0), z along the orbit normal."""
    r = system.moon_position_eq(moon, t)
    r2 = system.moon_position_eq(moon, t + 60.0)
    xb = -r / np.linalg.norm(r)
    zb = np.cross(r, r2 - r)
    zb /= np.linalg.norm(zb)
    yb = np.cross(zb, xb)
    return np.column_stack([xb, yb, zb])


@dataclass
class View:
    """Observer-target geometry for a render (vectors in the eq frame)."""

    distance: float          # observer-target distance [m]
    to_observer: np.ndarray  # unit vector target -> observer
    to_sun: np.ndarray       # unit vector target -> Ea
    up_hint: np.ndarray      # roughly "up" on the image (e.g. local zenith/north)
    body_axes: np.ndarray    # 3x3 body frame (columns) in the eq frame

    @property
    def phase_angle(self):
        return np.arccos(np.clip(self.to_observer @ self.to_sun, -1, 1))


def render_moon(surface: MoonSurface, view: View, nx: int, ny: int,
                pixscale_arcsec: float, center_arcsec=(0.0, 0.0),
                r_min=None, shadows=True, lunar_lambert=None, props=None,
                secondary_light=None) -> dict:
    """Render a patch of sky (nx x ny pixels) containing (part of) a moon.

    ``center_arcsec`` is the patch centre offset from the moon centre along
    (x_c, y_c).  Returns maps: ``reflect`` (I/F per unit albedo), ``albedo``,
    ``ice``, ``shadow``, ``disc``, ``lat``/``lon`` plus metadata.
    ``props`` are 3-D objects standing on the surface (see
    :mod:`geonosis.easter_eggs`); by default the moon's registered props.
    ``secondary_light``: unit vector (eq frame) towards a second, distant
    light such as Geonosis-shine.  Its I/F per unit albedo and per unit
    (normal-incidence) irradiance goes to ``reflect2``, without cast shadows
    (fine for light from nearly behind the observer).
    """
    moon = surface.moon
    R = moon.radius
    xc, yc, zc = camera_basis(view.to_observer, view.up_hint)
    pix_rad = pixscale_arcsec * ARCSEC
    pix_m = pix_rad * view.distance
    if r_min is None:
        r_min = max(2.0 * pix_m, 1.0)

    jj = (np.arange(nx) - (nx - 1) / 2) * pix_rad + center_arcsec[0] * ARCSEC
    ii = (np.arange(ny) - (ny - 1) / 2) * pix_rad + center_arcsec[1] * ARCSEC
    X, Y = np.meshgrid(jj * view.distance, ii * view.distance)
    rho2 = (X**2 + Y**2) / R**2
    disc = rho2 < 1.0
    out = dict(reflect=np.zeros((ny, nx)), albedo=np.zeros((ny, nx)),
               ice=np.zeros((ny, nx)), shadow=np.zeros((ny, nx), bool),
               disc=disc, pix_m=pix_m, phase_angle=view.phase_angle)
    if not disc.any():
        return out

    Z = np.sqrt(np.clip(1 - rho2, 0, None)) * R
    n_c = np.stack([X / R, Y / R, Z / R])                    # camera frame
    n_eq = (xc[:, None, None] * n_c[0] + yc[:, None, None] * n_c[1]
            + zc[:, None, None] * n_c[2])
    B = view.body_axes
    n_b = np.einsum("ji,jkl->ikl", B, n_eq)                  # B^T n
    lat = np.arcsin(np.clip(n_b[2], -1, 1))
    lon = np.arctan2(n_b[1], n_b[0])

    maps = surface.evaluate(lat[disc], lon[disc], r_min=r_min)
    h = np.zeros((ny, nx))
    h[disc] = maps["height"]
    out["albedo"][disc] = maps["albedo"]
    out["ice"][disc] = maps["ice"]

    # local east / north unit vectors (eq frame) -> projections on the image
    zb_eq = B[:, 2]
    east = np.cross(zb_eq[:, None, None], n_eq, axis=0)
    east /= np.linalg.norm(east, axis=0) + 1e-12
    north = np.cross(n_eq, east, axis=0)
    # image-plane height gradient [m/m]
    hy, hx = np.gradient(h, pix_m)
    ex, ey = np.tensordot(xc, east, 1), np.tensordot(yc, east, 1)
    nx_, ny_ = np.tensordot(xc, north, 1), np.tensordot(yc, north, 1)
    h_a = hx * ex + hy * ey
    h_b = hx * nx_ + hy * ny_
    # avoid limb blow-up of the chain rule where foreshortening is extreme
    lim = 3.0
    h_a, h_b = np.clip(h_a, -lim, lim), np.clip(h_b, -lim, lim)
    n_p = n_eq - h_a * east - h_b * north
    n_p /= np.linalg.norm(n_p, axis=0)

    s = view.to_sun
    mu0 = np.tensordot(s, n_p, 1)
    mu = np.tensordot(zc, n_p, 1)
    mu0_smooth = np.tensordot(s, n_eq, 1)
    if lunar_lambert is None:
        lunar_lambert = 0.45 if moon.surface == "ice" else 0.65
    L = lunar_lambert
    lit = disc & (mu0 > 0) & (mu > 0) & (mu0_smooth > -0.05)
    f = np.zeros((ny, nx))
    f[lit] = (L * 2 * mu0[lit] / (mu0[lit] + mu[lit]) + (1 - L) * mu0[lit])

    if shadows:
        sxy = np.array([s @ xc, s @ yc])
        if np.hypot(*sxy) > 1e-3:
            shadow = _cast_shadows(X, Y, Z + h * n_c[2], disc, sxy, s @ zc, pix_m)
            out["shadow"] = shadow & disc
            f[shadow] = 0.0
    out["reflect"] = f
    if secondary_light is not None:
        mu2 = np.tensordot(np.asarray(secondary_light, float), n_p, 1)
        lit2 = disc & (mu2 > 0) & (mu > 0)
        f2 = np.zeros((ny, nx))
        f2[lit2] = L * 2 * mu2[lit2] / (mu2[lit2] + mu[lit2]) + (1 - L) * mu2[lit2]
        out["reflect2"] = f2
    out["height"] = h
    out["lat"], out["lon"] = lat, lon
    if props is None:
        from .easter_eggs import props_for
        props = props_for(moon)
    for prop in props:
        prop.apply(out, surface, view, X, Y, Z, h, n_c, (xc, yc, zc), L)
    return out


def _cast_shadows(X, Y, Zt, disc, sxy, sz, pix_m):
    """Exact shadow test for parallel light rays under orthographic
    projection: rays project to parallel image lines along ``sxy``.
    A pixel is shadowed if terrain upstream (towards the sun) rises above the
    ray, i.e. G(q) = Z_t(q) - k*l(q) exceeds G at the pixel, with l the image
    coordinate along the sun direction and k = s_z / |s_xy|."""
    d = sxy / np.hypot(*sxy)
    k = sz / np.hypot(*sxy)
    ell = X * d[0] + Y * d[1]
    G = np.where(disc, Zt - k * ell, -1e30)
    ang = np.degrees(np.arctan2(d[1], d[0]))
    # rotate so the sun direction points to +x (array columns increasing)
    Gr = ndimage.rotate(G, ang, reshape=True, order=0, cval=-1e30)
    upstream = np.maximum.accumulate(Gr[:, ::-1], axis=1)[:, ::-1]
    # exclude the pixel itself: shift by one column
    upstream = np.concatenate([upstream[:, 1:], np.full((Gr.shape[0], 1), -1e30)], axis=1)
    back = ndimage.rotate(upstream, -ang, reshape=True, order=0, cval=-1e30)
    # crop back to original shape (centre)
    oy = (back.shape[0] - G.shape[0]) // 2
    ox = (back.shape[1] - G.shape[1]) // 2
    back = back[oy:oy + G.shape[0], ox:ox + G.shape[1]]
    # tolerance: the change of G across ~1.5 pixels, so nearest-neighbour
    # rotation round-off cannot create spurious self-shadowing
    Gd = np.where(disc, G, np.nan)
    gy, gx = np.gradient(Gd)
    tol = 1.5 * np.nan_to_num(np.maximum(np.abs(gx), np.abs(gy)), nan=0.0) + 1e-9 * np.abs(Zt).max()
    return disc & (back > G + tol)


def moon_weight_maps(render: dict, surface_type: str, rock_albedo_scale=0.25):
    """ScopeSim image weights: albedo x I/F, split into ice and rock parts."""
    w = render["albedo"] * render["reflect"]
    if surface_type == "ice":
        return {"ice": w * render["ice"],
                "rock": w * (1 - render["ice"]) * rock_albedo_scale}
    return {surface_type: w}


def image_hdu(data, pixscale_arcsec, ra_deg=0.0, dec_deg=0.0, bunit=""):
    """FITS ImageHDU with a simple TAN WCS centred on the array."""
    hdu = fits.ImageHDU(data=np.asarray(data, dtype=np.float32))
    h = hdu.header
    ny, nx = data.shape
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    h["CDELT1"], h["CDELT2"] = -pixscale_arcsec / 3600, pixscale_arcsec / 3600
    h["CRPIX1"], h["CRPIX2"] = (nx + 1) / 2, (ny + 1) / 2
    h["CRVAL1"], h["CRVAL2"] = ra_deg, dec_deg
    h["BUNIT"] = bunit
    return hdu


# --------------------------------------------------------------------------
# Planet + ring "postcard"
# --------------------------------------------------------------------------
def render_planet_and_ring(system: GeonosisSystem, t: float, view_dir_eq,
                           npix=900, fov_rp=7.0, up_hint=(0, 0, 1.0)):
    """RGB image of Geonosis and its ring seen from far away along
    ``view_dir_eq`` (planet -> observer, eq frame).  Includes the planet's
    shadow on the ring and the ring's shadow on the planet."""
    p, ring = system.planet, system.ring
    Rp = p.radius
    xc, yc, zc = camera_basis(view_dir_eq, up_hint)
    s = system.sun_direction_eq(t)
    s = s / np.linalg.norm(s)
    lin = (np.arange(npix) - (npix - 1) / 2) / npix * fov_rp * Rp
    X, Y = np.meshgrid(lin, lin)
    rgb = np.zeros((npix, npix, 3))

    # colour palette (INVENTED but inspired by the films: rust-orange desert)
    desert = np.array([0.80, 0.42, 0.20])
    ring_col = np.array([0.62, 0.52, 0.44])

    # --- planet
    rho2 = (X**2 + Y**2) / Rp**2
    on_planet = rho2 < 1
    Zp = np.sqrt(np.clip(1 - rho2, 0, None)) * Rp
    P = (xc[:, None, None] * X + yc[:, None, None] * Y + zc[:, None, None] * Zp)  # eq frame points
    n = P / Rp
    mu0 = np.tensordot(s, n, 1)
    lat = np.arcsin(np.clip(n[2], -1, 1))
    lon = np.arctan2(n[1], n[0])
    u_, v_ = Rp * lon * np.cos(lat), Rp * lat
    tex = (1 + 0.25 * value_noise(u_, v_, 1500e3, 3, octaves=6)
           + 0.10 * value_noise(u_, v_, 200e3, 5, octaves=4))
    band = 1 + 0.08 * np.sin(lat * 9 + 2 * value_noise(u_, v_, 2500e3, 9, octaves=2))
    shade = np.clip(mu0, 0, None)
    # ring shadow on planet: follow the ray towards the sun to the ring plane
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = -P[2] / s[2]
        hit = P + tt * s[:, None, None]
        rr = np.hypot(hit[0], hit[1]) / Rp
        in_ring = (tt > 0) & (rr > ring.inner) & (rr < ring.outer)
        ring_shadow = np.where(in_ring, np.exp(-ring.tau / max(abs(s[2]), 1e-3)), 1.0)
    col = desert[None, None, :] * (tex * band * shade * ring_shadow)[:, :, None]
    rgb[on_planet] = col[on_planet]

    # --- ring (plane z_eq = 0): intersect line of sight with the plane
    with np.errstate(divide="ignore", invalid="ignore"):
        # point on LOS: X xc + Y yc + w zc ; z-component = 0
        w = -(X * xc[2] + Y * yc[2]) / zc[2]
    Q = xc[:, None, None] * X + yc[:, None, None] * Y + zc[:, None, None] * w
    rq = np.hypot(Q[0], Q[1]) / Rp
    in_ring = (rq > ring.inner) & (rq < ring.outer) & np.isfinite(w)
    # radial structure: gaps and ringlets (INVENTED)
    radial = (1 + 0.35 * np.sin(rq * 37.0) * np.sin(rq * 5.3)
              - 0.8 * np.exp(-((rq - 2.35) / 0.025) ** 2))
    tau = ring.tau * np.clip(radial, 0.05, None)
    mu = abs(zc[2])
    alpha = np.arccos(np.clip(s @ zc, -1, 1))
    lit_side = (s[2] * zc[2]) > 0
    refl = np.array([ring_reflectance_factor_vec(mu, abs(s[2]), tt_, alpha)
                     for tt_ in [tau]])[0]
    if not lit_side:
        refl = refl * 0.3
    # planet's shadow on the ring: ray from ring point towards the sun hits the planet?
    b = np.tensordot(s, Q, 1)
    c = np.sum(Q**2, axis=0) - Rp**2
    in_shadow = (b < 0) & (b**2 - c > 0)
    refl = np.where(in_shadow, 0.0, refl)
    opacity = 1 - np.exp(-tau / max(mu, 1e-3))
    ring_rgb = ring_col[None, None, :] * (1.1 * refl)[:, :, None]
    # occlusion: ring point in front of planet (w > Zp) or off-planet
    front = in_ring & (~on_planet | (w > Zp))
    rgb[front] = (rgb[front] * (1 - opacity[front, None]) + ring_rgb[front])
    return np.clip(rgb, 0, 1)


def ring_reflectance_factor_vec(mu, mu0, tau, alpha):
    slab = 1 - np.exp(-tau * (1 / max(mu, 1e-4) + 1 / max(mu0, 1e-4)))
    return 2 * mu0 / (mu0 + mu) * slab * lambert_phase(alpha)
