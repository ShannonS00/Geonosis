"""Ephemerides as seen from an observatory on the surface of Geonosis.

The site (INVENTED) sits near the Petranaki arena at latitude +12 deg.
Geonosis rotates prograde about the equatorial z axis with the CANON 30 h
period (treated as the sidereal rotation period).  All vectors are in the
planet-equatorial frame of :mod:`geonosis.system`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .system import GeonosisSystem, Moon, RAD2ARCSEC, HOUR


@dataclass
class Site:
    name: str = "Petranaki ELT (INVENTED)"
    latitude_deg: float = 12.0
    longitude_deg: float = 0.0
    altitude_m: float = 1500.0
    min_elevation_deg: float = 20.0          # ELT pointing limit
    night_sun_alt_deg: float = -12.0         # thin, dusty air: nautical twilight


class GeonosisObserver:
    def __init__(self, system: GeonosisSystem, site: Site | None = None):
        self.sys = system
        self.site = site or Site()

    # ------------------------------------------------------------------
    def site_vectors(self, t):
        """Site position and local (east, north, up) unit vectors, shape (3, N)."""
        t = np.atleast_1d(t)
        p = self.sys.planet
        theta = np.radians(self.site.longitude_deg) + 2 * np.pi * t / p.rotation_period
        lat = np.radians(self.site.latitude_deg)
        up = np.vstack([np.cos(lat) * np.cos(theta), np.cos(lat) * np.sin(theta),
                        np.sin(lat) * np.ones_like(theta)])
        east = np.vstack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        north = np.cross(up.T, east.T).T
        r = (p.radius + self.site.altitude_m) * up
        return r, east, north, up

    @staticmethod
    def _altaz(d, east, north, up):
        dn = d / np.linalg.norm(d, axis=0)
        alt = np.degrees(np.arcsin(np.clip(np.sum(dn * up, 0), -1, 1)))
        az = np.degrees(np.arctan2(np.sum(dn * east, 0), np.sum(dn * north, 0))) % 360
        return alt, az

    def sun_altaz(self, t):
        r, e, n, up = self.site_vectors(t)
        s = self.sys.sun_direction_eq(np.atleast_1d(t)).reshape(3, -1)
        return self._altaz(s, e, n, up)

    def target_vector(self, moon: Moon, t):
        r_site, e, n, up = self.site_vectors(t)
        r_m = self.sys.moon_position_eq(moon, np.atleast_1d(t))
        r_m = r_m.reshape(3, -1)
        return r_m - r_site, r_m, (r_site, e, n, up)

    def moon_ephemeris(self, moon: Moon, t) -> dict:
        """Topocentric quantities for a moon at time(s) t [s]."""
        t = np.atleast_1d(np.asarray(t, float))
        d, r_m, (r_site, e, n, up) = self.target_vector(moon, t)
        dist = np.linalg.norm(d, axis=0)
        alt, az = self._altaz(d, e, n, up)
        s = self.sys.sun_direction_eq(t).reshape(3, -1)
        dn = d / dist
        phase = np.degrees(np.arccos(np.clip(np.sum(-dn * s, 0), -1, 1)))
        illum = (1 + np.cos(np.radians(phase))) / 2
        # eclipse: inside the planet's cylindrical shadow
        along = np.sum(r_m * s, 0)
        perp = np.linalg.norm(r_m - along * s, axis=0)
        eclipsed = (along < 0) & (perp < self.sys.planet.radius)
        # ring crossing of the line of sight
        ring_trans = np.ones_like(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = -r_site[2] / d[2]
        cross = (frac > 0) & (frac < 1)
        pc = r_site + frac * d
        rc = np.hypot(pc[0], pc[1]) / self.sys.planet.radius
        in_ring = cross & (rc > self.sys.ring.inner) & (rc < self.sys.ring.outer)
        ring_trans = np.where(in_ring, np.exp(-self.sys.ring.tau / np.abs(dn[2]).clip(1e-3)), 1.0)
        # angular rates: inertial (vs. the stars) and in the local alt-az frame
        dt = 10.0
        d2, *_ = self.target_vector(moon, t + dt)
        dn2 = d2 / np.linalg.norm(d2, axis=0)
        rate_inertial = np.degrees(np.arccos(np.clip(np.sum(dn * dn2, 0), -1, 1))) * 3600 / dt
        alt2, az2 = self._altaz(d2, *self.site_vectors(t + dt)[1:])
        daz = (az2 - az + 180) % 360 - 180
        rate_altaz = np.hypot(alt2 - alt, daz * np.cos(np.radians(alt))) * 3600 / dt
        sun_alt, _ = self.sun_altaz(t)
        return dict(
            t=t, alt=alt, az=az, distance=dist,
            ang_diam_arcsec=2 * np.arcsin(np.clip(moon.radius / dist, 0, 1)) * RAD2ARCSEC,
            phase_angle=phase, illuminated_fraction=illum, eclipsed=eclipsed,
            ring_transmission=ring_trans, rate_inertial=rate_inertial,
            rate_altaz=rate_altaz, sun_alt=sun_alt,
            m_per_mas=dist * np.pi / 180 / 3600e3,
            observable=(alt > self.site.min_elevation_deg) & (sun_alt < self.site.night_sun_alt_deg)
                       & ~eclipsed & (ring_trans > 0.5),
        )

    def sidereal_rate(self):
        """Apparent rate of the 'fixed stars' on the celestial equator ["/s]."""
        return 360 * 3600 / self.sys.planet.rotation_period

    def ring_altaz(self, t, n=720):
        """Sky positions of the ring edges from the site, with lit flags."""
        r_site, e, nn, up = self.site_vectors(t)
        s = self.sys.sun_direction_eq(np.atleast_1d(t)).reshape(3)
        psi = np.linspace(0, 2 * np.pi, n)
        out = {}
        Rp = self.sys.planet.radius
        for name, rr in (("inner", self.sys.ring.inner), ("outer", self.sys.ring.outer)):
            p = rr * Rp * np.vstack([np.cos(psi), np.sin(psi), np.zeros_like(psi)])
            d = p - r_site
            alt, az = self._altaz(d, e, nn, up)
            along = s @ p
            perp = np.linalg.norm(p - np.outer(s, along), axis=0)
            lit = ~((along < 0) & (perp < Rp))
            out[name] = dict(alt=alt, az=az, lit=lit)
        return out

    def night_window(self, t_start, step=600.0, span=2.0):
        """First complete night after t_start: (t_dusk, t_dawn)."""
        tt = t_start + np.arange(0, span * self.sys.planet.rotation_period, step)
        sun_alt, _ = self.sun_altaz(tt)
        dark = sun_alt < self.site.night_sun_alt_deg
        edges = np.flatnonzero(np.diff(dark.astype(int)))
        starts = [k + 1 for k in edges if dark[k + 1]]
        ends = [k for k in edges if not dark[k + 1]]
        for a in starts:
            later = [b for b in ends if b > a]
            if later:
                return tt[a], tt[later[0]]
        raise RuntimeError("no night found")


def hours(seconds):
    return np.asarray(seconds) / HOUR
