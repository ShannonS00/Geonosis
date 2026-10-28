"""The Geonosis system: star Ea, the ringed planet Geonosis and its 15 moons.

Every parameter carries a provenance tag:
  CANON    -- from Star Wars canon (Wookieepedia, "Geonosis")
  DERIVED  -- computed from canon values with real physics
  INVENTED -- our own choice, made to be physically consistent

Canon facts used: star "Ea", Geonosis is orbital position 2, diameter
11,370 km, rotation period 30 h, orbital period 256 standard days, rocky rings,
fifteen moons of which four are major, one of them the icy moon Bahryn.

Reference frames
----------------
* ``orbit``   : Ea-centred, x-y plane = Geonosis' orbital plane.
* ``eq``      : Geonosis-centred equatorial frame (= ring plane, z = spin axis).
* ``sky``     : as seen from Earth; x = east, y = north, z = towards Earth.
All internal units are SI (m, s, kg, rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import constants as const

G = const.G.si.value
AU = const.au.si.value
PC = const.pc.si.value
M_SUN = const.M_sun.si.value
R_SUN = const.R_sun.si.value
L_SUN = const.L_sun.si.value
M_EARTH = const.M_earth.si.value
SIGMA_SB = const.sigma_sb.si.value
DAY = 86400.0
HOUR = 3600.0
RAD2MAS = 180 / np.pi * 3600e3
RAD2ARCSEC = 180 / np.pi * 3600


# --------------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def solve_kepler(M, e, tol=1e-12):
    """Eccentric anomaly E for mean anomaly M (vectorised Newton solver)."""
    M = np.mod(np.asarray(M, dtype=float), 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


@dataclass
class Orbit:
    """Keplerian orbit around a central mass ``gm`` [m^3 s^-2]."""

    a: float                 # semi-major axis [m]
    e: float                 # eccentricity
    inc: float               # inclination to reference plane [rad]
    node: float              # longitude of ascending node [rad]
    peri: float              # argument of pericentre [rad]
    mean_anomaly0: float     # mean anomaly at t = 0 [rad]
    gm: float                # G * (M_central + M_body)

    @property
    def period(self) -> float:
        return 2 * np.pi * np.sqrt(self.a**3 / self.gm)

    @property
    def mean_motion(self) -> float:
        return 2 * np.pi / self.period

    @property
    def rotation(self) -> np.ndarray:
        return rot_z(self.node) @ rot_x(self.inc) @ rot_z(self.peri)

    def position(self, t) -> np.ndarray:
        """Position vector(s) [m] at time(s) t [s]; shape (3,) or (3, N)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        E = solve_kepler(self.mean_anomaly0 + self.mean_motion * t, self.e)
        x = self.a * (np.cos(E) - self.e)
        y = self.a * np.sqrt(1 - self.e**2) * np.sin(E)
        r = self.rotation @ np.vstack([x, y, np.zeros_like(x)])
        return r[:, 0] if r.shape[1] == 1 else r


# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------
@dataclass
class Star:
    name: str = "Ea"                                    # CANON
    mass: float = 1.00 * M_SUN                          # INVENTED (G2V-like)
    radius: float = 1.00 * R_SUN                        # INVENTED
    teff: float = 5770.0                                # INVENTED [K]
    distance: float = 1.30 * PC                         # INVENTED (user choice)
    ra_deg: float = 262.5                               # INVENTED (17h30m)
    dec_deg: float = -35.0                              # INVENTED
    spectral_type: str = "G2V"                          # INVENTED

    @property
    def luminosity(self) -> float:
        return 4 * np.pi * self.radius**2 * SIGMA_SB * self.teff**4


@dataclass
class Ring:
    inner: float            # [planet radii]
    outer: float            # [planet radii]
    tau: float              # normal optical depth
    albedo: float           # particle reflectivity (I/F at normal incidence)
    particle_density: float  # [kg m^-3]
    provenance: str = "CANON existence ('rocky rings'); INVENTED dimensions"


@dataclass
class Planet:
    name: str = "Geonosis"
    radius: float = 11370e3 / 2               # CANON (diameter 11,370 km)
    rotation_period: float = 30 * HOUR        # CANON
    orbital_period: float = 256 * DAY         # CANON (256 standard days)
    density: float = 4500.0                   # INVENTED [kg m^-3]
    obliquity: float = np.radians(18.0)       # INVENTED
    spin_longitude: float = np.radians(40.0)  # INVENTED azimuth of spin axis
    bond_albedo: float = 0.25                 # INVENTED (desert)
    geometric_albedo: float = 0.22            # INVENTED (V band)
    eccentricity: float = 0.03                # INVENTED

    @property
    def mass(self) -> float:
        return 4 / 3 * np.pi * self.radius**3 * self.density

    @property
    def gm(self) -> float:
        return G * self.mass

    @property
    def surface_gravity(self) -> float:
        return self.gm / self.radius**2


@dataclass
class Moon:
    designation: str          # Roman numeral, ordered by semi-major axis
    name: str
    radius: float             # [m]
    albedo: float             # geometric albedo (visual)
    surface: str              # 'ice', 'rock', 'rust'
    density: float            # [kg m^-3]
    a: float                  # semi-major axis [m]
    e: float
    inc_deg: float            # to Geonosis' equator; >90 = retrograde
    node_deg: float
    peri_deg: float
    mean_anomaly0_deg: float
    major: bool = False
    provenance: str = "INVENTED"
    orbit: Orbit | None = field(default=None, repr=False)

    @property
    def mass(self) -> float:
        return 4 / 3 * np.pi * self.radius**3 * self.density

    @property
    def label(self) -> str:
        return f"{self.designation} {self.name}" if self.name else self.designation


# 15 moons (CANON count), 4 major (CANON) -- Bahryn is CANON and icy.
# Names other than Bahryn are INVENTED: the three other major moons are named
# after Geonosian characters (Archduke Poggle the Lesser, Sun Fac and Queen
# Karina the Great). Minor moons only carry their Roman-numeral designation.
_MOON_TABLE = [
    # desig, name,      R[km], albedo, surface, rho,  a[km],   e,     inc,   node, peri, M0, major, provenance
    ("I",     "",          8,  0.08, "rock", 2800,  24_000, 0.002,   0.1,   10,   0,   20, False, "INVENTED (ring shepherd)"),
    ("II",    "",         12,  0.09, "rock", 2800,  31_000, 0.004,   0.3,  200,  40,  140, False, "INVENTED"),
    ("III",   "",         21,  0.10, "rock", 2800,  42_000, 0.006,   0.4,  300,  90,  250, False, "INVENTED"),
    ("IV",    "Sun Fac",  180, 0.12, "rock", 2900,  60_000, 0.004,   0.6,   80,  15,   75, True,  "INVENTED name/orbit (major moon)"),
    ("V",     "",          6,  0.10, "rock", 2600,  90_000, 0.010,   1.2,  130, 200,  310, False, "INVENTED"),
    ("VI",    "Bahryn",   610, 0.62, "ice",  1700, 120_000, 0.008,   0.3,  250, 100,  190, True,  "CANON name + icy; INVENTED size/orbit"),
    ("VII",   "Poggle",   400, 0.18, "rust", 3000, 190_000, 0.012,   0.9,   45, 300,   35, True,  "INVENTED name/orbit (major moon)"),
    ("VIII",  "",         15,  0.07, "rock", 2400, 240_000, 0.020,   2.0,  170,  60,  100, False, "INVENTED"),
    ("IX",    "Karina",   560, 0.25, "rock", 2700, 320_000, 0.015,   1.4,  320, 220,  280, True,  "INVENTED name/orbit (largest moon)"),
    ("X",     "",         25,  0.06, "rock", 2200, 420_000, 0.080,  28.0,   20, 120,   60, False, "INVENTED (captured)"),
    ("XI",    "",          5,  0.06, "rock", 2200, 435_000, 0.060,  35.0,  210, 250,  230, False, "INVENTED (captured)"),
    ("XII",   "",          9,  0.05, "rock", 2200, 480_000, 0.150, 152.0,  110,  10,  160, False, "INVENTED (captured, retrograde)"),
    ("XIII",  "",          3,  0.05, "rock", 2200, 520_000, 0.100, 165.0,  290, 330,  330, False, "INVENTED (captured, retrograde)"),
    ("XIV",   "",         11,  0.07, "rock", 2200, 560_000, 0.150, 141.0,   60, 170,   10, False, "INVENTED (captured, retrograde)"),
    ("XV",    "",          4,  0.05, "rock", 2200, 600_000, 0.080, 158.0,  160,  80,  120, False, "INVENTED (captured, retrograde)"),
]


class GeonosisSystem:
    """Container for the whole system plus convenience geometry."""

    # Earth-view geometry of Geonosis' orbit (INVENTED)
    sky_inclination = np.radians(55.0)   # orbit inclination to the sky plane
    sky_node = np.radians(120.0)         # position angle of ascending node
    mean_anomaly0 = np.radians(30.0)     # at t = 0 (= epoch below)
    peri_arg = np.radians(75.0)
    epoch_iso = "2031-01-01T00:00:00"    # t = 0 for all ephemerides

    def __init__(self):
        self.star = Star()
        self.planet = Planet()
        self.ring = Ring(inner=1.8, outer=2.9, tau=0.3, albedo=0.15,
                         particle_density=2500.0)
        # Planet orbit from Kepler III with the CANON period (DERIVED a)
        gm_star = G * (self.star.mass + self.planet.mass)
        a = (gm_star * (self.planet.orbital_period / (2 * np.pi))**2)**(1 / 3)
        self.planet_orbit = Orbit(a=a, e=self.planet.eccentricity, inc=0.0,
                                  node=0.0, peri=self.peri_arg,
                                  mean_anomaly0=self.mean_anomaly0, gm=gm_star)
        self.moons = []
        for row in _MOON_TABLE:
            (des, name, r_km, alb, surf, rho, a_km, e, inc, node, peri, m0,
             major, prov) = row
            m = Moon(des, name, r_km * 1e3, alb, surf, rho, a_km * 1e3, e,
                     inc, node, peri, m0, major, prov)
            m.orbit = Orbit(a=m.a, e=e, inc=np.radians(inc),
                            node=np.radians(node), peri=np.radians(peri),
                            mean_anomaly0=np.radians(m0),
                            gm=G * (self.planet.mass + m.mass))
            self.moons.append(m)

    # ------------------------------------------------------------------
    # Lookup and frames
    # ------------------------------------------------------------------
    def moon(self, key: str) -> Moon:
        for m in self.moons:
            if key in (m.name, m.designation, m.label):
                return m
        raise KeyError(key)

    @property
    def major_moons(self):
        return [m for m in self.moons if m.major]

    @property
    def eq_to_orbit(self) -> np.ndarray:
        """Rotation matrix: planet-equatorial frame -> orbital-plane frame."""
        p = self.planet
        return rot_z(p.spin_longitude) @ rot_x(p.obliquity)

    @property
    def orbit_to_sky(self) -> np.ndarray:
        """Orbital-plane frame -> Earth sky frame (x=E, y=N, z=to Earth)."""
        return rot_z(self.sky_node) @ rot_x(self.sky_inclination)

    def planet_position(self, t):
        """Ea -> Geonosis vector in the orbital-plane frame [m]."""
        return self.planet_orbit.position(t)

    def sun_direction_eq(self, t):
        """Unit vector Geonosis -> Ea in the planet equatorial frame."""
        r = -self.planet_position(t)
        r = r / np.linalg.norm(r, axis=0)
        return self.eq_to_orbit.T @ r

    def moon_position_eq(self, moon: Moon, t):
        return moon.orbit.position(t)

    # ------------------------------------------------------------------
    # Physical diagnostics
    # ------------------------------------------------------------------
    @property
    def semi_major_axis_au(self) -> float:
        return self.planet_orbit.a / AU

    @property
    def irradiance_earth_units(self) -> float:
        return (self.star.luminosity / L_SUN) / self.semi_major_axis_au**2

    def equilibrium_temperature(self, albedo=None) -> float:
        albedo = self.planet.bond_albedo if albedo is None else albedo
        return (self.star.teff * np.sqrt(self.star.radius / (2 * self.planet_orbit.a))
                * (1 - albedo)**0.25)

    @property
    def hill_radius(self) -> float:
        return self.planet_orbit.a * (1 - self.planet.eccentricity) * (
            self.planet.mass / (3 * self.star.mass))**(1 / 3)

    def roche_limit(self, satellite_density, rigid=False) -> float:
        coeff = 1.26 if rigid else 2.44
        return coeff * self.planet.radius * (self.planet.density / satellite_density)**(1 / 3)

    @property
    def synchronous_orbit(self) -> float:
        return (self.planet.gm * (self.planet.rotation_period / (2 * np.pi))**2)**(1 / 3)

    def stability_report(self) -> list[dict]:
        """Checks that the ring and every moon are dynamically sensible."""
        rows = []
        Rp = self.planet.radius
        roche_ring = self.roche_limit(self.ring.particle_density)
        rows.append(dict(
            body="Ring", check="outer edge inside fluid Roche limit",
            value=f"{self.ring.outer:.2f} Rp < {roche_ring / Rp:.2f} Rp",
            ok=self.ring.outer * Rp < roche_ring))
        for m in self.moons:
            roche = self.roche_limit(m.density, rigid=m.radius < 100e3)
            peri = m.a * (1 - m.e)
            apo = m.a * (1 + m.e)
            # prograde orbits stable to ~0.49 r_H, retrograde to ~0.7 r_H
            lim = (0.70 if m.inc_deg > 90 else 0.49) * self.hill_radius
            rows.append(dict(
                body=m.label, check="Roche < pericentre, apocentre < Hill limit",
                value=f"{roche / 1e3:,.0f} km < {peri / 1e3:,.0f} km ; "
                      f"{apo / 1e3:,.0f} km < {lim / 1e3:,.0f} km",
                ok=bool(peri > roche and apo < lim)))
        # Hill stability of neighbouring regular moons (Gladman 1993:
        # pairs separated by > 2*sqrt(3) mutual Hill radii are Hill-stable)
        regular = sorted([m for m in self.moons if m.inc_deg < 5], key=lambda m: m.a)
        for m1, m2 in zip(regular[:-1], regular[1:]):
            rh = ((m1.mass + m2.mass) / (3 * self.planet.mass))**(1 / 3) * (m1.a + m2.a) / 2
            sep = (m2.a * (1 - m2.e) - m1.a * (1 + m1.e)) / rh
            rows.append(dict(body=f"{m1.designation}-{m2.designation}",
                             check="separation > 2*sqrt(3) mutual Hill radii",
                             value=f"{sep:.1f} R_H,mutual", ok=bool(sep > 2 * np.sqrt(3))))
        return rows

    def summary_table(self) -> list[tuple[str, str, str]]:
        s, p, r = self.star, self.planet, self.ring
        return [
            ("Star", "Ea, G2V-like, 1 Msun, 1 Rsun, Teff 5770 K", "CANON name / INVENTED type"),
            ("Distance from Earth", f"{s.distance / PC:.2f} pc", "INVENTED (user choice)"),
            ("Sky position", f"RA {s.ra_deg / 15:.2f} h, Dec {s.dec_deg:+.1f} deg", "INVENTED"),
            ("Orbital position", "2nd planet of Ea", "CANON"),
            ("Diameter", f"{2 * p.radius / 1e3:,.0f} km", "CANON"),
            ("Rotation period", f"{p.rotation_period / HOUR:.0f} h", "CANON"),
            ("Orbital period", f"{p.orbital_period / DAY:.0f} d", "CANON"),
            ("Semi-major axis", f"{self.semi_major_axis_au:.3f} AU", "DERIVED (Kepler III)"),
            ("Irradiance", f"{self.irradiance_earth_units:.2f} x Earth", "DERIVED"),
            ("Equilibrium temperature", f"{self.equilibrium_temperature():.0f} K", "DERIVED"),
            ("Mass", f"{p.mass / M_EARTH:.2f} M_earth (rho = {p.density / 1e3:.1f} g/cc)", "INVENTED density"),
            ("Surface gravity", f"{p.surface_gravity / 9.81:.2f} g", "DERIVED"),
            ("Obliquity", f"{np.degrees(p.obliquity):.0f} deg", "INVENTED"),
            ("Ring", f"{r.inner}-{r.outer} Rp, tau = {r.tau}, albedo {r.albedo}", "CANON rings / INVENTED size"),
            ("Moons", "15 (4 major: Sun Fac, Bahryn, Poggle, Karina)", "CANON count, Bahryn CANON"),
            ("Hill radius", f"{self.hill_radius / 1e3:,.0f} km", "DERIVED"),
            ("Synchronous orbit", f"{self.synchronous_orbit / 1e3:,.0f} km", "DERIVED"),
        ]


def lambert_phase(alpha):
    """Lambert-sphere phase function, normalised to 1 at alpha = 0."""
    alpha = np.asarray(alpha)
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
