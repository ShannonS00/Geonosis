"""High-contrast imaging toolkit for the METIS campaign.

IRDB's METIS package has no coronagraphic imaging mode, so we model METIS'
ring-apodized vortex coronagraph (RAVC) physically with Fourier optics:

    ELT pupil x ring apodizer -> FFT -> charge-2 vortex -> IFFT -> Lyot stop
    -> matrix Fourier transform onto the METIS detector pixel grid.

Wavefront errors (INVENTED but representative):
  * static NCPA (fixed in the pupil frame -> removed by ADI);
  * quasi-static NCPA drifting during the night (the ADI-limiting term);
  * fast SCAO residuals (averaged within a frame -> smooth halo).

ScopeSim supplies the photometry (star / planet count rates through ELT +
METIS + filter + QE), the thermal background and the detector constants;
this module turns them into a pupil-tracking ADI frame cube, then reduces it
with classical ADI and PCA-ADI.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

D_ELT = 39.0          # m, outer diameter (circular approximation)
OBSC = 0.28           # central obscuration ratio
SPIDER_W = 0.54       # m, spider width (6 spiders)
RAD2MAS = 180 / np.pi * 3600e3


# --------------------------------------------------------------------------
# Pupil, apodizer, Lyot stop
# --------------------------------------------------------------------------
def pupil_grid(n_pup=256):
    dx = D_ELT / n_pup
    c = (np.arange(n_pup) - (n_pup - 1) / 2) * dx
    X, Y = np.meshgrid(c, c)
    return X, Y, dx


def spiders(X, Y, width, n=6, offset_deg=0.0):
    mask = np.ones_like(X, dtype=bool)
    for k in range(n):
        a = np.radians(offset_deg + 360.0 * k / n)
        # distance from a ray through the centre at angle a (half-line)
        d = np.abs(-np.sin(a) * X + np.cos(a) * Y)
        along = np.cos(a) * X + np.sin(a) * Y
        mask &= ~((d < width / 2) & (along > 0))
    return mask


def elt_pupil(X, Y):
    r = np.hypot(X, Y) / (D_ELT / 2)
    return ((r <= 1) & (r >= OBSC) & spiders(X, Y, SPIDER_W)).astype(float)


class RAVC:
    """Ring-apodized vortex coronagraph (charge 2) for the ELT pupil."""

    def __init__(self, n_pup=256, pad=8, ring_radius=None, ring_t=None):
        self.n_pup, self.pad = n_pup, pad
        self.X, self.Y, self.dx = pupil_grid(n_pup)
        self.pupil = elt_pupil(self.X, self.Y)
        self.r = np.hypot(self.X, self.Y) / (D_ELT / 2)
        n = n_pup * pad
        f = np.fft.fftfreq(n)
        FX, FY = np.meshgrid(f, f)
        self.vortex = np.exp(2j * np.arctan2(FY, FX))
        self.vortex[0, 0] = 0.0
        if ring_radius is None:
            ring_radius = self.optimise_ring()
        self._set_ring(ring_radius)

    def _set_ring(self, R1):
        """For a charge-2 vortex a uniform disc of radius a is mapped to
        (a/r)^2 outside a.  Cancelling the obscuration's contribution inside
        the Lyot stop (R1 < r < 1) requires the outer pupil (r > R1) to be
        attenuated to t = 1 - R0^2/R1^2 (exact RAVC condition)."""
        self.ring_radius = R1
        self.ring_t = 1 - OBSC**2 / R1**2
        self.apodizer = np.where(self.r >= R1, self.ring_t, 1.0)
        self.lyot = ((self.r <= 0.96) & (self.r >= R1 * 1.02)
                     & spiders(self.X, self.Y, 2.0 * SPIDER_W)).astype(float)

    # --- core propagation -------------------------------------------------
    def to_lyot(self, field, coronagraph=True):
        n = self.n_pup * self.pad
        big = np.zeros((n, n), complex)
        big[:self.n_pup, :self.n_pup] = field
        if not coronagraph:
            return field
        F = np.fft.fft2(big)
        F *= self.vortex
        L = np.fft.ifft2(F)[:self.n_pup, :self.n_pup]
        return L

    def optimise_ring(self, radii=np.linspace(0.36, 0.64, 15), wavelength=11.2e-6):
        """Pick the ring radius R1 that maximises off-axis (planet) peak
        throughput at 10 lambda/D; the transmission follows analytically."""
        best, best_peak = None, -1
        zero = np.zeros((self.n_pup, self.n_pup))
        lod = wavelength / D_ELT * RAD2MAS
        for R1 in radii:
            self._set_ring(R1)
            img = self.image(zero, wavelength, lod / 4, 33, tilt_mas=(10 * lod, 0),
                             coronagraph=True, center_mas=(10 * lod, 0))
            if img.max() > best_peak:
                best, best_peak = R1, img.max()
        return best

    # --- images -------------------------------------------------------------
    def image(self, phase, wavelength, pixscale_mas, npix, coronagraph=True,
              tilt_mas=(0.0, 0.0), amplitude=None, center_mas=(0.0, 0.0)):
        """Focal-plane intensity, normalised so that the full unapodized
        pupil without coronagraph integrates to 1 (fraction of the light
        entering the telescope that lands in each pixel)."""
        A = self.pupil * (self.apodizer if coronagraph else 1.0)
        if amplitude is not None:
            A = A * amplitude
        ph = phase.copy()
        if tilt_mas != (0.0, 0.0):
            k = 2 * np.pi / wavelength
            ph = ph + k * (self.X * tilt_mas[0] + self.Y * tilt_mas[1]) / RAD2MAS
        E = A * np.exp(1j * ph)
        E_l = self.to_lyot(E, coronagraph=coronagraph)
        E_l = E_l * (self.lyot if coronagraph else 1.0)
        return self._mft(E_l, wavelength, pixscale_mas, npix, center_mas)

    def _mft(self, E, wavelength, pixscale_mas, npix, center_mas=(0.0, 0.0)):
        x = (np.arange(self.n_pup) - (self.n_pup - 1) / 2) * self.dx
        th = (np.arange(npix) - npix // 2) * pixscale_mas
        Mx = np.exp(-2j * np.pi * np.outer(th + center_mas[0], x) / RAD2MAS / wavelength)
        My = np.exp(-2j * np.pi * np.outer(th + center_mas[1], x) / RAD2MAS / wavelength)
        F = My @ E @ Mx.T
        omega = (pixscale_mas / RAD2MAS) ** 2
        norm = self.dx**2 * omega / (wavelength**2 * np.sum(self.pupil**2))
        return np.abs(F) ** 2 * norm


# --------------------------------------------------------------------------
# Phase screens
# --------------------------------------------------------------------------
def phase_screen(n, dx, rms_m, wavelength, psd="ncpa", f_ctrl=None, rng=None):
    """Random phase screen [rad] with a given rms optical path [m]."""
    rng = np.random.default_rng() if rng is None else rng
    f = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(f, f)
    F = np.hypot(FX, FY)
    F[0, 0] = f[1]
    if psd == "ncpa":
        P = F ** -2.0
    elif psd == "ao":
        fc = f_ctrl
        P = np.where(F < fc, (F / fc) ** (-2 / 3), (F / fc) ** (-11 / 3))
    else:
        raise ValueError(psd)
    P[0, 0] = 0
    noise = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    scr = np.real(np.fft.ifft2(noise * np.sqrt(P)))
    scr -= scr.mean()
    scr *= rms_m / scr.std()
    return 2 * np.pi * scr / wavelength


class WavefrontModel:
    """Static + quasi-static (AR(1) drift) + fast AO residual wavefronts.
    Screens are stored as optical path [m] so they work at any wavelength."""

    def __init__(self, n_pup, dx, static_nm=40.0, quasi_nm=12.0,
                 quasi_tau_min=40.0, ao_nm=170.0, n_act=75, seed=1):
        self.rng = np.random.default_rng(seed)
        self.n, self.dx = n_pup, dx
        self.static = phase_screen(n_pup, dx, static_nm * 1e-9, 2 * np.pi, rng=self.rng)
        self.quasi_nm, self.tau = quasi_nm, quasi_tau_min
        self.quasi = phase_screen(n_pup, dx, quasi_nm * 1e-9, 2 * np.pi, rng=self.rng)
        self.ao_nm = ao_nm
        self.f_ctrl = n_act / (2 * D_ELT)

    def step(self, dt_min):
        rho = np.exp(-dt_min / self.tau)
        new = phase_screen(self.n, self.dx, self.quasi_nm * 1e-9, 2 * np.pi, rng=self.rng)
        self.quasi = rho * self.quasi + np.sqrt(1 - rho**2) * new

    def opd(self):
        """Current slow optical path difference [m]."""
        return self.static + self.quasi

    def fast_screen(self):
        return phase_screen(self.n, self.dx, self.ao_nm * 1e-9, 2 * np.pi,
                            psd="ao", f_ctrl=self.f_ctrl, rng=self.rng)


def smooth_ao_halo(cor: RAVC, wf: WavefrontModel, wavelength, pixscale_mas, npix,
                   n_real=24, coronagraph=True):
    """Average coronagraphic image over fast AO residual realisations minus the
    coherent (Strehl-scaled) part -> the incoherent smooth halo."""
    acc = np.zeros((npix, npix))
    for _ in range(n_real):
        ph = 2 * np.pi * wf.fast_screen() / wavelength
        acc += cor.image(ph, wavelength, pixscale_mas, npix, coronagraph=coronagraph)
    acc /= n_real
    strehl = np.exp(-(2 * np.pi * wf.ao_nm * 1e-9 / wavelength) ** 2)
    coh = strehl * cor.image(np.zeros((cor.n_pup, cor.n_pup)), wavelength,
                             pixscale_mas, npix, coronagraph=coronagraph)
    return np.clip(acc - coh, 0, None), strehl


# --------------------------------------------------------------------------
# ADI geometry and reduction
# --------------------------------------------------------------------------
def rotate_frames(cube, angles_deg):
    """Rotate each frame counter-clockwise by its angle (x = column, y = row,
    displayed with origin='lower').  ndimage.rotate turns clockwise in this
    convention, hence the sign flip."""
    return np.array([ndimage.rotate(f, -a, reshape=False, order=1) for f, a in zip(cube, angles_deg)])


def shift_psf(psf, dx_pix, dy_pix):
    return ndimage.shift(psf, (dy_pix, dx_pix), order=3, mode="constant")


def cadi(cube, pa_deg):
    """Classical ADI: subtract the median frame, derotate, mean-combine."""
    ref = np.median(cube, axis=0)
    res = cube - ref
    return np.mean(rotate_frames(res, -np.asarray(pa_deg)), axis=0)


def pca_adi(cube, pa_deg, ncomp=10, mask_inner_pix=0, annulus=None):
    """Full-frame PCA-ADI (SVD of the mean-subtracted frame matrix)."""
    nf, ny, nx = cube.shape
    yy, xx = np.indices((ny, nx))
    r = np.hypot(xx - nx // 2, yy - ny // 2)
    sel = r >= mask_inner_pix
    if annulus is not None:
        sel &= (r >= annulus[0]) & (r <= annulus[1])
    M = cube[:, sel]
    M = M - M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt[:ncomp]
    recon = (M @ V.T) @ V
    res = np.zeros_like(cube)
    res[:, sel] = M - recon
    return np.mean(rotate_frames(res, -np.asarray(pa_deg)), axis=0)


def aperture_sum(img, x, y, r):
    yy, xx = np.indices(img.shape)
    return img[(xx - x) ** 2 + (yy - y) ** 2 <= r**2].sum()


def snr_mawet(img, x, y, fwhm, cx=None, cy=None, exclude=1):
    """S/N with the small-sample t-test of Mawet et al. (2014): aperture flux
    vs. the population of non-overlapping apertures at the same radius."""
    ny, nx = img.shape
    cx = nx // 2 if cx is None else cx
    cy = ny // 2 if cy is None else cy
    rad = np.hypot(x - cx, y - cy)
    ang0 = np.arctan2(y - cy, x - cx)
    n_ap = int(np.floor(2 * np.pi * rad / fwhm))
    angs = ang0 + 2 * np.pi * np.arange(n_ap) / n_ap
    fl = np.array([aperture_sum(img, cx + rad * np.cos(a), cy + rad * np.sin(a), fwhm / 2) for a in angs])
    sig = fl[0]
    bg = fl[1 + exclude:len(fl) - exclude] if exclude else fl[1:]
    n2 = len(bg)
    return (sig - bg.mean()) / (bg.std(ddof=1) * np.sqrt(1 + 1 / n2)), sig


def contrast_curve(img, fwhm, star_peak_ref, throughput_fn, radii_pix, cx=None, cy=None):
    """5-sigma contrast vs radius (Mawet 2014 small-sample corrected),
    normalised to the off-axis PSF aperture flux ``star_peak_ref`` and
    corrected by the algorithmic throughput ``throughput_fn(r)``."""
    from scipy import stats
    ny, nx = img.shape
    cx = nx // 2 if cx is None else cx
    cy = ny // 2 if cy is None else cy
    out = []
    for rad in radii_pix:
        n_ap = int(np.floor(2 * np.pi * rad / fwhm))
        if n_ap < 3:
            out.append(np.nan)
            continue
        angs = 2 * np.pi * np.arange(n_ap) / n_ap
        fl = np.array([aperture_sum(img, cx + rad * np.cos(a), cy + rad * np.sin(a), fwhm / 2) for a in angs])
        tau = stats.t.ppf(1 - 2.87e-7, n_ap - 1)        # 5-sigma Gaussian false-alarm
        noise = fl.std(ddof=1) * np.sqrt(1 + 1 / n_ap)
        out.append(tau * noise / (star_peak_ref * throughput_fn(rad)))
    return np.array(out)
