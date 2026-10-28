"""ScopeSim helpers: instrument set-up, a custom Geonosis atmosphere and PSF,
noiseless rate maps and detector parameters."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np                              # noqa: E402
import scopesim as sim                           # noqa: E402
from scopesim.effects import AtmosphericTERCurve, AnisocadoConstPSF  # noqa: E402

from .spectra import WAVE_UM, star_flux, blackbody_intensity, sky_emission_photlam_arcsec2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INST_PKGS = ROOT / "inst_pkgs"
_ARCSEC2_PER_SR = (180 / np.pi * 3600) ** 2


def quiet_logs():
    for name in list(logging.root.manager.loggerDict):
        if name.startswith(("astar", "httpx", "scopesim")):
            logging.getLogger(name).setLevel(logging.ERROR)


def setup(quiet=True):
    sim.rc.__config__["!SIM.file.local_packages_path"] = str(INST_PKGS)
    if quiet:
        quiet_logs()


def _element_index(opt, name):
    for k, el in enumerate(opt.optics_manager.optical_elements):
        if el.meta.get("name") == name:
            return k
    raise KeyError(name)


# --------------------------------------------------------------------------
# METIS (Earth, Armazones)
# --------------------------------------------------------------------------
def metis_train(mode="img_n", filter_name="N2", **properties):
    setup()
    props = {"!OBS.filter_name": filter_name}
    props.update(properties)
    cmd = sim.UserCommands(use_instrument="METIS", set_modes=[mode], properties=props)
    opt = sim.OpticalTrain(cmd)
    quiet_logs()
    return opt


# --------------------------------------------------------------------------
# MICADO on Geonosis
# --------------------------------------------------------------------------
def geonosis_atmosphere(system, t, airmass=1.2, dust_tau_1um=0.12,
                        sky_k_ab_mag=14.0, t_air=300.0):
    """AtmosphericTERCurve for the Geonosis sky (INVENTED model).

    * transmission: dust extinction tau(lambda) = tau_1um (lambda/1um)^-1 per
      airmass (Geonosis is a dusty desert world; no strong molecular bands
      since the atmosphere is thin and dry);
    * emission: sunlight-coloured "ringshine" scattered by airborne dust,
      normalised to ``sky_k_ab_mag`` AB mag/arcsec^2 at 2.15 um, plus thermal
      emission of the warm (300 K) air with emissivity 1 - T.
    """
    w = WAVE_UM
    trans = np.exp(-dust_tau_1um * w**-1.0 * airmass)
    r = np.linalg.norm(system.planet_position(t))
    sun = star_flux(system, r, w)
    ringshine = sky_emission_photlam_arcsec2(w, sun, sky_k_ab_mag)
    thermal = (1 - trans) * blackbody_intensity(t_air, w) / _ARCSEC2_PER_SR
    emission = (ringshine + thermal) * 1e8          # PHOTLAM -> ph s-1 m-2 um-1
    return AtmosphericTERCurve(
        name="geonosis_atmosphere",
        array_dict={"wavelength": w, "transmission": trans, "emission": emission},
        wavelength_unit="um", emission_unit="ph s-1 m-2 um-1 arcsec-2",
        action="transmission", position=0)


def micado_train(fw2="Ks", fw1="open", pupil="open", mode="IMG_4mas",
                 window=1024, system=None, t=0.0, airmass=1.2,
                 strehl_ks=0.25, atmosphere_kwargs=None, dit=1.0, ndit=1):
    """MICADO SCAO optical train relocated to Geonosis.

    The Armazones skycalc atmosphere is switched off and replaced by
    :func:`geonosis_atmosphere`; the default Armazones SCAO PSF is replaced by
    an AnisoCADO PSF with the requested K-band Strehl (Geonosis seeing is
    dusty and worse than Armazones; INVENTED value)."""
    setup()
    props = {
        "!OBS.filter_name_fw1": fw1, "!OBS.filter_name_fw2": fw2,
        "!OBS.filter_name_pupil": pupil,
        "!OBS.airmass": airmass, "!OBS.dit": dit, "!OBS.ndit": ndit,
        "!DET.width": window, "!DET.height": window,
    }
    cmd = sim.UserCommands(use_instrument="MICADO", set_modes=["SCAO", mode],
                           properties=props)
    opt = sim.OpticalTrain(cmd)
    if system is not None:
        opt["skycalc_atmosphere"].include = False
        k = _element_index(opt, "armazones")
        opt.optics_manager.add_effect(
            geonosis_atmosphere(system, t, airmass=airmass, **(atmosphere_kwargs or {})), ext=k)
    if strehl_ks is not None:
        opt["relay_psf"].include = False
        k = _element_index(opt, "default_ro")
        psf = AnisocadoConstPSF(
            name="geonosis_scao_psf",
            filename=str(INST_PKGS / "MICADO" / "MICADO_AnisoCADO_rms_map.fits"),
            strehl=strehl_ks, wavelength=2.15, psf_side_length=512,
            convolve_mode="same", cmds=opt.cmds)
        opt.optics_manager.add_effect(psf, ext=k)
    quiet_logs()
    return opt


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def rate_map(opt, source) -> np.ndarray:
    """Noiseless detected photon rate [ph/s/pixel] on the image plane."""
    opt.observe(source, update=True)
    return np.array(opt.image_planes[0].data, dtype=float)


def detector_params(opt) -> dict:
    """Detector constants (read noise [e-], dark [e-/s], full well [e-],
    gain [e-/ADU], minimum DIT [s]) for the current detector readout mode."""
    for eff in opt.optics_manager.all_effects:       # apply the readout mode
        if type(eff).__name__ == "DetectorModePropertiesSetter":
            mode = opt.cmds["!OBS.detector_readout_mode"]
            for key, val in eff.mode_properties[mode].items():
                opt.cmds[key] = val
    out = {}
    for key in ("readout_noise", "dark_current", "full_well", "gain", "mindit"):
        try:
            out[key] = float(opt.cmds[f"!DET.{key}"])
        except Exception:
            out[key] = None
    for eff in opt.optics_manager.all_effects:       # MICADO keeps them in effects
        cls = type(eff).__name__
        if out["readout_noise"] is None and "ReadoutNoise" in cls and eff.include:
            out["readout_noise"] = float(eff.meta["noise_std"])
        if out["dark_current"] is None and cls == "DarkCurrent" and eff.include:
            out["dark_current"] = float(eff.meta["value"])
    return out
