"""Geonosis system model and ELT (METIS / MICADO) imaging campaigns."""

import os

# ScopeSim's progress bars clutter batch runs; tqdm reads this at import time.
os.environ.setdefault("TQDM_DISABLE", "1")
