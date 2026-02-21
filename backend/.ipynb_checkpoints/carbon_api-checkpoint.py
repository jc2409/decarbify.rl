"""
Live carbon intensity fetcher for GreenDispatch.

Uses the Electricity Maps API (https://api.electricitymap.org).
Falls back to deterministic mock data with time-of-day variation when
the ELECTRICITY_MAPS_TOKEN environment variable is not set.
"""

import os
import random
import time
from typing import Optional

# ── Zone mappings ──────────────────────────────────────────────────────────────

# Internal DC id → human-readable display key (used as dict key in return value)
_DC_DISPLAY: dict[str, str] = {
    "DC1": "DC1 (US-CA)",
    "DC2": "DC2 (Germany)",
    "DC3": "DC3 (Chile)",
    "DC4": "DC4 (Singapore)",
    "DC5": "DC5 (Australia)",
}

# Internal DC id → Electricity Maps zone code
_DC_ZONES: dict[str, str] = {
    "DC1": "US-CAL-CISO",
    "DC2": "DE",
    "DC3": "CL-SEN",
    "DC4": "SG",
    "DC5": "AU-NSW",
}

# Typical baseline carbon intensities (gCO₂/kWh) informed by real grid data.
# California: moderate renewables + gas; Germany: coal/renewables mix;
# Chile: hydro + coal; Singapore: almost all natural gas; NSW: coal-heavy.
_MOCK_BASE_CI: dict[str, float] = {
    "DC1": 230.0,
    "DC2": 340.0,
    "DC3": 195.0,
    "DC4": 455.0,
    "DC5": 630.0,
}

_API_BASE = "https://api.electricitymap.org/v3/carbon-intensity/latest"


# ── Private helpers ────────────────────────────────────────────────────────────

def _fetch_zone(zone: str, token: str) -> Optional[float]:
    """
    Call the Electricity Maps API for a single zone.
    Returns the carbonIntensity float, or None on any failure.
    """
    try:
        import requests  # soft dependency — only needed for live mode
        resp = requests.get(
            _API_BASE,
            params={"zone": zone},
            headers={"auth-token": token},
            timeout=8,
        )
        resp.raise_for_status()
        return float(resp.json()["carbonIntensity"])
    except Exception:
        return None


def _mock_carbon_intensity() -> dict[str, float]:
    """
    Generate realistic mock CI values with:
    - ±15 % random noise around the baseline
    - A solar dip (≈25 % reduction) for sun-rich grids during solar hours (10–16 UTC)
    """
    utc_hour = time.gmtime().tm_hour
    is_solar = 10 <= utc_hour <= 16

    result: dict[str, float] = {}
    for dc_id, base_ci in _MOCK_BASE_CI.items():
        noise = random.uniform(0.85, 1.15)
        # California and Chile benefit most from daytime solar
        solar_adj = 0.75 if (is_solar and dc_id in ("DC1", "DC3")) else 1.0
        result[_DC_DISPLAY[dc_id]] = round(base_ci * noise * solar_adj, 1)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def get_live_carbon_intensity() -> tuple[dict[str, float], bool]:
    """
    Fetch the current carbon intensity for all 5 GreenDispatch datacenters.

    Checks for ELECTRICITY_MAPS_TOKEN in the environment.  If present, queries
    the Electricity Maps API for each zone; any zone that fails falls back to
    mock data for that zone only.  If the token is absent, all zones use mock
    data immediately.

    Returns
    -------
    ci_data : dict[str, float]
        Maps display label (e.g. ``"DC1 (US-CA)"``) to carbon intensity in
        gCO₂/kWh, ordered DC1 → DC5.
    is_live : bool
        ``True`` if at least one zone was successfully fetched from the real
        API, ``False`` if entirely mock.
    """
    token = os.environ.get("ELECTRICITY_MAPS_TOKEN", "").strip()

    if not token:
        return _mock_carbon_intensity(), False

    mock_fallback = _mock_carbon_intensity()
    results: dict[str, float] = {}
    any_live = False

    for dc_id, zone in _DC_ZONES.items():
        label = _DC_DISPLAY[dc_id]
        ci = _fetch_zone(zone, token)
        if ci is not None:
            results[label] = ci
            any_live = True
        else:
            # Per-zone fallback so a partial outage doesn't kill the whole panel.
            results[label] = mock_fallback[label]

    return results, any_live
