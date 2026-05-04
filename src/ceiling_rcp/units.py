"""Metric ↔ imperial display helpers.

World coordinates stay in metres throughout the codebase; these helpers
just turn a metres value into a display string. The frontend mirrors the
same logic (in app.js) so editor and PDF read identically.
"""
from __future__ import annotations

from math import gcd

UNIT_SYSTEMS = ("metric", "imperial")
DEFAULT_UNITS = "metric"

INCHES_PER_METRE = 39.3700787401574803


def _round_to_sixteenth(inches: float) -> tuple[int, int, int]:
    """Round ``inches`` to the nearest 1/16″ and return ``(feet, whole_inches,
    sixteenths)`` where ``sixteenths ∈ {0, 1, …, 15}`` and the inches
    portion has been carried into feet when it reaches 12.

    Always returns a non-negative tuple — the caller handles sign.
    """
    total_sixteenths = round(inches * 16.0)
    feet, rem = divmod(total_sixteenths, 12 * 16)
    whole_inches, sixteenths = divmod(rem, 16)
    return int(feet), int(whole_inches), int(sixteenths)


def _fraction(sixteenths: int) -> str:
    """Reduce ``sixteenths/16`` to lowest terms and format as a unicode-free
    fraction (``"5/8"``). Returns empty string for 0."""
    if sixteenths == 0:
        return ""
    g = gcd(sixteenths, 16)
    return f"{sixteenths // g}/{16 // g}"


def _imperial_length(metres: float) -> str:
    """Length as feet and inches with 1/16″ precision, signed.

    >>> _imperial_length(0)
    '0"'
    >>> _imperial_length(0.0254)            # 1"
    '1"'
    >>> _imperial_length(1.2954)            # 4'-3"
    '4\\'-3"'
    >>> _imperial_length(0.3048)            # 1' exactly
    "1'-0\""
    """
    sign = "-" if metres < 0 else ""
    inches = abs(metres) * INCHES_PER_METRE
    feet, whole, sixteenths = _round_to_sixteenth(inches)
    frac = _fraction(sixteenths)
    if feet == 0:
        # Sub-foot: print just inches, with fraction if any.
        if whole == 0 and frac == "":
            return f'{sign}0"'
        if whole == 0:
            return f'{sign}{frac}"'
        if frac:
            return f'{sign}{whole} {frac}"'
        return f'{sign}{whole}"'
    inches_part = f"{whole}"
    if frac:
        inches_part += f" {frac}"
    return f"{sign}{feet}'-{inches_part}\""


def _imperial_height_delta(metres: float) -> str:
    """Height-delta as inches (with feet only when ≥ 12″), signed with leading
    "+" / "−"."""
    if metres == 0.0:
        return '0"'
    sign = "−" if metres < 0 else "+"
    inches = abs(metres) * INCHES_PER_METRE
    feet, whole, sixteenths = _round_to_sixteenth(inches)
    frac = _fraction(sixteenths)
    if feet == 0:
        if whole == 0 and frac == "":
            return '0"'
        if whole == 0:
            return f'{sign}{frac}"'
        if frac:
            return f'{sign}{whole} {frac}"'
        return f'{sign}{whole}"'
    inches_part = f"{whole}"
    if frac:
        inches_part += f" {frac}"
    return f"{sign}{feet}'-{inches_part}\""


def format_length(metres: float, system: str) -> str:
    """Length as a human display string.

    - ``metric``: millimetres for < 1 m, metres with two decimals otherwise.
    - ``imperial``: feet and fractional inches at 1/16″ resolution.
    """
    if system == "imperial":
        return _imperial_length(float(metres))
    abs_m = abs(metres)
    if abs_m < 1.0:
        return f"{round(metres * 1000):.0f} mm"
    return f"{metres:.2f} m"


def format_height_delta(metres: float, system: str) -> str:
    """Signed height delta — leading ``+`` / ``−``, zero printed plain."""
    if system == "imperial":
        return _imperial_height_delta(float(metres))
    if metres == 0.0:
        return "0 mm"
    sign = "−" if metres < 0 else "+"
    return f"{sign}{round(abs(metres) * 1000):.0f} mm"


def format_short_length(metres: float, system: str) -> str:
    """Compact length used in the side panel meta column. Same as
    ``format_length`` for now; broken out so we can tune typography in
    one place if needed."""
    return format_length(metres, system)


__all__ = [
    "UNIT_SYSTEMS",
    "DEFAULT_UNITS",
    "format_length",
    "format_height_delta",
    "format_short_length",
]
