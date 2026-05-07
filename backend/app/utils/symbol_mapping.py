"""Symbol mapping between StockPulse format and Qlib format.

StockPulse uses exchange-suffixed symbols (e.g., 600000.SS, 000001.SZ, 0700.HK).
Qlib uses prefix-based symbols (e.g., SH600000, SZ000001, HK00700).

This module provides bidirectional conversion.
"""
import logging

logger = logging.getLogger(__name__)

# StockPulse suffix -> Qlib prefix
_SP_TO_QLIB_SUFFIX = {
    ".SS": "SH",
    ".SZ": "SZ",
    ".HK": "HK",
}

# Qlib prefix -> StockPulse suffix
_QLIB_TO_SP_PREFIX = {
    "SH": ".SS",
    "SZ": ".SZ",
    "HK": ".HK",
}

# Metals mapping
_METAL_SP_TO_QLIB = {
    "GC=F": "GCF",
    "SI=F": "SIF",
    "PL=F": "PLF",
    "PA=F": "PAF",
}
_METAL_QLIB_TO_SP = {v: k for k, v in _METAL_SP_TO_QLIB.items()}


def stockpulse_to_qlib(symbol: str, market: str = "") -> str:
    """Convert StockPulse symbol to Qlib format.

    Examples:
        600000.SS -> SH600000
        000001.SZ -> SZ000001
        0700.HK   -> HK00700
        AAPL      -> AAPL (unchanged for US)
        GC=F      -> GCF
    """
    # Metal symbols
    if symbol in _METAL_SP_TO_QLIB:
        return _METAL_SP_TO_QLIB[symbol]

    # Check for exchange suffix
    for suffix, prefix in _SP_TO_QLIB_SUFFIX.items():
        if symbol.upper().endswith(suffix):
            code = symbol[: -len(suffix)]
            return f"{prefix}{code}"

    # US symbols pass through unchanged
    return symbol


def qlib_to_stockpulse(symbol: str, market: str = "") -> str:
    """Convert Qlib symbol to StockPulse format.

    Examples:
        SH600000 -> 600000.SS
        SZ000001 -> 000001.SZ
        HK00700  -> 0700.HK
        AAPL     -> AAPL (unchanged for US)
        GCF      -> GC=F
    """
    # Metal symbols
    if symbol in _METAL_QLIB_TO_SP:
        return _METAL_QLIB_TO_SP[symbol]

    # Check for Qlib prefix
    for prefix, suffix in _QLIB_TO_SP_PREFIX.items():
        if symbol.startswith(prefix) and len(symbol) > len(prefix):
            code = symbol[len(prefix) :]
            # Verify remaining part is numeric (to avoid false matches like "SHOP" -> "HOP.SZ")
            if code.isdigit():
                return f"{code}{suffix}"

    # US symbols pass through unchanged
    return symbol


def normalize_symbol_for_qlib(symbol: str, market: str) -> str:
    """Normalize any symbol format for Qlib usage.

    Handles: bare codes (600000), suffixed (600000.SS), already-Qlib (SH600000)
    """
    # Already in Qlib format?
    for prefix in _QLIB_TO_SP_PREFIX:
        if symbol.startswith(prefix) and symbol[len(prefix) :].isdigit():
            return symbol

    # Has StockPulse suffix?
    for suffix in _SP_TO_QLIB_SUFFIX:
        if symbol.upper().endswith(suffix):
            return stockpulse_to_qlib(symbol)

    # Bare numeric code -- infer exchange from market
    if symbol.isdigit() and market in ("sh", "sz", "cn"):
        if symbol.startswith(("6", "9")):
            return f"SH{symbol}"
        elif symbol.startswith(("0", "2", "3")):
            return f"SZ{symbol}"

    # US / metal -- pass through
    return stockpulse_to_qlib(symbol, market)
