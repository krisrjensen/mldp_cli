#!/usr/bin/env python3
"""
Filename: spectral_features.py
Author(s): Kristophor Jensen
Date Created: 20251105_000000
Date Revised: 20251105_000000
File version: 1.0.0.1
Description: Spectral feature extraction functions for MLDP
             Implements SNR, mean PSD, slope, and SFM for different frequency bands
             All frequency bands are RELATIVE to max usable frequency (Nyquist)
"""

import numpy as np
from typing import Tuple


def _get_frequency_band_bins(segment_length: int, band_name: str) -> Tuple[int, int]:
    """
    Calculate bin range for a frequency band relative to maximum usable bins.

    All bands are defined relative to max_bin (Nyquist), NOT absolute frequencies.
    This ensures correct behavior across all decimation levels.

    Args:
        segment_length: Length of the segment (after decimation)
        band_name: Name of the band ('ultra_high', 'full', 'high', 'mid', 'low')

    Returns:
        Tuple of (start_bin, end_bin) inclusive

    Examples:
        For segment_length=8192 (no decimation):
            max_bin = 4096
            ultra_high: bins 820 to 4096 (0.2 to 1.0)
            full: bins 0 to 4096 (0.0 to 1.0)

        For segment_length=64 (decimation ÷128):
            max_bin = 32
            ultra_high: bins 6 to 32 (0.2 to 1.0)
            full: bins 0 to 32 (0.0 to 1.0)
    """
    # Maximum usable bin (Nyquist frequency)
    max_bin = segment_length // 2

    if band_name == 'ultra_high':
        # Top 80% of spectrum: 0.2 × max_bin to max_bin
        start_bin = int(0.2 * max_bin)
        end_bin = max_bin
    elif band_name == 'full':
        # Entire usable spectrum: 0 to max_bin
        start_bin = 0
        end_bin = max_bin
    elif band_name == 'high':
        # Upper half: 0.5 × max_bin to max_bin
        start_bin = int(0.5 * max_bin)
        end_bin = max_bin
    elif band_name == 'mid':
        # Middle 50%: 0.25 × max_bin to 0.75 × max_bin
        start_bin = int(0.25 * max_bin)
        end_bin = int(0.75 * max_bin)
    elif band_name == 'low':
        # Lower half: 0 to 0.5 × max_bin
        start_bin = 0
        end_bin = int(0.5 * max_bin)
    else:
        raise ValueError(f"Unknown frequency band: {band_name}")

    return (start_bin, end_bin)


def _compute_psd(signal: np.ndarray) -> np.ndarray:
    """
    Compute power spectral density using FFT.

    Args:
        signal: Input time-domain signal

    Returns:
        Power spectral density (one-sided spectrum, positive frequencies only)
    """
    # Compute FFT
    fft_result = np.fft.rfft(signal)

    # Compute power spectral density
    psd = np.abs(fft_result) ** 2

    # Normalize by segment length
    psd = psd / len(signal)

    return psd


def _compute_snr_in_band(signal: np.ndarray, start_bin: int, end_bin: int) -> float:
    """
    Compute signal-to-noise ratio within a frequency band.

    SNR is defined as the ratio of signal power to noise power.
    Signal power = power in the band
    Noise power = estimated from high-frequency bins or variation

    Args:
        signal: Input time-domain signal
        start_bin: Starting frequency bin (inclusive)
        end_bin: Ending frequency bin (inclusive)

    Returns:
        SNR in dB
    """
    psd = _compute_psd(signal)

    # Ensure valid bin range
    start_bin = max(0, start_bin)
    end_bin = min(len(psd) - 1, end_bin)

    # Extract band power
    band_psd = psd[start_bin:end_bin+1]
    signal_power = np.sum(band_psd)

    # Estimate noise power from the variation in the band
    # Use standard deviation of PSD as noise estimate
    noise_power = np.std(band_psd) ** 2

    # Avoid division by zero
    if noise_power < 1e-12:
        noise_power = 1e-12

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return float(snr_db)


def _compute_mean_psd_in_band(signal: np.ndarray, start_bin: int, end_bin: int) -> float:
    """
    Compute mean power spectral density within a frequency band.

    Args:
        signal: Input time-domain signal
        start_bin: Starting frequency bin (inclusive)
        end_bin: Ending frequency bin (inclusive)

    Returns:
        Mean PSD in the band
    """
    psd = _compute_psd(signal)

    # Ensure valid bin range
    start_bin = max(0, start_bin)
    end_bin = min(len(psd) - 1, end_bin)

    # Extract band and compute mean
    band_psd = psd[start_bin:end_bin+1]
    mean_psd = np.mean(band_psd)

    return float(mean_psd)


def _compute_slope_in_band(signal: np.ndarray, start_bin: int, end_bin: int) -> float:
    """
    Compute spectral slope (pink noise ratio) within a frequency band.

    Slope is computed as the linear regression slope of log(PSD) vs log(frequency).
    Pink noise has slope ≈ -1, white noise has slope ≈ 0.

    Args:
        signal: Input time-domain signal
        start_bin: Starting frequency bin (inclusive)
        end_bin: Ending frequency bin (inclusive)

    Returns:
        Spectral slope
    """
    psd = _compute_psd(signal)

    # Ensure valid bin range
    start_bin = max(1, start_bin)  # Avoid log(0) for frequency
    end_bin = min(len(psd) - 1, end_bin)

    # Extract band
    band_psd = psd[start_bin:end_bin+1]

    # Create frequency bins (1, 2, 3, ... for relative indexing)
    freq_bins = np.arange(1, len(band_psd) + 1)

    # Avoid log(0) by adding small epsilon
    band_psd = band_psd + 1e-12

    # Compute log-log linear regression
    log_freq = np.log10(freq_bins)
    log_psd = np.log10(band_psd)

    # Fit linear regression: log_psd = slope * log_freq + intercept
    slope = np.polyfit(log_freq, log_psd, 1)[0]

    return float(slope)


def _compute_sfm_in_band(signal: np.ndarray, start_bin: int, end_bin: int) -> float:
    """
    Compute spectral flatness measure (SFM) within a frequency band.

    SFM = geometric_mean(PSD) / arithmetic_mean(PSD)

    SFM ranges from 0 to 1:
    - Near 0: tonal (peaked spectrum)
    - Near 1: noise-like (flat spectrum)

    Args:
        signal: Input time-domain signal
        start_bin: Starting frequency bin (inclusive)
        end_bin: Ending frequency bin (inclusive)

    Returns:
        Spectral flatness measure (0 to 1)
    """
    psd = _compute_psd(signal)

    # Ensure valid bin range
    start_bin = max(0, start_bin)
    end_bin = min(len(psd) - 1, end_bin)

    # Extract band
    band_psd = psd[start_bin:end_bin+1]

    # Avoid log(0) by adding small epsilon
    band_psd = band_psd + 1e-12

    # Geometric mean
    geometric_mean = np.exp(np.mean(np.log(band_psd)))

    # Arithmetic mean
    arithmetic_mean = np.mean(band_psd)

    # SFM
    sfm = geometric_mean / arithmetic_mean

    return float(sfm)


# ==============================================================================
# PUBLIC API: Feature extraction functions
# These functions are called by the feature extractor via globals()
# ==============================================================================

# ULTRA-HIGH BAND (0.2 × max_bin to max_bin)

def v_ultra_high_snr(signal: np.ndarray) -> float:
    """Voltage ultra-high band SNR"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_snr_in_band(signal, start_bin, end_bin)

def c_ultra_high_snr(signal: np.ndarray) -> float:
    """Current ultra-high band SNR"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_snr_in_band(signal, start_bin, end_bin)

def v_ultra_high_mean_psd(signal: np.ndarray) -> float:
    """Voltage ultra-high band mean PSD"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_mean_psd_in_band(signal, start_bin, end_bin)

def c_ultra_high_mean_psd(signal: np.ndarray) -> float:
    """Current ultra-high band mean PSD"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_mean_psd_in_band(signal, start_bin, end_bin)

def v_ultra_high_slope(signal: np.ndarray) -> float:
    """Voltage ultra-high band spectral slope"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_slope_in_band(signal, start_bin, end_bin)

def c_ultra_high_slope(signal: np.ndarray) -> float:
    """Current ultra-high band spectral slope"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_slope_in_band(signal, start_bin, end_bin)

def v_ultra_high_sfm(signal: np.ndarray) -> float:
    """Voltage ultra-high band spectral flatness measure"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_sfm_in_band(signal, start_bin, end_bin)

def c_ultra_high_sfm(signal: np.ndarray) -> float:
    """Current ultra-high band spectral flatness measure"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'ultra_high')
    return _compute_sfm_in_band(signal, start_bin, end_bin)


# FULL BAND (0 to max_bin)

def v_full_snr(signal: np.ndarray) -> float:
    """Voltage full band SNR"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_snr_in_band(signal, start_bin, end_bin)

def c_full_snr(signal: np.ndarray) -> float:
    """Current full band SNR"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_snr_in_band(signal, start_bin, end_bin)

def v_full_mean_psd(signal: np.ndarray) -> float:
    """Voltage full band mean PSD"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_mean_psd_in_band(signal, start_bin, end_bin)

def c_full_mean_psd(signal: np.ndarray) -> float:
    """Current full band mean PSD"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_mean_psd_in_band(signal, start_bin, end_bin)

def v_full_slope(signal: np.ndarray) -> float:
    """Voltage full band spectral slope"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_slope_in_band(signal, start_bin, end_bin)

def c_full_slope(signal: np.ndarray) -> float:
    """Current full band spectral slope"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_slope_in_band(signal, start_bin, end_bin)

def v_full_sfm(signal: np.ndarray) -> float:
    """Voltage full band spectral flatness measure"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_sfm_in_band(signal, start_bin, end_bin)

def c_full_sfm(signal: np.ndarray) -> float:
    """Current full band spectral flatness measure"""
    start_bin, end_bin = _get_frequency_band_bins(len(signal), 'full')
    return _compute_sfm_in_band(signal, start_bin, end_bin)
