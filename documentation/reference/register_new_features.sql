-- ============================================================================
-- New Feature Functions Registration Script
-- ============================================================================
-- Filename: register_new_features.sql
-- Author: Kristophor Jensen
-- Date Created: 2025-10-29
-- Description: Registers 64 new feature functions for arc detection
--
-- Features by category:
-- - Derivative volatility: 6 features
-- - Moving average: 8 features
-- - STFT: 16 features
-- - Pink noise: 24 features
-- - Composite: 10 features
--
-- Total: 64 features across 6 feature sets
-- ============================================================================

-- ============================================================================
-- PART 1: REGISTER FEATURES IN ml_features_lut
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Derivative Volatility Features (6 features)
-- Category: temporal, Behavior: sample_wise (arrays) or aggregate (scalars)
-- Feature IDs: 31-36
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
(31, 'volatility_dxdt_n1', 'temporal', 'sample_wise', 'volatility_dxdt_n1',
 'First derivative volatility: (dX/dt)^2 / range(X) - point-wise array', true),

(32, 'volatility_dxdt_n2', 'temporal', 'sample_wise', 'volatility_dxdt_n2',
 'Second derivative volatility: (d²X/dt²)^2 / range(X) - point-wise array', true),

(33, 'volatility_dxdt_n3', 'temporal', 'sample_wise', 'volatility_dxdt_n3',
 'Third derivative volatility: (d³X/dt³)^2 / range(X) - point-wise array', true),

(34, 'volatility_dxdt_n1_mean', 'temporal', 'aggregate', 'volatility_dxdt_n1_mean',
 'Mean first derivative volatility across signal - scalar value', true),

(35, 'volatility_dxdt_n1_max', 'temporal', 'aggregate', 'volatility_dxdt_n1_max',
 'Maximum first derivative volatility - peak volatility event', true),

(36, 'volatility_dxdt_n2_mean', 'temporal', 'aggregate', 'volatility_dxdt_n2_mean',
 'Mean second derivative volatility (acceleration-based) - scalar value', true);


-- ----------------------------------------------------------------------------
-- Moving Average Features (8 features)
-- Category: temporal, Behavior: sample_wise (returns arrays)
-- Feature IDs: 37-44
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
(37, 'moving_average_n8', 'temporal', 'sample_wise', 'moving_average_n8',
 '8-point moving average with linear edge ramping', true),

(38, 'moving_average_n16', 'temporal', 'sample_wise', 'moving_average_n16',
 '16-point moving average with linear edge ramping', true),

(39, 'moving_average_n32', 'temporal', 'sample_wise', 'moving_average_n32',
 '32-point moving average with linear edge ramping (standard window)', true),

(40, 'moving_average_n64', 'temporal', 'sample_wise', 'moving_average_n64',
 '64-point moving average with linear edge ramping', true),

(41, 'moving_average_n128', 'temporal', 'sample_wise', 'moving_average_n128',
 '128-point moving average with linear edge ramping', true),

(42, 'moving_average_n256', 'temporal', 'sample_wise', 'moving_average_n256',
 '256-point moving average with linear edge ramping', true),

(43, 'moving_average_n512', 'temporal', 'sample_wise', 'moving_average_n512',
 '512-point moving average with linear edge ramping', true),

(44, 'moving_average_adaptive', 'temporal', 'sample_wise', 'moving_average_adaptive',
 'Adaptive window moving average based on local variance', true);


-- ----------------------------------------------------------------------------
-- STFT Features (16 features)
-- Category: spectral, Behavior: sample_wise (spectrograms) or aggregate (scalars)
-- Feature IDs: 45-60
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
-- Complete spectrograms (for visualization/advanced analysis)
(45, 'stft_n8_magnitude', 'spectral', 'sample_wise', 'stft_n8_magnitude',
 'STFT magnitude spectrogram: 8 time slices, 0% overlap - 2D array', true),

(46, 'stft_n8_o20_magnitude', 'spectral', 'sample_wise', 'stft_n8_o20_magnitude',
 'STFT magnitude spectrogram: 8 slices, 20% overlap - 2D array', true),

(47, 'stft_n16_o50_magnitude', 'spectral', 'sample_wise', 'stft_n16_o50_magnitude',
 'STFT magnitude spectrogram: 16 slices, 50% overlap (high time resolution) - 2D array', true),

-- Time-averaged spectra
(48, 'stft_time_avg_spectrum_n8', 'spectral', 'sample_wise', 'stft_time_avg_spectrum_n8',
 'Time-averaged power spectrum from STFT (8 slices) - 1D frequency array', true),

(49, 'stft_time_avg_spectrum_n16', 'spectral', 'sample_wise', 'stft_time_avg_spectrum_n16',
 'Time-averaged power spectrum from STFT (16 slices) - 1D frequency array', true),

-- Scalar aggregates (for SVM training)
(50, 'stft_mean_power_n8', 'spectral', 'aggregate', 'stft_mean_power_n8',
 'Mean power across STFT time-frequency plane (8 slices) - scalar', true),

(51, 'stft_max_power_n8', 'spectral', 'aggregate', 'stft_max_power_n8',
 'Maximum power in STFT time-frequency plane (peak spectral event) - scalar', true),

(52, 'stft_total_energy_n8', 'spectral', 'aggregate', 'stft_total_energy_n8',
 'Total energy across STFT time-frequency plane (8 slices) - scalar', true),

-- Frequency band statistics
(53, 'stft_low_freq_power_n8', 'spectral', 'aggregate', 'stft_low_freq_power_n8',
 'Total power in low frequency band (0-100 Hz) from STFT - scalar', true),

(54, 'stft_mid_freq_power_n8', 'spectral', 'aggregate', 'stft_mid_freq_power_n8',
 'Total power in mid frequency band (100-1000 Hz) from STFT - scalar', true),

(55, 'stft_high_freq_power_n8', 'spectral', 'aggregate', 'stft_high_freq_power_n8',
 'Total power in high frequency band (1-10 kHz) from STFT - scalar', true),

(56, 'stft_band_ratio_n8', 'spectral', 'aggregate', 'stft_band_ratio_n8',
 'Ratio of low/high frequency power from STFT - arc signature indicator', true),

-- Spectral shape descriptors
(57, 'stft_mean_frequency_n8', 'spectral', 'aggregate', 'stft_mean_frequency_n8',
 'Mean spectral centroid across time (center of mass of spectrum) - Hz', true),

(58, 'stft_frequency_spread_n8', 'spectral', 'aggregate', 'stft_frequency_spread_n8',
 'Mean spectral spread (bandwidth around centroid) - Hz', true),

(59, 'stft_spectral_centroid_n8', 'spectral', 'aggregate', 'stft_spectral_centroid_n8',
 'Mean spectral centroid (alias for mean_frequency) - Hz', true),

-- Temporal evolution
(60, 'stft_peak_frequency_evolution_n8', 'spectral', 'sample_wise', 'stft_peak_frequency_evolution_n8',
 'Dominant frequency per time window - shows frequency evolution - 1D time array', true);


-- ----------------------------------------------------------------------------
-- Pink Noise Features - TMR Method (12 features)
-- Category: spectral, Behavior: aggregate (scalars)
-- Frequency range: 2-1000 Hz per TMR paper
-- Feature IDs: 61-72
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
(61, 'pink_noise_tmr_A_mean_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_mean_n64_m8',
 'Mean TMR A parameter (pink noise magnitude) across 8 frequency bands - scalar', true),

(62, 'pink_noise_tmr_A_max_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_max_n64_m8',
 'Maximum TMR A parameter across 8 bands - strongest arc indicator - scalar', true),

(63, 'pink_noise_tmr_gamma_mean_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_gamma_mean_n64_m8',
 'Mean TMR gamma parameter (spectral slope) across 8 bands - scalar', true),

(64, 'pink_noise_tmr_A_band1_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_band1_n64_m8',
 'TMR A parameter in frequency band 1 (2-5 Hz) - very low frequency arc signature', true),

(65, 'pink_noise_tmr_A_band2_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_band2_n64_m8',
 'TMR A parameter in frequency band 2 (5-12 Hz) - low frequency arc signature', true),

(66, 'pink_noise_tmr_A_band3_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_band3_n64_m8',
 'TMR A parameter in frequency band 3 (12-28 Hz) - mid-low frequency arc signature', true),

(67, 'pink_noise_tmr_A_band4_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_band4_n64_m8',
 'TMR A parameter in frequency band 4 (28-63 Hz) - mid frequency arc signature', true),

(68, 'pink_noise_tmr_gamma_max_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_gamma_max_n64_m8',
 'Maximum TMR gamma parameter (strongest 1/f character) across 8 bands', true),

(69, 'pink_noise_tmr_c_mean_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_c_mean_n64_m8',
 'Mean TMR c parameter (white noise floor) across 8 bands - baseline noise level', true),

(70, 'pink_noise_tmr_r2_mean_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_r2_mean_n64_m8',
 'Mean TMR R² goodness of fit across 8 bands - model fit quality', true),

(71, 'pink_noise_tmr_A_std_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_A_std_n64_m8',
 'Standard deviation of TMR A parameter across 8 bands - spectral variability', true),

(72, 'pink_noise_tmr_gamma_std_n64_m8', 'spectral', 'aggregate', 'pink_noise_tmr_gamma_std_n64_m8',
 'Standard deviation of TMR gamma across 8 bands - spectral slope variability', true);


-- ----------------------------------------------------------------------------
-- Pink Noise Features - Band Power Method (8 features)
-- Category: spectral, Behavior: aggregate (scalars)
-- Feature IDs: 73-80
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
(73, 'pink_noise_bandpower_total_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_total_n64_m8',
 'Total power across all 8 frequency bands (2-1000 Hz) - scalar', true),

(74, 'pink_noise_bandpower_lowfreq_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_lowfreq_n64_m8',
 'Total power in low frequency bands (bands 1-4, 2-63 Hz) - scalar', true),

(75, 'pink_noise_bandpower_highfreq_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_highfreq_n64_m8',
 'Total power in high frequency bands (bands 5-8, 63-1000 Hz) - scalar', true),

(76, 'pink_noise_bandpower_ratio_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_ratio_n64_m8',
 'Ratio of low/high frequency band power - arc signature indicator', true),

(77, 'pink_noise_bandpower_variance_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_variance_n64_m8',
 'Temporal variance of band power (across time) - power fluctuation measure', true),

(78, 'pink_noise_bandpower_max_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_max_n64_m8',
 'Maximum band power across all bands and time - peak power event', true),

(79, 'pink_noise_bandpower_band1_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_band1_n64_m8',
 'Average power in band 1 (2-5 Hz) across time - very low frequency power', true),

(80, 'pink_noise_bandpower_band4_n64_m8', 'spectral', 'aggregate', 'pink_noise_bandpower_band4_n64_m8',
 'Average power in band 4 (28-63 Hz) across time - mid frequency power', true);


-- ----------------------------------------------------------------------------
-- Pink Noise Features - Ratio Method (4 features)
-- Category: spectral, Behavior: aggregate (scalars)
-- Feature IDs: 81-84
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
(81, 'pink_noise_ratio_mean_n64', 'spectral', 'aggregate', 'pink_noise_ratio_mean_n64',
 'Mean low/high frequency power ratio across time (2-100Hz / 100-1000Hz)', true),

(82, 'pink_noise_ratio_max_n64', 'spectral', 'aggregate', 'pink_noise_ratio_max_n64',
 'Maximum low/high frequency ratio - peak pink noise event', true),

(83, 'pink_noise_ratio_variance_n64', 'spectral', 'aggregate', 'pink_noise_ratio_variance_n64',
 'Temporal variance of low/high frequency ratio - ratio fluctuation', true),

(84, 'pink_noise_ratio_slope_n64', 'spectral', 'aggregate', 'pink_noise_ratio_slope_n64',
 'Linear trend of low/high ratio over segment duration - temporal evolution', true);


-- ----------------------------------------------------------------------------
-- Composite Features (10 features)
-- Category: composite, Behavior: aggregate (scalars) or sample_wise (arrays)
-- Feature IDs: 85-94
-- ----------------------------------------------------------------------------

INSERT INTO ml_features_lut
(feature_id, feature_name, feature_category, behavior_type, computation_function, description, is_active)
VALUES
-- STFT + Volatility compositions
(85, 'stft_volatility_n1_8slices', 'composite', 'aggregate', 'stft_volatility_n1_8slices',
 'Mean power of STFT applied to first derivative volatility - scalar', true),

(86, 'stft_volatility_n1_8slices_o20', 'composite', 'aggregate', 'stft_volatility_n1_8slices_o20',
 'Mean power of STFT (20% overlap) applied to first derivative volatility', true),

(87, 'stft_volatility_n1_low_freq', 'composite', 'aggregate', 'stft_volatility_n1_low_freq',
 'Low frequency (0-100 Hz) power of STFT applied to volatility', true),

(88, 'stft_volatility_n2_8slices', 'composite', 'aggregate', 'stft_volatility_n2_8slices',
 'Mean power of STFT applied to second derivative volatility', true),

-- Moving Average + Volatility compositions
(89, 'volatility_dxdt_n1_ma16', 'composite', 'sample_wise', 'volatility_dxdt_n1_ma16',
 '16-point smoothed first derivative volatility - array', true),

(90, 'volatility_dxdt_n1_ma32', 'composite', 'sample_wise', 'volatility_dxdt_n1_ma32',
 '32-point smoothed first derivative volatility - array', true),

(91, 'volatility_dxdt_n1_ma64', 'composite', 'sample_wise', 'volatility_dxdt_n1_ma64',
 '64-point smoothed first derivative volatility - array', true),

(92, 'volatility_dxdt_n2_ma32', 'composite', 'sample_wise', 'volatility_dxdt_n2_ma32',
 '32-point smoothed second derivative volatility - array', true),

-- Advanced multi-level compositions
(93, 'stft_ma_volatility_n1', 'composite', 'aggregate', 'stft_ma_volatility_n1',
 'STFT of smoothed volatility: STFT(MA(volatility, 32), 8) - three-level composition', true),

(94, 'volatility_spectral_product', 'composite', 'aggregate', 'volatility_spectral_product',
 'Product of volatility mean and STFT power - combined time-frequency metric', true);


-- ============================================================================
-- VERIFICATION QUERY: Check feature registration
-- ============================================================================

-- Count features by category
SELECT
    feature_category,
    COUNT(*) as feature_count
FROM ml_features_lut
WHERE feature_name LIKE 'volatility_%'
   OR feature_name LIKE 'moving_average_%'
   OR feature_name LIKE 'stft_%'
   OR feature_name LIKE 'pink_noise_%'
GROUP BY feature_category
ORDER BY feature_category;

-- Expected results:
-- temporal: 14 (6 volatility + 8 moving_average)
-- spectral: 40 (16 STFT + 24 pink_noise)
-- composite: 10
-- TOTAL: 64 features

-- List all new features
SELECT
    feature_id,
    feature_name,
    feature_category,
    behavior_type,
    computation_function
FROM ml_features_lut
WHERE feature_name LIKE 'volatility_%'
   OR feature_name LIKE 'moving_average_%'
   OR feature_name LIKE 'stft_%'
   OR feature_name LIKE 'pink_noise_%'
ORDER BY feature_category, feature_name;
