-- Update ml_distance_functions_lut for sklearn.metrics.pairwise_distances and POT
--
-- This script:
-- 1. Adds pairwise_metric_name column
-- 2. Updates all 17 distance functions to use sklearn, POT, or custom implementations
--
-- Run with: psql -h localhost -p 5432 -d arc_detection -f update_distance_functions_lut.sql

-- Add column for pairwise metric name
ALTER TABLE ml_distance_functions_lut
ADD COLUMN IF NOT EXISTS pairwise_metric_name TEXT;

-- ========================================
-- sklearn.metrics.pairwise_distances functions
-- ========================================

-- Update manhattan (L1)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'manhattan',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'L1 distance (Manhattan/Cityblock) - Sum of absolute differences'
WHERE function_name = 'manhattan';

-- Update euclidean (L2)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'euclidean',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'L2 distance (Euclidean) - Square root of sum of squared differences'
WHERE function_name = 'euclidean';

-- Update cosine
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'cosine',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Cosine distance - 1 - cosine similarity'
WHERE function_name = 'cosine';

-- Update pearson (correlation in sklearn)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'correlation',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Pearson correlation distance - 1 - Pearson correlation coefficient'
WHERE function_name = 'pearson';

-- Update braycurtis
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'braycurtis',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Bray-Curtis distance - Sum of absolute differences divided by sum of values'
WHERE function_name = 'braycurtis';

-- Update canberra
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'canberra',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Canberra distance - Weighted version of L1 distance'
WHERE function_name = 'canberra';

-- Update chebyshev
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'chebyshev',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Chebyshev distance - Maximum absolute difference between components'
WHERE function_name = 'chebyshev';

-- Update sqeuclidean
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'sqeuclidean',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Squared Euclidean distance - Sum of squared differences (no square root)'
WHERE function_name = 'sqeuclidean';

-- Update mahalanobis
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'mahalanobis',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Mahalanobis distance - Accounts for correlations between variables (requires covariance matrix)'
WHERE function_name = 'mahalanobis';

-- Update jensenshannon
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'jensenshannon',
    library_name = 'sklearn.metrics.pairwise',
    function_import = 'pairwise_distances',
    description = 'Jensen-Shannon divergence - Symmetric version of KL divergence based on information theory'
WHERE function_name = 'jensenshannon';

-- ========================================
-- POT (Python Optimal Transport) functions
-- ========================================

-- Update wasserstein (uses POT library)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = 'wasserstein_1d',
    library_name = 'pot',
    function_import = 'ot.wasserstein_1d',
    description = 'Wasserstein distance (Earth Mover Distance) - Optimal transport distance between distributions'
WHERE function_name = 'wasserstein';

-- ========================================
-- Custom implementation functions
-- ========================================

-- Update fidelity (custom implementation)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Fidelity distance - Hellinger-related metric for probability distributions'
WHERE function_name = 'fidelity';

-- Update kullback_leibler (custom symmetric KL divergence)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Kullback-Leibler divergence (symmetric) - Information-theoretic distance'
WHERE function_name = 'kullback_leibler';

-- Update kumar_hassebrook (custom implementation)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Kumar-Hassebrook distance - Similarity coefficient based distance'
WHERE function_name = 'kumar_hassebrook';

-- Update additive_symmetric (custom implementation)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Additive symmetric chi-squared distance - Chi-squared test based metric'
WHERE function_name = 'additive_symmetric';

-- Update taneja (custom implementation)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Taneja divergence - Information-theoretic distance measure'
WHERE function_name = 'taneja';

-- Update wavehedges (custom implementation)
UPDATE ml_distance_functions_lut
SET pairwise_metric_name = NULL,
    library_name = 'custom',
    function_import = 'custom_distance_engine',
    description = 'Wave Hedges distance - Histogram comparison metric'
WHERE function_name = 'wavehedges';

-- ========================================
-- Verification
-- ========================================

-- Display all distance functions with their configurations
SELECT distance_function_id, function_name, pairwise_metric_name, library_name, function_import, is_active
FROM ml_distance_functions_lut
ORDER BY
    CASE library_name
        WHEN 'sklearn.metrics.pairwise' THEN 1
        WHEN 'pot' THEN 2
        WHEN 'custom' THEN 3
        ELSE 4
    END,
    distance_function_id;
