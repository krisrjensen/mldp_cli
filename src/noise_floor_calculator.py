"""
Filename: noise_floor_calculator.py
Author(s): Kristophor Jensen
Date Created: 20251029_000000
Date Revised: 20251029_000000
File version: 1.0.0.2
Description: Calculates noise floor values from approved steady-state segments using spectral PSD methods

Changelog:
v1.0.0.2 (2025-10-29):
  - Fixed store_noise_floor() to convert numpy types to Python native types
  - Prevents PostgreSQL "schema 'np' does not exist" error
  - Converts np.float64 to float and np.int64 to int before SQL insert

v1.0.0.1 (2025-10-29):
  - Fixed SQL query to use experiment_041_feature_fileset instead of experiment_status
  - Added feature_file_path to query SELECT
  - Updated _load_segment_data() to use feature_file_path field
  - Fixed table joins to properly access data_type_id

v1.0.0.0 (2025-10-29):
  - Initial implementation
  - Spectral PSD calculation using Welch's method
  - 10th percentile noise floor estimation
  - Multi-segment averaging
  - Database storage in experiment_noise_floor table
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import signal
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class NoiseFloorCalculator:
    """
    Calculates noise floor values from approved steady-state segments
    """

    def __init__(self, db_connection_params: Dict[str, str], data_root: str):
        """
        Initialize noise floor calculator

        Args:
            db_connection_params: Database connection parameters (host, database, user, password)
            data_root: Root directory containing fileset/ and adc_data/ folders
        """
        self.db_params = db_connection_params
        self.data_root = Path(data_root)

        # Verify data directories exist
        self.fileset_dir = self.data_root / 'fileset'
        self.adc_data_dir = self.data_root / 'adc_data'

        if not self.fileset_dir.exists() and not self.adc_data_dir.exists():
            logger.warning(f"Neither {self.fileset_dir} nor {self.adc_data_dir} exist")

    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)

    def _query_approved_segments(
        self,
        data_type_id: Optional[int] = None,
        experiment_id: int = 41
    ) -> List[Dict]:
        """
        Query segments from approved files with feature extraction data

        Args:
            data_type_id: Specific data type ID or None for all
            experiment_id: Experiment ID (default 41)

        Returns:
            List of segment records with file paths and metadata
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query segments from feature_fileset table (has data_type_id)
                # Only include segments from approved files (experiment_status.status=true)
                feature_table = f"experiment_{experiment_id:03d}_feature_fileset"
                file_training_table = f"experiment_{experiment_id:03d}_file_training_data"

                query = f"""
                    SELECT DISTINCT
                        ff.segment_id,
                        ff.data_type_id,
                        dt.data_type_name,
                        ds.segment_length,
                        f.sampling_rate,
                        ft.file_id,
                        ff.decimation_factor,
                        ff.feature_file_path
                    FROM {feature_table} ff
                    JOIN ml_data_types_lut dt ON ff.data_type_id = dt.data_type_id
                    JOIN data_segments ds ON ff.segment_id = ds.segment_id
                    JOIN {file_training_table} ft ON ds.experiment_file_id = ft.file_id
                    JOIN files f ON ft.file_id = f.file_id
                    JOIN experiment_status es ON ft.file_id = es.file_id
                    WHERE es.status = true
                      AND ff.decimation_factor = 0
                """

                if data_type_id is not None:
                    query += " AND ff.data_type_id = %s"
                    cur.execute(query, (data_type_id,))
                else:
                    cur.execute(query)

                segments = cur.fetchall()

                logger.info(f"Found {len(segments)} segments from approved files")
                return segments

        finally:
            conn.close()

    def _load_segment_data(self, segment: Dict) -> Optional[np.ndarray]:
        """
        Load segment data from filesystem

        Args:
            segment: Segment record from database

        Returns:
            Segment data array or None if not found
        """
        # Get file path from feature_file_path field
        feature_file_path = segment.get('feature_file_path')

        if not feature_file_path:
            logger.error(f"No feature_file_path for segment {segment['segment_id']}")
            return None

        segment_path = Path(feature_file_path)

        # Check if absolute path exists
        if segment_path.exists():
            data_path = segment_path
        else:
            # Try relative to data_root
            relative_path = self.data_root / segment_path

            if relative_path.exists():
                data_path = relative_path
            else:
                logger.warning(f"Segment not found: {segment_path}")
                return None

        try:
            # Load .npy file
            data = np.load(data_path)
            logger.debug(f"Loaded segment {segment['segment_id']}: shape={data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load segment {segment['segment_id']}: {e}")
            return None

    def _calculate_segment_noise_floor(
        self,
        data: np.ndarray,
        sampling_rate: float = 250000.0
    ) -> float:
        """
        Calculate noise floor for a single segment using spectral PSD method

        Method:
          1. Compute Power Spectral Density using Welch's method
          2. Take 10th percentile of PSD (excludes signal peaks, focuses on noise)
          3. Convert to RMS units

        Args:
            data: Segment time-series data (1D array)
            sampling_rate: Sampling rate in Hz

        Returns:
            Noise floor value in RMS units
        """
        # Flatten if 2D
        if data.ndim > 1:
            data = data.flatten()

        # Compute PSD using Welch's method
        nperseg = min(1024, len(data) // 4)

        try:
            freqs, psd = signal.welch(
                data,
                fs=sampling_rate,
                nperseg=nperseg,
                window='hann',
                scaling='density'
            )

            # Exclude DC component (freq = 0)
            psd = psd[freqs > 0]
            freqs = freqs[freqs > 0]

            # Calculate 10th percentile of PSD
            psd_10th = np.percentile(psd, 10)

            # Take all PSD values at or below 10th percentile
            noise_psd = psd[psd <= psd_10th]

            # Convert to RMS: sqrt(mean(psd))
            noise_floor_rms = np.sqrt(np.mean(noise_psd))

            logger.debug(f"Segment noise floor: {noise_floor_rms:.6e} RMS")
            return noise_floor_rms

        except Exception as e:
            logger.error(f"Failed to calculate PSD: {e}")
            raise

    def calculate_noise_floor(
        self,
        data_type_id: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Calculate noise floors for one or all data types

        Args:
            data_type_id: Specific data type ID or None for all

        Returns:
            Dict mapping data_type_id to noise floor results:
                {
                    data_type_id: {
                        'noise_floor': float,
                        'num_segments': int,
                        'data_type_name': str
                    }
                }
        """
        # Query approved segments
        segments = self._query_approved_segments(data_type_id)

        if not segments:
            logger.warning("No approved steady-state segments found")
            return {}

        # Group segments by data_type_id
        segments_by_type = {}
        for seg in segments:
            dt_id = seg['data_type_id']
            if dt_id not in segments_by_type:
                segments_by_type[dt_id] = []
            segments_by_type[dt_id].append(seg)

        results = {}

        # Calculate noise floor for each data type
        for dt_id, dt_segments in segments_by_type.items():
            logger.info(f"Processing {len(dt_segments)} segments for data_type_id {dt_id}")

            noise_floors = []
            data_type_name = dt_segments[0]['data_type_name']
            sampling_rate = dt_segments[0].get('sampling_rate', 250000.0)

            for i, seg in enumerate(dt_segments):
                # Load segment data
                data = self._load_segment_data(seg)

                if data is None:
                    continue

                # Calculate noise floor for this segment
                try:
                    nf = self._calculate_segment_noise_floor(data, sampling_rate)
                    noise_floors.append(nf)
                except Exception as e:
                    logger.error(f"Failed to calculate noise floor for segment {seg['segment_id']}: {e}")
                    continue

                # Progress logging
                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i + 1}/{len(dt_segments)} segments")

            if not noise_floors:
                logger.warning(f"No valid noise floor calculations for data_type_id {dt_id}")
                continue

            # Average noise floors across segments
            avg_noise_floor = np.mean(noise_floors)

            logger.info(
                f"Data type {data_type_name} (ID {dt_id}): "
                f"noise_floor={avg_noise_floor:.6e} ({len(noise_floors)} segments)"
            )

            results[dt_id] = {
                'noise_floor': avg_noise_floor,
                'num_segments': len(noise_floors),
                'data_type_name': data_type_name
            }

        return results

    def store_noise_floor(self, data_type_id: int, noise_floor: float, num_segments: int):
        """
        Store noise floor value in database

        Args:
            data_type_id: Data type ID
            noise_floor: Calculated noise floor value
            num_segments: Number of segments used in calculation
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                # INSERT or UPDATE using ON CONFLICT
                query = """
                    INSERT INTO experiment_noise_floor
                    (data_type_id, noise_floor, calculation_method, num_segments_used, last_calculated)
                    VALUES (%s, %s, 'spectral_psd', %s, NOW())
                    ON CONFLICT (data_type_id)
                    DO UPDATE SET
                        noise_floor = EXCLUDED.noise_floor,
                        calculation_method = EXCLUDED.calculation_method,
                        num_segments_used = EXCLUDED.num_segments_used,
                        last_calculated = NOW()
                """

                # Convert numpy types to Python native types for PostgreSQL
                noise_floor_py = float(noise_floor)
                num_segments_py = int(num_segments)

                cur.execute(query, (data_type_id, noise_floor_py, num_segments_py))
                conn.commit()

                logger.info(f"Stored noise floor for data_type_id {data_type_id}: {noise_floor_py:.6e}")

        finally:
            conn.close()

    def get_noise_floors(self) -> List[Dict]:
        """
        Retrieve all noise floor values from database

        Returns:
            List of noise floor records
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT
                        nf.data_type_id,
                        dt.data_type_name,
                        nf.noise_floor,
                        nf.calculation_method,
                        nf.num_segments_used,
                        nf.last_calculated,
                        nf.notes
                    FROM experiment_noise_floor nf
                    JOIN ml_data_types_lut dt ON nf.data_type_id = dt.data_type_id
                    ORDER BY dt.data_type_name
                """

                cur.execute(query)
                return cur.fetchall()

        finally:
            conn.close()

    def clear_noise_floor(self, data_type_id: Optional[int] = None) -> int:
        """
        Clear noise floor entries from database

        Args:
            data_type_id: Specific data type ID or None for all

        Returns:
            Number of rows deleted
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                if data_type_id is None:
                    # Clear all
                    cur.execute("DELETE FROM experiment_noise_floor")
                else:
                    # Clear specific
                    cur.execute(
                        "DELETE FROM experiment_noise_floor WHERE data_type_id = %s",
                        (data_type_id,)
                    )

                rows_deleted = cur.rowcount
                conn.commit()

                logger.info(f"Cleared {rows_deleted} noise floor entries")
                return rows_deleted

        finally:
            conn.close()

    def table_exists(self) -> bool:
        """
        Check if experiment_noise_floor table exists

        Returns:
            True if table exists
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = 'experiment_noise_floor'
                    )
                """)

                return cur.fetchone()[0]

        finally:
            conn.close()

    def get_entry_count(self) -> int:
        """
        Get number of entries in experiment_noise_floor table

        Returns:
            Number of entries
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM experiment_noise_floor")
                return cur.fetchone()[0]

        finally:
            conn.close()
