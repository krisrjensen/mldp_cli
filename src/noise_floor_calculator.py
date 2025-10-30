"""
Filename: noise_floor_calculator.py
Author(s): Kristophor Jensen
Date Created: 20251029_000000
Date Revised: 20251029_000000
File version: 1.0.0.5
Description: Calculates noise floor values from approved steady-state segments using spectral PSD methods

Changelog:
v1.0.0.5 (2025-10-29):
  - MAJOR REWRITE: Load from raw ADC files in adc_data/ instead of pre-extracted segments
  - Changed __init__ to use adc_data/ directory (all segments available)
  - Updated _query_approved_segments() to include beginning_index, bit_depth
  - Rewrote _load_segment_data() to:
    * Load raw ADC file from adc_data/{file_id:08d}.npy
    * Extract segment using beginning_index and segment_length
    * Requantize from 12-bit to target bit_depth (6, 8, 10, 12)
    * Requantization uses right-shift: adc10>>2, adc8>>4, adc6>>6
  - Fixes issue where segment_files/ don't exist for all segments

v1.0.0.4 (2025-10-29):
  - Fixed path construction: Added experiment_id parameter to __init__
  - Path now correctly built as: data_root/experiment{id:03d}/segment_files/
  - Fixed SQL query: Changed is_quantized to is_raw = false (correct column name)
  - All 4 CLI commands now pass self.current_experiment to calculator

v1.0.0.3 (2025-10-29):
  - CRITICAL FIX: Changed to load RAW segment files instead of processed feature files
  - Updated __init__ to use segment_files/ directory instead of fileset/adc_data
  - Rewrote _query_approved_segments() to query data_segments directly
  - Added CROSS JOIN with ml_data_types_lut to get all data type combinations
  - Filter for is_quantized=true to get only ADC data types
  - Rewrote _load_segment_data() to construct raw segment file paths
  - Path format: segment_files/S{length}/T{type}/D000000/SID{id}_F{file}_D000000_T{type}_S{length}_R*.npy
  - Use glob pattern matching for R* suffix
  - Default segment_length=8192 to match available segment_files/ data

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

    def __init__(self, db_connection_params: Dict[str, str], data_root: str, experiment_id: int = 41):
        """
        Initialize noise floor calculator

        Args:
            db_connection_params: Database connection parameters (host, database, user, password)
            data_root: Root directory containing adc_data/ folder
            experiment_id: Experiment ID (default 41)
        """
        self.db_params = db_connection_params
        self.data_root = Path(data_root)
        self.experiment_id = experiment_id

        # Path to raw ADC data files
        self.adc_data_dir = self.data_root / 'adc_data'

        if not self.adc_data_dir.exists():
            logger.warning(f"ADC data directory does not exist: {self.adc_data_dir}")

    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)

    def _query_approved_segments(
        self,
        data_type_id: Optional[int] = None,
        experiment_id: int = 41,
        segment_length: int = 8192
    ) -> List[Dict]:
        """
        Query raw segments from approved files

        Args:
            data_type_id: Specific data type ID or None for all
            experiment_id: Experiment ID (default 41)
            segment_length: Segment length to use (default 8192)

        Returns:
            List of segment records with metadata for extracting from raw ADC files
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query raw segments from data_segments table
                # Only include segments from approved files (experiment_status.status=true)
                file_training_table = f"experiment_{experiment_id:03d}_file_training_data"

                # Query segments cross-joined with data types
                # This gives us one row per (segment, data_type) combination
                query = f"""
                    SELECT DISTINCT
                        ds.segment_id,
                        ds.beginning_index,
                        ds.segment_length,
                        dt.data_type_id,
                        dt.data_type_name,
                        dt.bit_depth,
                        f.sampling_rate,
                        ft.file_id
                    FROM data_segments ds
                    JOIN {file_training_table} ft ON ds.experiment_file_id = ft.file_id
                    JOIN files f ON ft.file_id = f.file_id
                    JOIN experiment_status es ON ft.file_id = es.file_id
                    CROSS JOIN ml_data_types_lut dt
                    WHERE es.status = true
                      AND ds.segment_length = %s
                      AND dt.is_raw = false
                """

                params = [segment_length]

                if data_type_id is not None:
                    query += " AND dt.data_type_id = %s"
                    params.append(data_type_id)

                cur.execute(query, params)

                segments = cur.fetchall()

                logger.info(f"Found {len(segments)} segment/data_type combinations from approved files")
                return segments

        finally:
            conn.close()

    def _load_segment_data(self, segment: Dict) -> Optional[np.ndarray]:
        """
        Load and extract segment data from raw ADC file

        Args:
            segment: Segment record from database with file_id, beginning_index,
                    segment_length, bit_depth

        Returns:
            Requantized segment data array or None if not found
        """
        segment_id = segment['segment_id']
        file_id = segment['file_id']
        beginning_index = segment['beginning_index']
        segment_length = segment['segment_length']
        bit_depth = segment['bit_depth']
        data_type_name = segment['data_type_name']

        # Construct path to raw ADC file
        adc_file_path = self.adc_data_dir / f"{file_id:08d}.npy"

        if not adc_file_path.exists():
            logger.warning(f"ADC file not found: {adc_file_path}")
            return None

        try:
            # Load raw ADC data file (12-bit uint32 data)
            raw_data = np.load(adc_file_path)

            # Extract segment using beginning_index
            end_index = beginning_index + segment_length

            if end_index > len(raw_data):
                logger.error(f"Segment extends beyond file: {end_index} > {len(raw_data)}")
                return None

            segment_data = raw_data[beginning_index:end_index]

            # Requantize from 12-bit to target bit depth
            # Raw data is 12-bit (0-4095), shape (N, 4) for 4 channels
            if bit_depth == 12:
                # Use as-is
                quantized_data = segment_data
            else:
                # Requantize by right-shifting
                # adc10: shift right by 2 (4095 -> 1023)
                # adc8: shift right by 4 (4095 -> 255)
                # adc6: shift right by 6 (4095 -> 63)
                shift_amount = 12 - bit_depth
                quantized_data = segment_data >> shift_amount

            logger.debug(
                f"Loaded segment {segment_id} ({data_type_name}): "
                f"shape={quantized_data.shape}, range=[{quantized_data.min()}, {quantized_data.max()}]"
            )

            return quantized_data

        except Exception as e:
            logger.error(f"Failed to load segment {segment_id} from file {file_id}: {e}")
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
