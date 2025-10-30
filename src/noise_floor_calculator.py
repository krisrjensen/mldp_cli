"""
Filename: noise_floor_calculator.py
Author(s): Kristophor Jensen
Date Created: 20251029_000000
Date Revised: 20251029_000000
File version: 1.0.0.12
Description: Calculates noise floor values from approved steady-state segments using standard deviation

Changelog:
v1.0.0.12 (2025-10-29):
  - CRITICAL FIX: Auto-detect source bit depth from data range
  - Removed incorrect premature validation of raw data
  - Raw ADC data may be stored in 12, 14, or 16-bit formats
  - Automatically detect source format: max <= 4095 (12-bit), <= 16383 (14-bit), <= 65535 (16-bit)
  - Calculate shift amount as: source_bit_depth - target_bit_depth
  - Example: 16-bit source to 8-bit target: shift by 8 (not 4)
  - Added debug logging for source detection and shift amounts
  - Fixed: Values like 25341 now correctly shifted to valid ranges

v1.0.0.11 (2025-10-29):
  - MAJOR REWRITE: Separate voltage and current channel processing
  - Added noise_floor_voltage and noise_floor_current columns
  - Strict segment_length = 8192 enforcement with verification
  - Added channel selection (voltage=channel 0, current=channel 1)
  - Calculate and store separate noise floors for voltage and current
  - Extensive validation and error reporting
  - Fixed: Ensure only 8192-sample segments are processed

v1.0.0.10 (2025-10-29):
  - Added range validation for requantized data in _load_segment_data()
  - Validates all values are within expected range for each bit depth
  - Skips segments with values outside valid range

v1.0.0.9 (2025-10-29):
  - MAJOR FIX: Changed from PSD method to direct standard deviation calculation
  - New method: Calculate std per channel, take median across channels
  - Returns noise floor in ADC counts (RMS)
"""

import numpy as np
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class NoiseFloorCalculator:
    """
    Calculates noise floor values from approved steady-state segments

    Uses direct standard deviation calculation on raw ADC data
    Processes voltage and current channels separately
    """

    def __init__(self, db_params: Dict, data_root: str, experiment_id: int):
        """
        Initialize noise floor calculator

        Args:
            db_params: Database connection parameters
            data_root: Root directory for data files
            experiment_id: Experiment ID (e.g., 41)
        """
        self.db_params = db_params
        self.data_root = Path(data_root)
        self.experiment_id = experiment_id

        # Path to raw ADC data files
        self.adc_data_dir = self.data_root / "adc_data"

        if not self.adc_data_dir.exists():
            raise ValueError(f"ADC data directory does not exist: {self.adc_data_dir}")

        logger.info(f"Initialized NoiseFloorCalculator for experiment {experiment_id}")
        logger.info(f"ADC data directory: {self.adc_data_dir}")

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
        Query raw segments from approved files with STRICT segment_length filtering

        Args:
            data_type_id: Specific data type ID or None for all
            experiment_id: Experiment ID (default 41)
            segment_length: REQUIRED segment length (default 8192, STRICTLY enforced)

        Returns:
            List of segment records with metadata for extracting from raw ADC files
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                file_training_table = f"experiment_{experiment_id:03d}_file_training_data"

                # STRICT segment_length filter - ONLY 8192-sample segments
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
                      AND dt.bit_depth <= 12
                """

                params = [segment_length]

                if data_type_id is not None:
                    query += " AND dt.data_type_id = %s"
                    params.append(data_type_id)

                # Add ORDER BY to ensure consistent processing
                query += " ORDER BY ds.segment_id, dt.data_type_id"

                cur.execute(query, params)
                segments = cur.fetchall()

                # VERIFY segment_length is actually 8192 for ALL segments
                invalid_lengths = [s for s in segments if s['segment_length'] != segment_length]
                if invalid_lengths:
                    logger.error(f"Found {len(invalid_lengths)} segments with wrong length!")
                    for s in invalid_lengths[:5]:  # Show first 5
                        logger.error(f"  Segment {s['segment_id']}: length={s['segment_length']} (expected {segment_length})")
                    raise ValueError(f"Query returned segments with wrong length! Expected {segment_length}")

                logger.info(f"Found {len(segments)} segment/data_type combinations (segment_length={segment_length})")
                return segments

        finally:
            conn.close()

    def _load_segment_data(
        self,
        segment: Dict,
        channel_type: str
    ) -> Optional[np.ndarray]:
        """
        Load and extract segment data from raw ADC file for specific channel

        Args:
            segment: Segment record from database with file_id, beginning_index,
                    segment_length, bit_depth
            channel_type: 'voltage' or 'current'

        Returns:
            Requantized segment data array (1D) for specified channel or None if invalid
        """
        segment_id = segment['segment_id']
        file_id = segment['file_id']
        beginning_index = segment['beginning_index']
        segment_length = segment['segment_length']
        bit_depth = segment['bit_depth']
        data_type_name = segment['data_type_name']

        # STRICT: Verify segment_length is 8192
        if segment_length != 8192:
            logger.error(
                f"Segment {segment_id} has wrong length {segment_length} (expected 8192). "
                f"This should have been filtered by query!"
            )
            return None

        # Determine channel index
        # Assuming raw data shape is (N, 4) with channels: [voltage, current, ?, ?]
        # User said "select the proper voltage or current channel"
        if channel_type == 'voltage':
            channel_idx = 0
        elif channel_type == 'current':
            channel_idx = 1
        else:
            raise ValueError(f"Invalid channel_type: {channel_type}. Must be 'voltage' or 'current'")

        # Construct path to raw ADC file
        adc_file_path = self.adc_data_dir / f"{file_id:08d}.npy"

        if not adc_file_path.exists():
            logger.warning(f"ADC file not found: {adc_file_path}")
            return None

        try:
            # Load raw ADC data file (12-bit data)
            raw_data = np.load(adc_file_path)

            # Validate shape
            if raw_data.ndim != 2:
                logger.error(
                    f"Raw data has wrong shape {raw_data.shape} for file {file_id}. "
                    f"Expected 2D array (N, channels)"
                )
                return None

            num_channels = raw_data.shape[1]
            if channel_idx >= num_channels:
                logger.error(
                    f"Channel index {channel_idx} out of bounds for file {file_id} "
                    f"with {num_channels} channels"
                )
                return None

            # Ensure unsigned integer type (handle signed data)
            if raw_data.dtype != np.uint32:
                logger.debug(f"Converting {raw_data.dtype} to uint32 for file {file_id}")
                if np.issubdtype(raw_data.dtype, np.signedinteger):
                    raw_data = np.bitwise_and(raw_data, 0xFFFFFFFF).astype(np.uint32)
                else:
                    raw_data = raw_data.astype(np.uint32)

            # Extract segment using beginning_index
            end_index = beginning_index + segment_length

            if end_index > len(raw_data):
                logger.error(f"Segment extends beyond file: {end_index} > {len(raw_data)}")
                return None

            # Extract segment data for ALL channels first
            segment_data_all_channels = raw_data[beginning_index:end_index].copy()

            # Extract the specific channel
            segment_data_channel = segment_data_all_channels[:, channel_idx].copy()

            # NOTE: Raw data may be stored in 16-bit or 32-bit format
            # The actual ADC values will be extracted via right-shifting below
            logger.debug(
                f"Segment {segment_id} raw data range: "
                f"[{segment_data_channel.min()}, {segment_data_channel.max()}]"
            )

            # Validate bit_depth
            if bit_depth is None or bit_depth <= 0 or bit_depth > 12:
                logger.error(f"Invalid bit_depth {bit_depth} for segment {segment_id}")
                return None

            # Determine source bit depth from data range
            # Raw ADC data may be stored in 14-bit, 16-bit, or other formats
            data_max = segment_data_channel.max()
            if data_max <= 4095:
                source_bit_depth = 12
            elif data_max <= 16383:
                source_bit_depth = 14
            elif data_max <= 65535:
                source_bit_depth = 16
            else:
                logger.error(
                    f"Segment {segment_id} data max {data_max} exceeds 16-bit range. "
                    f"Unsupported format."
                )
                return None

            logger.debug(
                f"Segment {segment_id}: Detected source_bit_depth={source_bit_depth} "
                f"(data_max={data_max}), target bit_depth={bit_depth}"
            )

            # Requantize from source bit depth to target bit depth
            if source_bit_depth == bit_depth:
                # Already at target bit depth
                quantized_data = segment_data_channel.astype(np.uint32)
            else:
                # Right-shift from source to target
                # Example: 16-bit to 8-bit: shift by 16-8=8
                # Example: 14-bit to 8-bit: shift by 14-8=6
                # Example: 12-bit to 8-bit: shift by 12-8=4
                shift_amount = source_bit_depth - bit_depth

                logger.debug(
                    f"Segment {segment_id}: Shifting {source_bit_depth}-bit -> {bit_depth}-bit "
                    f"(shift_amount={shift_amount})"
                )

                # Use numpy right shift to ensure proper unsigned handling
                quantized_data = np.right_shift(segment_data_channel, shift_amount).astype(np.uint32)

            # Validate quantized data range
            # adc6: [0, 63], adc8: [0, 255], adc10: [0, 1023], adc12: [0, 4095]
            max_value = (2 ** bit_depth) - 1
            if np.any(quantized_data < 0) or np.any(quantized_data > max_value):
                invalid_count = np.sum((quantized_data < 0) | (quantized_data > max_value))
                logger.warning(
                    f"Segment {segment_id} ({data_type_name}, {channel_type}) contains {invalid_count} "
                    f"values outside valid range [0, {max_value}]. Skipping segment."
                )
                return None

            logger.debug(
                f"Loaded segment {segment_id} ({data_type_name}, {channel_type}): "
                f"shape={quantized_data.shape}, dtype={quantized_data.dtype}, "
                f"range=[{quantized_data.min()}, {quantized_data.max()}]"
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
        Calculate noise floor for a single segment using standard deviation method

        Method:
          - Calculate standard deviation (RMS noise) for the channel

        Args:
            data: Segment time-series data, shape (N,) for single channel
            sampling_rate: Sampling rate in Hz (not used in this method)

        Returns:
            Noise floor value in ADC counts (RMS)
        """
        try:
            # Ensure float type for calculations
            data_float = data.astype(np.float64)

            # Calculate standard deviation (RMS noise)
            noise_floor_rms = np.std(data_float)

            logger.debug(f"Segment noise floor: {noise_floor_rms:.6f} ADC counts (RMS)")
            return noise_floor_rms

        except Exception as e:
            logger.error(f"Failed to calculate noise floor: {e}")
            raise

    def calculate_noise_floor(
        self,
        data_type_id: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Calculate noise floors for one or all data types (voltage and current separately)

        Args:
            data_type_id: Specific data type ID or None for all

        Returns:
            Dict mapping data_type_id to noise floor results:
                {
                    data_type_id: {
                        'noise_floor_voltage': float,
                        'noise_floor_current': float,
                        'num_segments': int,
                        'data_type_name': str
                    }
                }
        """
        # Query approved segments (STRICT segment_length = 8192)
        print("Querying approved segments from database (segment_length=8192 ONLY)...")
        segments = self._query_approved_segments(data_type_id, segment_length=8192)

        if not segments:
            print("⚠️  No approved steady-state segments found with length 8192")
            logger.warning("No approved steady-state segments found")
            return {}

        print(f"✓ Found {len(segments)} segment/data_type combinations (all with length=8192)")

        # Group segments by data_type_id
        segments_by_type = {}
        for seg in segments:
            dt_id = seg['data_type_id']
            if dt_id not in segments_by_type:
                segments_by_type[dt_id] = []
            segments_by_type[dt_id].append(seg)

        # Build data type summary
        type_summary = ', '.join([f"{v[0]['data_type_name']}({len(v)})" for v in segments_by_type.values()])
        print(f"Processing {len(segments_by_type)} data types: {type_summary}\n")

        results = {}

        # Calculate noise floor for each data type
        for dt_idx, (dt_id, dt_segments) in enumerate(segments_by_type.items(), 1):
            data_type_name = dt_segments[0]['data_type_name']
            sampling_rate = dt_segments[0].get('sampling_rate', 250000.0)

            print(f"[{dt_idx}/{len(segments_by_type)}] Processing {data_type_name} (ID {dt_id}): {len(dt_segments)} segments")
            logger.info(f"Processing {len(dt_segments)} segments for data_type_id {dt_id}")

            # Process voltage and current channels separately
            voltage_noise_floors = []
            current_noise_floors = []
            successful_voltage = 0
            successful_current = 0
            failed_voltage = 0
            failed_current = 0

            for i, seg in enumerate(dt_segments):
                # Process VOLTAGE channel
                voltage_data = self._load_segment_data(seg, channel_type='voltage')
                if voltage_data is not None:
                    try:
                        nf_v = self._calculate_segment_noise_floor(voltage_data, sampling_rate)
                        voltage_noise_floors.append(nf_v)
                        successful_voltage += 1
                    except Exception as e:
                        failed_voltage += 1
                        logger.error(f"Failed to calculate voltage noise floor for segment {seg['segment_id']}: {e}")
                else:
                    failed_voltage += 1

                # Process CURRENT channel
                current_data = self._load_segment_data(seg, channel_type='current')
                if current_data is not None:
                    try:
                        nf_c = self._calculate_segment_noise_floor(current_data, sampling_rate)
                        current_noise_floors.append(nf_c)
                        successful_current += 1
                    except Exception as e:
                        failed_current += 1
                        logger.error(f"Failed to calculate current noise floor for segment {seg['segment_id']}: {e}")
                else:
                    failed_current += 1

                # Progress updates every 100 segments
                if (i + 1) % 100 == 0:
                    pct = int((i + 1) / len(dt_segments) * 100)
                    print(f"  Progress: {i+1}/{len(dt_segments)} ({pct}%) - V:{successful_voltage}✓/{failed_voltage}❌, C:{successful_current}✓/{failed_current}❌")

            # Calculate final noise floors (median across all segments)
            if len(voltage_noise_floors) > 0:
                final_voltage_nf = np.median(voltage_noise_floors)
            else:
                print(f"  ❌ Failed: No valid voltage segments for {data_type_name}")
                logger.error(f"No valid voltage segments for data_type_id {dt_id}")
                continue

            if len(current_noise_floors) > 0:
                final_current_nf = np.median(current_noise_floors)
            else:
                print(f"  ❌ Failed: No valid current segments for {data_type_name}")
                logger.error(f"No valid current segments for data_type_id {dt_id}")
                continue

            print(f"  ✓ Success: V:{successful_voltage} segs, C:{successful_current} segs")
            print(f"  Voltage noise floor: {final_voltage_nf:.6f} ADC counts (RMS)")
            print(f"  Current noise floor: {final_current_nf:.6f} ADC counts (RMS)")

            # Store result
            results[dt_id] = {
                'noise_floor_voltage': final_voltage_nf,
                'noise_floor_current': final_current_nf,
                'num_segments': len(voltage_noise_floors),  # Should be same for both
                'data_type_name': data_type_name
            }

            # Save to database
            self._save_noise_floor(dt_id, final_voltage_nf, final_current_nf, len(voltage_noise_floors))

        return results

    def _save_noise_floor(
        self,
        data_type_id: int,
        noise_floor_voltage: float,
        noise_floor_current: float,
        num_segments: int
    ):
        """
        Save noise floor to database

        Args:
            data_type_id: Data type ID
            noise_floor_voltage: Calculated voltage noise floor
            noise_floor_current: Calculated current noise floor
            num_segments: Number of segments used in calculation
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                # Upsert noise floor
                query = """
                    INSERT INTO experiment_noise_floor
                    (data_type_id, noise_floor_voltage, noise_floor_current,
                     calculation_method, num_segments_used, last_calculated)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (data_type_id)
                    DO UPDATE SET
                        noise_floor_voltage = EXCLUDED.noise_floor_voltage,
                        noise_floor_current = EXCLUDED.noise_floor_current,
                        calculation_method = EXCLUDED.calculation_method,
                        num_segments_used = EXCLUDED.num_segments_used,
                        last_calculated = EXCLUDED.last_calculated
                """

                cur.execute(query, (
                    data_type_id,
                    noise_floor_voltage,
                    noise_floor_current,
                    'std_dev',
                    num_segments,
                    datetime.now()
                ))

                conn.commit()
                logger.info(f"Saved noise floor for data_type_id {data_type_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save noise floor for data_type_id {data_type_id}: {e}")
            raise
        finally:
            conn.close()

    def get_noise_floor(self, data_type_id: int) -> Optional[Tuple[float, float]]:
        """
        Get noise floor from database

        Args:
            data_type_id: Data type ID

        Returns:
            Tuple of (voltage_noise_floor, current_noise_floor) or None if not found
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT noise_floor_voltage, noise_floor_current
                    FROM experiment_noise_floor
                    WHERE data_type_id = %s
                """

                cur.execute(query, (data_type_id,))
                result = cur.fetchone()

                if result:
                    return (result['noise_floor_voltage'], result['noise_floor_current'])
                else:
                    return None

        finally:
            conn.close()

    def get_noise_floors(self) -> List[Dict]:
        """
        Get all noise floor entries from database

        Returns:
            List of noise floor entries with data type information
        """
        conn = self._get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT
                        nf.data_type_id,
                        dt.data_type_name,
                        nf.noise_floor_voltage,
                        nf.noise_floor_current,
                        nf.calculation_method,
                        nf.num_segments_used,
                        nf.last_calculated
                    FROM experiment_noise_floor nf
                    JOIN ml_data_types_lut dt ON nf.data_type_id = dt.data_type_id
                    ORDER BY dt.data_type_name
                """

                cur.execute(query)
                results = cur.fetchall()

                return results

        finally:
            conn.close()
