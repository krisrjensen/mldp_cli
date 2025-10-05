#!/usr/bin/env python3
"""
Filename: experiment_feature_extractor.py
Author: Kristophor Jensen
Date Created: 20250916_090000
Date Revised: 20251005_135000
File version: 1.1.0.0
Description: Extract features from segments and generate feature filesets
             Updated to support multi-column amplitude-processed segment files
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentFeatureExtractor:
    """Extract features from segments and generate feature filesets"""
    
    def __init__(self, experiment_id: int, db_conn):
        self.experiment_id = experiment_id
        self.db_conn = db_conn
        self.segment_table = f"experiment_{experiment_id:03d}_segment_training_data"
        self.feature_table = f"experiment_{experiment_id:03d}_feature_fileset"
        
        # Base paths for data
        self.base_segment_path = Path("/Volumes/ArcData/V3_database")
        self.base_feature_path = Path("/Volumes/ArcData/V3_database")
        
    def create_feature_fileset_table(self):
        """Create the feature fileset tracking table"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.feature_table} (
                    feature_file_id SERIAL PRIMARY KEY,
                    experiment_id INTEGER NOT NULL,
                    segment_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    feature_set_id INTEGER NOT NULL,
                    n_value INTEGER,
                    feature_file_path TEXT,
                    num_chunks INTEGER,
                    extraction_status VARCHAR(50),
                    extraction_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(experiment_id, segment_id, feature_set_id, n_value)
                )
            """)
            
            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_segment 
                ON {self.feature_table}(segment_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_feature_set 
                ON {self.feature_table}(feature_set_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_status 
                ON {self.feature_table}(extraction_status)
            """)
            
            self.db_conn.commit()
            logger.info(f"Created/verified table: {self.feature_table}")
            return True
            
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error creating feature table: {e}")
            return False
        finally:
            cursor.close()
    
    def get_experiment_feature_sets(self) -> List[Dict]:
        """Get feature sets configured for this experiment"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                SELECT 
                    efs.*,
                    fs.feature_set_name,
                    fs.category
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.experiment_id = %s
                ORDER BY efs.priority_order, efs.feature_set_id
            """, (self.experiment_id,))
            
            return [dict(row) for row in cursor]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting feature sets: {e}")
            return []
        finally:
            cursor.close()
    
    def get_selected_segments(self) -> List[Dict]:
        """Get segments selected for training"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(f"""
                SELECT
                    st.segment_id,
                    st.file_id,
                    st.segment_index,
                    st.segment_label_id,
                    ds.segment_length,
                    ds.segment_id_code
                FROM {self.segment_table} st
                JOIN data_segments ds ON st.segment_id = ds.segment_id
                WHERE st.experiment_id = %s
                ORDER BY st.selection_order
            """, (self.experiment_id,))

            segments = []
            for row in cursor:
                seg = dict(row)
                # Construct file path using segment properties
                # Path structure: /Volumes/ArcData/V3_database/experiment041/segment_files/S{length}/T{type}/D{dec}/SID{seg_id}_F{file_id}_D{dec}_T{type}_S{orig}_R{res}.npy
                # We need to search for the file since division/decimation/type can vary
                import glob
                base_path = f"/Volumes/ArcData/V3_database/experiment{self.experiment_id:03d}/segment_files"
                # Match new naming convention: SID{segment}_F{file}_D{dec}_T{type}_S{orig}_R{res}.npy
                pattern = f"{base_path}/S*/T*/D*/SID{seg['segment_id']:08d}_F{seg['file_id']:08d}_*.npy"

                matches = glob.glob(pattern)
                if matches:
                    seg['segment_file_path'] = matches[0]  # Use first match
                else:
                    logger.warning(f"No file found for segment {seg['segment_id']} with pattern {pattern}")
                    seg['segment_file_path'] = None

                segments.append(seg)

            # Filter out segments without files
            segments = [s for s in segments if s['segment_file_path'] is not None]

            return segments

        except psycopg2.Error as e:
            logger.error(f"Error getting segments: {e}")
            return []
        finally:
            cursor.close()
    
    def extract_features(self,
                        feature_set_ids: List[int] = None,
                        segment_ids: List[int] = None,
                        max_segments: int = None,
                        force_reextract: bool = False) -> Dict[str, Any]:
        """
        Extract features from ALL segment files (all decimation levels and ADC types)
        Scans entire segment_files filesystem and mirrors to feature_files

        Args:
            feature_set_ids: Specific feature sets to extract (None = all active)
            segment_ids: Specific segments to process (None = all)
            max_segments: Maximum segment FILES to process (not unique segments)
            force_reextract: Re-extract even if already exists

        Returns:
            Dictionary with extraction results
        """
        import re
        from pathlib import Path

        # Create table if needed
        if not self.create_feature_fileset_table():
            return {'success': False, 'error': 'Failed to create feature table'}

        # Build segment_id → original_length mapping from database
        logger.info("Loading segment_id → original_length mapping from database...")
        cursor = self.db_conn.cursor()
        cursor.execute(f"""
            SELECT ds.segment_id, ds.segment_length
            FROM data_segments ds
            JOIN {self.segment_table} st ON ds.segment_id = st.segment_id
            WHERE st.experiment_id = %s
        """, (self.experiment_id,))

        segment_original_lengths = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        logger.info(f"Loaded original lengths for {len(segment_original_lengths):,} segments")

        # Scan filesystem for ALL segment files
        segment_base = Path(f"/Volumes/ArcData/V3_database/experiment{self.experiment_id:03d}/segment_files")
        logger.info(f"Scanning filesystem: {segment_base}")

        all_segment_files = list(segment_base.rglob("*.npy"))
        logger.info(f"Found {len(all_segment_files):,} segment files")

        # Parse file paths to extract metadata
        segment_metadata = []
        for seg_path in all_segment_files:
            relative_path = seg_path.relative_to(segment_base)
            parts = relative_path.parts

            if len(parts) >= 4:
                stored_size_str = parts[0]  # S008192
                adc_type = parts[1]          # TADC10
                division = parts[2]          # D000000
                filename = parts[3]          # SID00155140_F00000001_D000000_TTRAW_S008192_R008192.npy

                # Parse stored size (resulting size after decimation)
                stored_size = int(stored_size_str.replace('S', ''))

                # Parse filename components: SID{seg}_F{file}_D{dec}_T{type}_S{orig}_R{res}.npy
                match = re.search(r'SID(\d+)_F(\d+)_D(\d+)_T(\w+)_S(\d+)_R(\d+)', filename)
                if match:
                    segment_id = int(match.group(1))
                    file_id = int(match.group(2))
                    decimation = int(match.group(3))
                    file_data_type = match.group(4)  # From filename
                    original_size_from_file = int(match.group(5))
                    resulting_size_from_file = int(match.group(6))

                    # Get original length from database
                    original_length = segment_original_lengths.get(segment_id)

                    segment_metadata.append({
                        'path': seg_path,
                        'segment_id': segment_id,
                        'file_id': file_id,
                        'stored_size': stored_size,
                        'adc_type': adc_type,
                        'division': division,
                        'decimation': decimation,
                        'original_size': original_size_from_file,
                        'resulting_size': resulting_size_from_file,
                        'filename': filename,
                        'original_length': original_length
                    })

        logger.info(f"Parsed {len(segment_metadata):,} segment files with metadata")

        # Filter by segment_ids if specified
        if segment_ids:
            segment_metadata = [m for m in segment_metadata if m['segment_id'] in segment_ids]
            logger.info(f"Filtered to {len(segment_metadata):,} files matching segment_ids")

        # Limit if specified
        if max_segments:
            segment_metadata = segment_metadata[:max_segments]
            logger.info(f"Limited to {max_segments:,} files")

        # Get feature sets with overrides
        if feature_set_ids:
            feature_sets = [
                self._get_feature_set_with_overrides(fs_id)
                for fs_id in feature_set_ids
            ]
            feature_sets = [fs for fs in feature_sets if fs is not None]
        else:
            # Get all active feature sets
            all_sets = self.get_experiment_feature_sets()
            active_ids = [fs['feature_set_id'] for fs in all_sets if fs.get('is_active', True)]
            feature_sets = [
                self._get_feature_set_with_overrides(fs_id)
                for fs_id in active_ids
            ]
            feature_sets = [fs for fs in feature_sets if fs is not None]

        logger.info(f"Extracting {len(feature_sets)} feature sets for {len(segment_metadata):,} segment files")
        total_work = len(segment_metadata) * len(feature_sets)
        logger.info(f"Total extractions to perform: {total_work:,}")

        # Load existing extractions into memory for fast lookups
        logger.info("Loading existing extractions into memory...")
        cursor = self.db_conn.cursor()
        cursor.execute(f"""
            SELECT segment_id, feature_set_id, adc_type, stored_segment_length
            FROM {self.feature_table}
            WHERE extraction_status = 'completed'
        """)
        existing_extractions = set(cursor.fetchall())
        cursor.close()
        logger.info(f"Loaded {len(existing_extractions):,} existing extractions")

        # Track statistics
        total_extracted = 0
        extraction_times = []
        failed_extractions = []
        start_time = datetime.now()

        # Progress tracking
        print(f"\n📊 Progress:")
        print(f"   Total segment files: {len(segment_metadata):,}")
        print(f"   Feature sets per file: {len(feature_sets)}")
        print(f"   Total extractions: {total_work:,}")
        print()

        # Process each segment file
        for i, meta in enumerate(segment_metadata):
            # Update progress every 100 files
            if i % 100 == 0:
                pct = 100 * i / len(segment_metadata)
                elapsed = (datetime.now() - start_time).total_seconds()

                # Calculate rate and ETA
                if i > 0:
                    rate = total_extracted / elapsed
                    remaining = total_work - total_extracted
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60

                    print(f"\r   Files: {i:>6,}/{len(segment_metadata):,} ({pct:5.1f}%) | "
                          f"Extracted: {total_extracted:>8,}/{total_work:,} | "
                          f"Rate: {rate:6.1f}/s | "
                          f"ETA: {eta_minutes:5.1f}m   ", end='', flush=True)

            seg_path = str(meta['path'])

            for fs in feature_sets:
                fs_id = fs['feature_set_id']
                fs_name = fs['feature_set_name']

                try:
                    # Check if already exists (using in-memory set for speed)
                    if not force_reextract:
                        key = (meta['segment_id'], fs_id, meta['adc_type'], meta['stored_size'])
                        if key in existing_extractions:
                            continue

                    start_time = datetime.now()

                    # Extract features
                    feature_array = self._extract_feature_set_from_segment(seg_path, fs)

                    # Determine output path (mirror structure)
                    output_path = self._get_feature_output_path(
                        seg_path,
                        fs_id,
                        fs['set_n_value']
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save
                    np.save(output_path, feature_array)

                    # Record to database with ALL metadata
                    extraction_time = (datetime.now() - start_time).total_seconds()
                    self._store_extraction_result(
                        segment_id=meta['segment_id'],
                        file_id=meta['file_id'],
                        feature_set_id=fs_id,
                        n_value=fs['set_n_value'],
                        feature_file_path=str(output_path),
                        num_chunks=feature_array.shape[0],
                        extraction_time=extraction_time,
                        stored_segment_length=meta['stored_size'],
                        adc_type=meta['adc_type'],
                        adc_division=meta['division'],
                        original_segment_length=meta['original_length']
                    )

                    total_extracted += 1
                    extraction_times.append(extraction_time)

                except Exception as e:
                    logger.error(f"Failed: {seg_path}, set {fs_name}: {e}")
                    failed_extractions.append({
                        'segment_id': meta['segment_id'],
                        'feature_set_id': fs_id,
                        'error': str(e)
                    })

        # Final progress update
        print()  # New line after progress bar

        # Calculate statistics
        avg_time = np.mean(extraction_times) if extraction_times else 0
        total_time = sum(extraction_times)

        return {
            'success': True,
            'total_segments': len(segment_metadata),
            'total_feature_sets': len(feature_sets),
            'total_extracted': total_extracted,
            'failed_count': len(failed_extractions),
            'average_extraction_time': avg_time,
            'total_extraction_time': total_time,
            'failed_extractions': failed_extractions[:10]
        }

    def _check_existing_extraction(self, segment_id: int, feature_set_id: int,
                                   adc_type: str = None, stored_segment_length: int = None) -> bool:
        """Check if extraction already exists for specific ADC type and stored size"""
        cursor = self.db_conn.cursor()
        try:
            if adc_type is not None and stored_segment_length is not None:
                # Check for specific ADC type AND stored size (complete check)
                cursor.execute(f"""
                    SELECT 1 FROM {self.feature_table}
                    WHERE segment_id = %s
                      AND feature_set_id = %s
                      AND adc_type = %s
                      AND stored_segment_length = %s
                      AND extraction_status = 'completed'
                    LIMIT 1
                """, (segment_id, feature_set_id, adc_type, stored_segment_length))
            elif adc_type is not None:
                # Check for specific ADC type only (legacy)
                cursor.execute(f"""
                    SELECT 1 FROM {self.feature_table}
                    WHERE segment_id = %s
                      AND feature_set_id = %s
                      AND adc_type = %s
                      AND extraction_status = 'completed'
                    LIMIT 1
                """, (segment_id, feature_set_id, adc_type))
            else:
                # Legacy check (any ADC type)
                cursor.execute(f"""
                    SELECT 1 FROM {self.feature_table}
                    WHERE segment_id = %s
                      AND feature_set_id = %s
                      AND extraction_status = 'completed'
                    LIMIT 1
                """, (segment_id, feature_set_id))
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def _get_feature_set_with_overrides(self, feature_set_id: int) -> Optional[Dict]:
        """Get feature set configuration with per-feature channel overrides

        Returns dict with:
            - feature_set_id, feature_set_name, set_channel, set_n_value
            - features: list of dicts with feature_name, feature_order, feature_channel,
                       n_value_override, effective_channel, effective_n_value
        """
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Get feature set metadata
            cursor.execute("""
                SELECT
                    efs.feature_set_id,
                    fs.feature_set_name,
                    efs.data_channel as set_channel,
                    efs.n_value as set_n_value,
                    efs.windowing_strategy
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.feature_set_id = %s
                  AND efs.experiment_id = %s
            """, (feature_set_id, self.experiment_id))

            set_config = cursor.fetchone()
            if not set_config:
                return None

            # Get features with overrides
            cursor.execute("""
                SELECT
                    fsf.feature_set_feature_id,
                    fsf.feature_id,
                    f.feature_name,
                    f.behavior_type,
                    f.computation_function,
                    fsf.feature_order,
                    fsf.data_channel as feature_channel,
                    fsf.n_value_override,
                    COALESCE(fsf.data_channel, %s) as effective_channel,
                    COALESCE(fsf.n_value_override, %s) as effective_n_value
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                WHERE fsf.feature_set_id = %s
                ORDER BY fsf.feature_order
            """, (set_config['set_channel'], set_config['set_n_value'], feature_set_id))

            features = [dict(row) for row in cursor]

            return {
                'feature_set_id': set_config['feature_set_id'],
                'feature_set_name': set_config['feature_set_name'],
                'set_channel': set_config['set_channel'],
                'set_n_value': set_config['set_n_value'],
                'windowing_strategy': set_config['windowing_strategy'],
                'features': features
            }

        except psycopg2.Error as e:
            logger.error(f"Error getting feature set config: {e}")
            return None
        finally:
            cursor.close()

    def _load_channel_data(self, seg_data: Dict[str, np.ndarray], channel_spec: str) -> np.ndarray:
        """Load or compute channel data based on channel specification

        Args:
            seg_data: Dictionary with 'source_current' and 'load_voltage' arrays
            channel_spec: One of 'source_current', 'load_voltage', 'impedance', 'power'

        Returns:
            numpy array of requested channel data
        """
        if channel_spec == 'source_current':
            return seg_data['source_current']

        elif channel_spec == 'load_voltage':
            return seg_data['load_voltage']

        elif channel_spec == 'impedance':
            # Compute impedance = V / I
            V = seg_data['load_voltage']
            I = seg_data['source_current']
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = V / I
                Z[~np.isfinite(Z)] = 0  # Replace inf/nan with 0
            return Z

        elif channel_spec == 'power':
            # Compute power = V * I
            V = seg_data['load_voltage']
            I = seg_data['source_current']
            P = V * I
            return P

        elif channel_spec == 'source_current,load_voltage':
            # Multi-channel case - return both
            # This is for backward compatibility
            return {
                'source_current': seg_data['source_current'],
                'load_voltage': seg_data['load_voltage']
            }

        else:
            raise ValueError(f"Unknown channel specification: {channel_spec}")

    def _apply_statistic(self, chunk: np.ndarray, feature: Dict) -> float:
        """
        Apply statistical computation to a chunk

        Args:
            chunk: Data chunk (n_value,)
            feature: Feature definition with computation_function

        Returns:
            Scalar statistic value
        """
        comp_func = feature.get('computation_function')
        feature_name = feature.get('feature_name', 'unknown')

        if not comp_func:
            # Legacy features or missing function - infer from feature name
            name_lower = feature_name.lower()
            if 'variance' in name_lower or 'var' in name_lower:
                return float(np.var(chunk))
            elif 'mean' in name_lower:
                return float(np.mean(chunk))
            elif 'max' in name_lower:
                return float(np.max(chunk))
            elif 'min' in name_lower:
                return float(np.min(chunk))
            elif 'std' in name_lower:
                return float(np.std(chunk))
            else:
                logger.warning(f"Unknown computation for {feature_name}, using mean")
                return float(np.mean(chunk))

        # Standard numpy functions
        if comp_func == 'np.var':
            return float(np.var(chunk))
        elif comp_func == 'np.mean':
            return float(np.mean(chunk))
        elif comp_func == 'np.max':
            return float(np.max(chunk))
        elif comp_func == 'np.min':
            return float(np.min(chunk))
        elif comp_func == 'np.std':
            return float(np.std(chunk))
        elif comp_func == 'np.median':
            return float(np.median(chunk))
        elif comp_func == 'np.ptp':
            return float(np.ptp(chunk))
        else:
            raise ValueError(f"Unsupported computation function: {comp_func}")

    def _extract_non_overlapping(self, channel_data: np.ndarray, feature: Dict, n_value: int) -> np.ndarray:
        """
        Extract feature using non-overlapping chunks (repeated values)

        Args:
            channel_data: Input signal (segment_length,)
            feature: Feature dict
            n_value: Window size

        Returns:
            Array (segment_length,) with statistic repeated per chunk
        """
        segment_length = len(channel_data)
        num_chunks = segment_length // n_value

        if num_chunks == 0:
            raise ValueError(f"Segment length {segment_length} < n_value {n_value}")

        # Compute statistic per chunk
        chunk_values = []
        for i in range(num_chunks):
            chunk = channel_data[i*n_value:(i+1)*n_value]
            stat_value = self._apply_statistic(chunk, feature)
            chunk_values.append(stat_value)

        # Repeat each value n_value times
        output = np.repeat(chunk_values, n_value)

        # Handle remainder if segment_length not divisible by n_value
        if len(output) < segment_length:
            remainder_length = segment_length - len(output)
            output = np.pad(output, (0, remainder_length), mode='edge')

        return output

    def _sliding_variance_optimized(self, data: np.ndarray, n_value: int) -> np.ndarray:
        """
        Compute sliding window variance using incremental updates

        Args:
            data: Input signal (segment_length,)
            n_value: Window size

        Returns:
            Array (segment_length,) with per-sample variance

        Complexity: O(L) instead of O(L * N)
        """
        L = len(data)
        output = np.zeros(L)

        # First window (repeated for first N samples)
        window = data[0:n_value]
        window_sum = np.sum(window)
        window_sq_sum = np.sum(window ** 2)
        var = (window_sq_sum / n_value) - (window_sum / n_value) ** 2
        output[0:n_value] = var

        # Slide window
        for i in range(n_value, L):
            old_sample = data[i - n_value]
            new_sample = data[i]

            window_sum = window_sum - old_sample + new_sample
            window_sq_sum = window_sq_sum - old_sample**2 + new_sample**2

            var = (window_sq_sum / n_value) - (window_sum / n_value) ** 2
            output[i] = var

        return output

    def _extract_sliding_window(self, channel_data: np.ndarray, feature: Dict, n_value: int) -> np.ndarray:
        """
        Extract feature using sliding window

        Args:
            channel_data: Input signal (segment_length,)
            feature: Feature dict
            n_value: Window size

        Returns:
            Array (segment_length,) with per-sample statistics
        """
        segment_length = len(channel_data)

        # Check if we can use optimized variance
        comp_func = feature.get('computation_function')
        feature_name = feature.get('feature_name', '').lower()

        if comp_func == 'np.var' or 'variance' in feature_name:
            return self._sliding_variance_optimized(channel_data, n_value)

        # General sliding window (slower but works for any statistic)
        output = np.zeros(segment_length)

        # First window (repeated for first N samples)
        first_window = channel_data[0:n_value]
        first_stat = self._apply_statistic(first_window, feature)
        output[0:n_value] = first_stat

        # Slide window
        for i in range(n_value, segment_length):
            window = channel_data[i-n_value+1:i+1]
            output[i] = self._apply_statistic(window, feature)

        return output

    def _extract_single_feature(
        self,
        channel_data: np.ndarray,
        feature: Dict,
        n_value: int,
        windowing_strategy: str = 'non_overlapping'
    ) -> np.ndarray:
        """
        Extract single feature from channel data with per-index alignment

        Args:
            channel_data: Input signal (segment_length,)
            feature: Feature dict with behavior_type, computation_function, feature_name
            n_value: Window size (effective n_value from override or set default)
            windowing_strategy: 'non_overlapping' or 'sliding_window'

        Returns:
            Array of shape (segment_length,) with:
            - sample_wise: raw values
            - chunk_statistic/aggregate: values repeated per chunk or sliding
        """
        behavior = feature.get('behavior_type')

        if behavior == 'sample_wise':
            # Return channel data as-is (no windowing for sample-wise features)
            return channel_data

        elif behavior in ['chunk_statistic', 'aggregate']:
            if windowing_strategy == 'non_overlapping':
                return self._extract_non_overlapping(channel_data, feature, n_value)
            elif windowing_strategy == 'sliding_window':
                return self._extract_sliding_window(channel_data, feature, n_value)
            else:
                raise ValueError(f"Unknown windowing strategy: {windowing_strategy}")

        else:
            raise ValueError(f"Unknown behavior type: {behavior} for feature {feature['feature_name']}")

    def _extract_feature_set_from_segment(
        self,
        segment_file_path: str,
        feature_set_config: Dict
    ) -> np.ndarray:
        """
        Extract complete feature set from a segment file

        Args:
            segment_file_path: Path to segment .npz file
            feature_set_config: Config from _get_feature_set_with_overrides()
                               Must include 'windowing_strategy' key

        Returns:
            Array of shape (segment_length, num_features)
        """
        # Load raw segment data (always load both channels)
        seg_data = np.load(segment_file_path)

        # Check if it's a .npz archive or plain .npy array
        if isinstance(seg_data, np.lib.npyio.NpzFile):
            # .npz file with named arrays
            raw_data = {
                'source_current': seg_data['source_current'],
                'load_voltage': seg_data['load_voltage']
            }
        else:
            # .npy file with 2D array
            if len(seg_data.shape) == 2:
                if seg_data.shape[1] == 2:
                    # Legacy 2-column format (raw voltage, current only)
                    raw_data = {
                        'load_voltage': seg_data[:, 0],
                        'source_current': seg_data[:, 1]
                    }
                elif seg_data.shape[1] >= 6:
                    # New multi-column format with amplitude processing
                    # Columns: [raw_voltage, raw_current, amp1_voltage, amp1_current, amp2_voltage, amp2_current, ...]
                    # Use raw columns (0-1) as base, store all amplitude-processed columns
                    raw_data = {
                        'load_voltage': seg_data[:, 0],
                        'source_current': seg_data[:, 1]
                    }
                    # Store amplitude-processed columns for later use
                    # Format: amplitude_<method_index>_<channel>
                    num_amplitude_methods = (seg_data.shape[1] - 2) // 2
                    for i in range(num_amplitude_methods):
                        col_start = 2 + (i * 2)
                        raw_data[f'amplitude_{i}_voltage'] = seg_data[:, col_start]
                        raw_data[f'amplitude_{i}_current'] = seg_data[:, col_start + 1]
                else:
                    raise ValueError(
                        f"Unexpected segment file format: shape {seg_data.shape}. "
                        f"Expected (N, 2) for legacy or (N, 6+) for amplitude-processed"
                    )
            else:
                raise ValueError(
                    f"Unexpected segment file format: shape {seg_data.shape}. "
                    f"Expected 2D array"
                )

        segment_length = len(raw_data['source_current'])

        # Get windowing strategy
        windowing_strategy = feature_set_config.get('windowing_strategy', 'non_overlapping')

        # Extract each feature in order
        feature_outputs = []

        for feat in feature_set_config['features']:
            # Get effective channel and n-value
            channel_spec = feat['effective_channel']
            n_value = feat['effective_n_value']

            # Load or compute channel data
            channel_data = self._load_channel_data(raw_data, channel_spec)

            # Handle multi-channel case (backward compatibility)
            if isinstance(channel_data, dict):
                # Old multi-channel format - use first channel
                channel_data = channel_data['source_current']

            # Extract feature (returns array of length segment_length)
            feature_output = self._extract_single_feature(
                channel_data,
                feat,
                n_value,
                windowing_strategy
            )

            # Verify length
            if len(feature_output) != segment_length:
                raise ValueError(
                    f"Feature {feat['feature_name']} output length {len(feature_output)} "
                    f"!= segment length {segment_length}"
                )

            feature_outputs.append(feature_output)

        # Stack features horizontally
        if len(feature_outputs) == 1:
            output = feature_outputs[0].reshape(-1, 1)
        else:
            output = np.column_stack(feature_outputs)

        # Final shape: (segment_length, num_features)
        logger.debug(
            f"Extracted feature set {feature_set_config['feature_set_name']}: "
            f"shape {output.shape}, windowing={windowing_strategy}"
        )

        return output

    def _extract_with_mpcctl(self, segment: Dict, feature_set: Dict, channel: str) -> Dict:
        """Extract features using mpcctl command line tool"""
        try:
            # Build mpcctl command
            seg_path = segment['segment_file_path']
            fs_id = feature_set['feature_set_id']
            n_value = feature_set.get('n_value', 0)
            
            # Determine output path (mirror structure from segment_files to feature_files)
            output_path = self._get_feature_output_path(seg_path, fs_id, n_value)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd = [
                'mpcctl',
                'feature_extract',
                '--input', str(seg_path),
                '--output', str(output_path),
                '--feature-set', str(fs_id),
                '--channel', channel
            ]
            
            if n_value is not None and n_value > 0:
                cmd.extend(['--n-value', str(n_value)])
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Check if output file was created
                if output_path.exists():
                    # Count chunks if N > 0
                    num_chunks = 1
                    if n_value is not None and n_value > 0:
                        # Count files with pattern *_N_NNNNNNNN_*
                        pattern = f"*_N_{n_value:08d}_*"
                        num_chunks = len(list(output_path.parent.glob(pattern)))
                    
                    return {
                        'success': True,
                        'output_path': str(output_path),
                        'num_chunks': num_chunks
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Output file not created'
                    }
            else:
                return {
                    'success': False,
                    'error': f'mpcctl failed: {result.stderr}'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Feature extraction timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_with_python(self, segment: Dict, feature_set: Dict, channel: str) -> Dict:
        """Extract features using Python (fallback method)"""
        try:
            # This is a placeholder for Python-based feature extraction
            # In reality, this would load the segment data and compute features
            
            seg_path = segment['segment_file_path']
            fs_id = feature_set['feature_set_id']
            n_value = feature_set.get('n_value', 0)
            
            # Determine output path
            output_path = self._get_feature_output_path(seg_path, fs_id, n_value)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load segment data (placeholder)
            # segment_data = np.load(seg_path)
            
            # Extract features based on feature_set definition
            # features = self._compute_features(segment_data, feature_set, channel, n_value)
            
            # Save features
            # np.save(output_path, features)
            
            # For now, create empty file as placeholder
            output_path.touch()
            
            return {
                'success': True,
                'output_path': str(output_path),
                'num_chunks': 1,
                'warning': 'Python extraction not fully implemented - placeholder file created'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_feature_output_path(self, segment_path: str, feature_set_id: int, n_value: int) -> Path:
        """Generate output path for feature file"""
        # Convert segment path to Path object
        seg_path = Path(segment_path)
        
        # Replace 'segment_files' with 'feature_files' in path
        path_parts = seg_path.parts
        new_parts = []
        for part in path_parts:
            if part == 'segment_files':
                new_parts.append('feature_files')
            else:
                new_parts.append(part)
        
        # Build new path
        feature_path = Path(*new_parts)
        
        # Add feature set ID to filename
        stem = feature_path.stem
        suffix = feature_path.suffix

        if n_value is not None and n_value > 0:
            # Add N value to filename
            new_name = f"{stem}_FS{feature_set_id:04d}_N_{n_value:08d}{suffix}"
        else:
            new_name = f"{stem}_FS{feature_set_id:04d}{suffix}"
        
        return feature_path.parent / new_name
    
    def _store_extraction_result(self, segment_id: int, file_id: int,
                                 feature_set_id: int, n_value: int,
                                 feature_file_path: str, num_chunks: int,
                                 extraction_time: float,
                                 stored_segment_length: int = None,
                                 adc_type: str = None,
                                 adc_division: str = None,
                                 original_segment_length: int = None):
        """Store extraction result in database"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                INSERT INTO {self.feature_table}
                (experiment_id, segment_id, file_id, feature_set_id, n_value,
                 feature_file_path, num_chunks, extraction_status, extraction_time,
                 stored_segment_length, adc_type, adc_division, original_segment_length)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, segment_id, feature_set_id, n_value, adc_type)
                DO UPDATE SET
                    feature_file_path = EXCLUDED.feature_file_path,
                    num_chunks = EXCLUDED.num_chunks,
                    extraction_status = EXCLUDED.extraction_status,
                    extraction_time = EXCLUDED.extraction_time,
                    stored_segment_length = EXCLUDED.stored_segment_length,
                    adc_division = EXCLUDED.adc_division,
                    original_segment_length = EXCLUDED.original_segment_length,
                    created_at = CURRENT_TIMESTAMP
            """, (
                self.experiment_id, segment_id, file_id, feature_set_id, n_value,
                feature_file_path, num_chunks, 'completed', extraction_time,
                stored_segment_length, adc_type, adc_division, original_segment_length
            ))

            self.db_conn.commit()

        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error storing extraction result: {e}")
        finally:
            cursor.close()
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """Get status of feature extraction for this experiment"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Get overall counts
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_extractions,
                    COUNT(DISTINCT segment_id) as unique_segments,
                    COUNT(DISTINCT feature_set_id) as unique_feature_sets,
                    SUM(num_chunks) as total_chunks,
                    AVG(extraction_time) as avg_extraction_time,
                    SUM(extraction_time) as total_extraction_time
                FROM {self.feature_table}
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            overall = cursor.fetchone()
            
            # Get per-feature-set counts
            cursor.execute(f"""
                SELECT 
                    ft.feature_set_id,
                    fs.feature_set_name,
                    COUNT(*) as segment_count,
                    AVG(ft.extraction_time) as avg_time
                FROM {self.feature_table} ft
                JOIN ml_feature_sets_lut fs ON ft.feature_set_id = fs.feature_set_id
                WHERE ft.experiment_id = %s
                GROUP BY ft.feature_set_id, fs.feature_set_name
                ORDER BY ft.feature_set_id
            """, (self.experiment_id,))
            
            per_feature_set = [dict(row) for row in cursor]
            
            return {
                'overall': dict(overall) if overall else {},
                'per_feature_set': per_feature_set
            }
            
        except psycopg2.Error as e:
            logger.error(f"Error getting extraction status: {e}")
            return {'overall': {}, 'per_feature_set': []}
        finally:
            cursor.close()