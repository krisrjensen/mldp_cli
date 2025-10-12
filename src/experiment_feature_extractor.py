#!/usr/bin/env python3
"""
Filename: experiment_feature_extractor.py
Author: Kristophor Jensen
Date Created: 20250916_090000
Date Revised: 20251011_000000
File version: 1.2.0.1
Description: Extract features from segments and generate feature filesets
             Updated to support multi-column amplitude-processed segment files
             Updated to use normalized database schema with foreign keys
             Added tqdm progress bar
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
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ExperimentFeatureExtractor:
    """Extract features from segments and generate feature filesets"""
    
    def __init__(self, experiment_id: int, db_conn):
        self.experiment_id = experiment_id
        self.db_conn = db_conn
        self.segment_table = f"experiment_{experiment_id:03d}_segment_training_data"
        self.feature_table = f"experiment_{experiment_id:03d}_feature_fileset"

        # Read custom paths from database (if configured)
        custom_segment_path = None
        custom_feature_path = None
        try:
            cursor = db_conn.cursor()
            cursor.execute("""
                SELECT segment_data_base_path, feature_data_base_path
                FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))
            result = cursor.fetchone()
            if result:
                custom_segment_path = result[0]
                custom_feature_path = result[1]
            cursor.close()
        except Exception as e:
            logger.warning(f"Could not read custom paths from database: {e}")

        # Use custom paths if configured, otherwise use defaults
        if custom_segment_path and custom_feature_path:
            # Custom paths are full paths like "/custom/path/experiment041/segment_files"
            # Extract the base directory
            self.base_segment_path = Path(custom_segment_path).parent.parent
            self.base_feature_path = Path(custom_feature_path).parent.parent
            logger.info(f"Using CUSTOM data paths from database")
            logger.info(f"  Segment base: {self.base_segment_path}")
            logger.info(f"  Feature base: {self.base_feature_path}")
        else:
            # Default: /Volumes/ArcData/V3_database/
            self.base_segment_path = Path("/Volumes/ArcData/V3_database")
            self.base_feature_path = Path("/Volumes/ArcData/V3_database")
            logger.info(f"Using DEFAULT data paths")
        
    def create_feature_fileset_table(self):
        """Create the feature fileset tracking table with normalized schema"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.feature_table} (
                    segment_id INTEGER NOT NULL,
                    decimation_factor INTEGER NOT NULL,
                    data_type_id INTEGER NOT NULL,
                    amplitude_processing_method_id INTEGER NOT NULL,
                    experiment_feature_set_id BIGINT NOT NULL,
                    feature_set_feature_id BIGINT NOT NULL,
                    feature_file_path TEXT NOT NULL,
                    extraction_status_id INTEGER NOT NULL DEFAULT 1,
                    extraction_time_seconds FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (segment_id, decimation_factor, data_type_id, amplitude_processing_method_id, experiment_feature_set_id, feature_set_feature_id),
                    FOREIGN KEY (segment_id) REFERENCES data_segments(segment_id) ON DELETE CASCADE,
                    FOREIGN KEY (data_type_id) REFERENCES ml_data_types_lut(data_type_id),
                    FOREIGN KEY (amplitude_processing_method_id) REFERENCES ml_experiments_amplitude_methods(experiment_amplitude_id) ON DELETE CASCADE,
                    FOREIGN KEY (experiment_feature_set_id) REFERENCES ml_experiments_feature_sets(experiment_feature_set_id) ON DELETE CASCADE,
                    FOREIGN KEY (feature_set_feature_id) REFERENCES ml_feature_set_features(feature_set_feature_id) ON DELETE CASCADE,
                    FOREIGN KEY (extraction_status_id) REFERENCES ml_extraction_status_lut(status_id)
                )
            """)

            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_segment
                ON {self.feature_table}(segment_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_data_type
                ON {self.feature_table}(data_type_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_decimation
                ON {self.feature_table}(decimation_factor)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_amplitude
                ON {self.feature_table}(amplitude_processing_method_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_feature_set
                ON {self.feature_table}(experiment_feature_set_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_feature
                ON {self.feature_table}(feature_set_feature_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_status
                ON {self.feature_table}(extraction_status_id)
            """)

            # Composite index for distance calculation lookups
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.feature_table}_lookup
                ON {self.feature_table}(segment_id, data_type_id, decimation_factor, amplitude_processing_method_id)
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

    def _get_data_type_id(self, adc_type: str) -> int:
        """Convert adc_type string (e.g., 'TADC8') to data_type_id

        Args:
            adc_type: ADC type string like 'TADC8', 'TADC10', 'TRAW', etc.

        Returns:
            data_type_id from ml_data_types_lut

        Raises:
            ValueError: If adc_type is not found in lookup table
        """
        cursor = self.db_conn.cursor()
        try:
            # Remove 'T' prefix if present (TADC8 → ADC8)
            type_name = adc_type.upper().replace('T', '')

            cursor.execute("""
                SELECT data_type_id
                FROM ml_data_types_lut
                WHERE UPPER(data_type_name) = %s
            """, (type_name,))

            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Unknown data type: {adc_type} (cleaned: {type_name})")

            return result[0]
        finally:
            cursor.close()

    def _get_experiment_feature_set_id(self, feature_set_id: int) -> int:
        """Get experiment_feature_set_id from feature_set_id for current experiment

        Args:
            feature_set_id: Feature set ID from ml_feature_sets_lut

        Returns:
            experiment_feature_set_id from ml_experiments_feature_sets junction table

        Raises:
            ValueError: If feature set not configured for this experiment
        """
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT experiment_feature_set_id
                FROM ml_experiments_feature_sets
                WHERE experiment_id = %s AND feature_set_id = %s
            """, (self.experiment_id, feature_set_id))

            result = cursor.fetchone()
            if not result:
                raise ValueError(
                    f"Feature set {feature_set_id} not configured for experiment {self.experiment_id}"
                )

            return result[0]
        finally:
            cursor.close()

    def _get_amplitude_method_ids(self) -> List[int]:
        """Get list of amplitude_processing_method_ids configured for this experiment

        Returns:
            List of experiment_amplitude_id values from ml_experiments_amplitude_methods
        """
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT experiment_amplitude_id
                FROM ml_experiments_amplitude_methods
                WHERE experiment_id = %s
                ORDER BY experiment_amplitude_id
            """, (self.experiment_id,))

            results = cursor.fetchall()
            if not results:
                raise ValueError(f"No amplitude methods configured for experiment {self.experiment_id}")

            return [row[0] for row in results]
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

        # Get amplitude method IDs for this experiment
        amplitude_method_ids = self._get_amplitude_method_ids()
        logger.info(f"Amplitude methods configured: {len(amplitude_method_ids)}")

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
            SELECT segment_id, decimation_factor, data_type_id, amplitude_processing_method_id,
                   experiment_feature_set_id, feature_set_feature_id
            FROM {self.feature_table}
            WHERE extraction_status_id = 3
        """)
        existing_extractions = set(cursor.fetchall())
        cursor.close()
        logger.info(f"Loaded {len(existing_extractions):,} existing extractions")

        # Track statistics
        total_extracted = 0
        extraction_times = []
        failed_extractions = []
        start_time = datetime.now()

        # Progress tracking with tqdm
        print(f"\n📊 Starting feature extraction:")
        print(f"   Total segment files: {len(segment_metadata):,}")
        print(f"   Feature sets per file: {len(feature_sets)}")
        print(f"   Total extractions: {total_work:,}\n")

        # Process each segment file with progress bar
        with tqdm(total=len(segment_metadata), desc="Extracting features", unit="file") as pbar:
            for i, meta in enumerate(segment_metadata):
                seg_path = str(meta['path'])

                for fs in feature_sets:
                    fs_id = fs['feature_set_id']
                    fs_name = fs['feature_set_name']

                    try:
                        # Convert to normalized IDs once per iteration
                        data_type_id = self._get_data_type_id(meta['adc_type'])
                        experiment_feature_set_id = self._get_experiment_feature_set_id(fs_id)
                        decimation_factor = int(meta['division'].replace('D', ''))

                        # Get feature_set_feature_id values for this feature set
                        feature_set_feature_ids = [f['feature_set_feature_id'] for f in fs['features']]

                        # Check if already exists (using in-memory set for speed)
                        # Check if ALL (amplitude_method, feature) combinations exist
                        if not force_reextract:
                            all_exist = all(
                                (meta['segment_id'], decimation_factor, data_type_id, amp_id, experiment_feature_set_id, feat_id) in existing_extractions
                                for amp_id in amplitude_method_ids
                                for feat_id in feature_set_feature_ids
                            )
                            if all_exist:
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

                        # Record to database with normalized schema (one row per amplitude_method × feature)
                        extraction_time = (datetime.now() - start_time).total_seconds()

                        self._store_extraction_result(
                            segment_id=meta['segment_id'],
                            decimation_factor=decimation_factor,
                            data_type_id=data_type_id,
                            amplitude_method_ids=amplitude_method_ids,
                            experiment_feature_set_id=experiment_feature_set_id,
                            feature_set_feature_ids=feature_set_feature_ids,
                            feature_file_path=str(output_path),
                            extraction_time=extraction_time
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

                # Update progress bar after processing all feature sets for this segment
                pbar.update(1)

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

    def _check_existing_extraction(self, segment_id: int,
                                   decimation_factor: int,
                                   data_type_id: int,
                                   amplitude_processing_method_id: int,
                                   experiment_feature_set_id: int,
                                   feature_set_feature_id: int) -> bool:
        """Check if extraction already exists with normalized schema

        Args:
            segment_id: Segment ID
            decimation_factor: Decimation factor (0, 7, 15, 31, 63, etc.)
            data_type_id: Data type ID from ml_data_types_lut
            amplitude_processing_method_id: Amplitude method ID from ml_experiments_amplitude_methods
            experiment_feature_set_id: Junction table ID from ml_experiments_feature_sets
            feature_set_feature_id: Feature ID from ml_feature_set_features

        Returns:
            True if extraction exists and is completed, False otherwise
        """
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                SELECT 1 FROM {self.feature_table}
                WHERE segment_id = %s
                  AND decimation_factor = %s
                  AND data_type_id = %s
                  AND amplitude_processing_method_id = %s
                  AND experiment_feature_set_id = %s
                  AND feature_set_feature_id = %s
                  AND extraction_status_id = 3
                LIMIT 1
            """, (segment_id, decimation_factor, data_type_id, amplitude_processing_method_id,
                  experiment_feature_set_id, feature_set_feature_id))
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
        Extract complete feature set from a segment file FOR ALL AMPLITUDE METHODS

        Args:
            segment_file_path: Path to segment .npy file
            feature_set_config: Config from _get_feature_set_with_overrides()
                               Must include 'windowing_strategy' key

        Returns:
            Array of shape (segment_length, num_amplitude_methods)
            Each column represents the feature computed from that amplitude method
        """
        # Load segment data
        seg_data = np.load(segment_file_path)

        # Parse segment file to extract ALL amplitude-processed data
        # Build dict: {amplitude_method_index: {'load_voltage': array, 'source_current': array}}
        amplitude_data = {}

        if isinstance(seg_data, np.lib.npyio.NpzFile):
            # .npz format - not yet supported for multi-amplitude
            raise NotImplementedError("Multi-amplitude .npz format not yet supported")

        elif len(seg_data.shape) == 2:
            if seg_data.shape[1] == 2:
                # Legacy 2-column format (raw only)
                # Only method index 0 available
                amplitude_data[0] = {
                    'load_voltage': seg_data[:, 0],
                    'source_current': seg_data[:, 1]
                }
            elif seg_data.shape[1] >= 4 and seg_data.shape[1] % 2 == 0:
                # Multi-column format with amplitude processing
                # Format: [amp0_V, amp0_I, amp1_V, amp1_I, amp2_V, amp2_I, ...]
                # Each pair represents voltage and current for that amplitude method
                num_amplitude_methods = seg_data.shape[1] // 2

                for method_idx in range(num_amplitude_methods):
                    col_v = method_idx * 2
                    col_i = method_idx * 2 + 1
                    amplitude_data[method_idx] = {
                        'load_voltage': seg_data[:, col_v],
                        'source_current': seg_data[:, col_i]
                    }
            else:
                raise ValueError(
                    f"Unexpected segment file shape: {seg_data.shape}. "
                    f"Expected (N, 2) for legacy or (N, even_number>=4) for multi-amplitude"
                )
        else:
            raise ValueError(
                f"Unexpected segment dimensions: {seg_data.shape}. "
                f"Expected 2D array"
            )

        segment_length = len(amplitude_data[0]['source_current'])
        num_amplitude_methods = len(amplitude_data)

        # Get windowing strategy
        windowing_strategy = feature_set_config.get('windowing_strategy', 'non_overlapping')

        # Extract features for EACH feature, then for EACH amplitude method
        # Output will be shape (segment_length, num_features * num_amplitude_methods)
        # Column order: [feat0_amp0, feat0_amp1, feat1_amp0, feat1_amp1, ...]
        all_outputs = []

        for feat in feature_set_config['features']:
            channel_spec = feat['effective_channel']
            n_value = feat['effective_n_value']

            # Extract this feature for all amplitude methods
            feature_amplitude_outputs = []

            for method_idx in sorted(amplitude_data.keys()):
                method_data = amplitude_data[method_idx]

                # Load or compute channel data from THIS amplitude method
                channel_data = self._load_channel_data(method_data, channel_spec)

                # Handle multi-channel case
                if isinstance(channel_data, dict):
                    channel_data = channel_data['source_current']

                # Extract feature
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

                feature_amplitude_outputs.append(feature_output)

            # Add all amplitude method columns for this feature
            all_outputs.extend(feature_amplitude_outputs)

        # Stack all columns: features × amplitude_methods
        # Final shape: (segment_length, num_features * num_amplitude_methods)
        output = np.column_stack(all_outputs)
        num_features = len(feature_set_config['features'])

        logger.debug(
            f"Extracted feature set {feature_set_config['feature_set_name']}: "
            f"shape {output.shape} (length × {num_amplitude_methods} amplitude methods), "
            f"windowing={windowing_strategy}"
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
    
    def _store_extraction_result(self, segment_id: int,
                                 decimation_factor: int,
                                 data_type_id: int,
                                 amplitude_method_ids: List[int],
                                 experiment_feature_set_id: int,
                                 feature_set_feature_ids: List[int],
                                 feature_file_path: str,
                                 extraction_time: float):
        """Store extraction result in database with normalized schema

        Inserts one row per (amplitude_method, feature) combination. All rows
        share the same file path (the file contains all features as columns).

        Args:
            segment_id: Segment ID from data_segments table
            decimation_factor: Decimation factor (0, 7, 15, 31, 63, etc.)
            data_type_id: Data type ID from ml_data_types_lut
            amplitude_method_ids: List of amplitude method IDs for this experiment
            experiment_feature_set_id: Junction table ID from ml_experiments_feature_sets
            feature_set_feature_ids: List of feature_set_feature_id values for this feature set
            feature_file_path: Path to feature file on disk (same for all combinations)
            extraction_time: Time taken to extract features (seconds)
        """
        cursor = self.db_conn.cursor()
        try:
            # Insert one row per (amplitude_method, feature) combination
            for amplitude_method_id in amplitude_method_ids:
                for feature_set_feature_id in feature_set_feature_ids:
                    cursor.execute(f"""
                        INSERT INTO {self.feature_table}
                        (segment_id, decimation_factor, data_type_id, amplitude_processing_method_id,
                         experiment_feature_set_id, feature_set_feature_id, feature_file_path,
                         extraction_status_id, extraction_time_seconds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 3, %s)
                        ON CONFLICT (segment_id, decimation_factor, data_type_id, amplitude_processing_method_id,
                                     experiment_feature_set_id, feature_set_feature_id)
                        DO UPDATE SET
                            feature_file_path = EXCLUDED.feature_file_path,
                            extraction_status_id = 3,
                            extraction_time_seconds = EXCLUDED.extraction_time_seconds,
                            created_at = NOW()
                    """, (segment_id, decimation_factor, data_type_id, amplitude_method_id,
                          experiment_feature_set_id, feature_set_feature_id, feature_file_path,
                          extraction_time))

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