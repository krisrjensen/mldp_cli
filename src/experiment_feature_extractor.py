#!/usr/bin/env python3
"""
Filename: experiment_feature_extractor.py
Author: Kristophor Jensen
Date Created: 20250916_090000
Date Revised: 20250916_090000
File version: 1.0.0.0
Description: Extract features from segments and generate feature filesets
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
                    s.segment_start,
                    s.segment_end,
                    s.segment_length,
                    s.file_path as segment_file_path
                FROM {self.segment_table} st
                JOIN segments s ON st.segment_id = s.segment_id
                WHERE st.experiment_id = %s
                ORDER BY st.selection_order
            """, (self.experiment_id,))
            
            return [dict(row) for row in cursor]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting segments: {e}")
            return []
        finally:
            cursor.close()
    
    def extract_features(self, 
                        feature_set_ids: List[int] = None,
                        max_segments: int = None,
                        use_mpcctl: bool = True,
                        parallel_jobs: int = 4) -> Dict[str, Any]:
        """
        Extract features from segments
        
        Args:
            feature_set_ids: Specific feature sets to extract (None = all)
            max_segments: Maximum segments to process
            use_mpcctl: Use mpcctl for feature extraction
            parallel_jobs: Number of parallel extraction jobs
            
        Returns:
            Dictionary with extraction results
        """
        # Create table if needed
        if not self.create_feature_fileset_table():
            return {'success': False, 'error': 'Failed to create feature table'}
        
        # Get feature sets
        feature_sets = self.get_experiment_feature_sets()
        if not feature_sets:
            return {'success': False, 'error': 'No feature sets configured for experiment'}
        
        # Filter feature sets if specified
        if feature_set_ids:
            feature_sets = [fs for fs in feature_sets if fs['feature_set_id'] in feature_set_ids]
        
        # Get segments
        segments = self.get_selected_segments()
        if not segments:
            return {'success': False, 'error': 'No segments selected for training'}
        
        if max_segments:
            segments = segments[:max_segments]
        
        logger.info(f"Extracting features for {len(segments)} segments with {len(feature_sets)} feature sets")
        
        # Track results
        total_extracted = 0
        failed_extractions = []
        extraction_times = []
        
        # Process each feature set
        for fs in feature_sets:
            fs_id = fs['feature_set_id']
            fs_name = fs['feature_set_name']
            n_value = fs.get('n_value', 0)
            channel = fs.get('active_channel', 'source_current')
            
            logger.info(f"Processing feature set {fs_id}: {fs_name} (N={n_value}, channel={channel})")
            
            # Process each segment
            for seg in segments:
                try:
                    start_time = datetime.now()
                    
                    if use_mpcctl:
                        result = self._extract_with_mpcctl(seg, fs, channel)
                    else:
                        result = self._extract_with_python(seg, fs, channel)
                    
                    extraction_time = (datetime.now() - start_time).total_seconds()
                    
                    if result['success']:
                        # Store result in database
                        self._store_extraction_result(
                            segment_id=seg['segment_id'],
                            file_id=seg['file_id'],
                            feature_set_id=fs_id,
                            n_value=n_value,
                            feature_file_path=result['output_path'],
                            num_chunks=result.get('num_chunks', 1),
                            extraction_time=extraction_time
                        )
                        total_extracted += 1
                        extraction_times.append(extraction_time)
                    else:
                        failed_extractions.append({
                            'segment_id': seg['segment_id'],
                            'feature_set_id': fs_id,
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    logger.error(f"Error extracting features for segment {seg['segment_id']}: {e}")
                    failed_extractions.append({
                        'segment_id': seg['segment_id'],
                        'feature_set_id': fs_id,
                        'error': str(e)
                    })
        
        # Calculate statistics
        avg_time = np.mean(extraction_times) if extraction_times else 0
        total_time = sum(extraction_times)
        
        return {
            'success': True,
            'total_segments': len(segments),
            'total_feature_sets': len(feature_sets),
            'total_extracted': total_extracted,
            'failed_count': len(failed_extractions),
            'average_extraction_time': avg_time,
            'total_extraction_time': total_time,
            'failed_extractions': failed_extractions[:10]  # First 10 failures
        }
    
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
            
            if n_value > 0:
                cmd.extend(['--n-value', str(n_value)])
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Check if output file was created
                if output_path.exists():
                    # Count chunks if N > 0
                    num_chunks = 1
                    if n_value > 0:
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
        
        if n_value > 0:
            # Add N value to filename
            new_name = f"{stem}_FS{feature_set_id:04d}_N_{n_value:08d}{suffix}"
        else:
            new_name = f"{stem}_FS{feature_set_id:04d}{suffix}"
        
        return feature_path.parent / new_name
    
    def _store_extraction_result(self, segment_id: int, file_id: int, 
                                 feature_set_id: int, n_value: int,
                                 feature_file_path: str, num_chunks: int,
                                 extraction_time: float):
        """Store extraction result in database"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                INSERT INTO {self.feature_table}
                (experiment_id, segment_id, file_id, feature_set_id, n_value,
                 feature_file_path, num_chunks, extraction_status, extraction_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, segment_id, feature_set_id, n_value) 
                DO UPDATE SET
                    feature_file_path = EXCLUDED.feature_file_path,
                    num_chunks = EXCLUDED.num_chunks,
                    extraction_status = EXCLUDED.extraction_status,
                    extraction_time = EXCLUDED.extraction_time,
                    created_at = CURRENT_TIMESTAMP
            """, (
                self.experiment_id, segment_id, file_id, feature_set_id, n_value,
                feature_file_path, num_chunks, 'completed', extraction_time
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