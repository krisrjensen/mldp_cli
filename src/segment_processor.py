#!/usr/bin/env python3
"""
Filename: segment_processor.py
Author(s): Kristophor Jensen
Date Created: 20250902_120000
Date Revised: 20250902_120000
File version: 0.0.0.1
Description: Segment fileset processor for experiment 18
"""

import os
import sys
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentFilesetProcessor:
    """
    Process segment filesets with decimation and data type conversions
    """
    
    # Default decimations and data types for experiment 18
    DEFAULT_DECIMATIONS = [1, 3, 7, 15, 31, 63, 127, 255, 511]
    DEFAULT_DATA_TYPES = ['ADC14', 'ADC12', 'ADC10', 'ADC8', 'ADC6']
    
    def __init__(self, experiment_id=18):
        """Initialize processor"""
        self.experiment_id = experiment_id
        self.base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')
        self.fileset_path = Path('/Volumes/ArcData/V3_database/fileset')
        self.adc_path = Path('/Volumes/ArcData/V3_database/adc_data')
        
        # Print output location
        print(f"\n📁 Output directory: {self.base_path}")
        
        # Statistics
        self.stats = {
            'files_created': 0,
            'files_skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Progress tracking
        self.progress_file = self.base_path / 'generation_progress.json'
        self.completed = self.load_progress()
    
    def connect_db(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
    
    def load_progress(self):
        """Load progress from checkpoint file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(json.load(f).get('completed', []))
        return set()
    
    def save_progress(self):
        """Save progress to checkpoint file"""
        with open(self.progress_file, 'w') as f:
            json.dump({'completed': list(self.completed)}, f)
    
    def get_experiment_files(self, file_range=None):
        """Get files from experiment_018_file_training_data"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            if file_range:
                min_id, max_id = map(int, file_range.split('-'))
                query = """
                    SELECT DISTINCT file_id 
                    FROM experiment_018_file_training_data 
                    WHERE experiment_id = %s AND file_id BETWEEN %s AND %s
                    ORDER BY file_id
                """
                cursor.execute(query, (self.experiment_id, min_id, max_id))
            else:
                query = """
                    SELECT DISTINCT file_id 
                    FROM experiment_018_file_training_data 
                    WHERE experiment_id = %s
                    ORDER BY file_id
                """
                cursor.execute(query, (self.experiment_id,))
            
            files = [row[0] for row in cursor.fetchall()]
            return files
            
        finally:
            cursor.close()
            conn.close()
    
    def get_file_segments(self, file_id, sizes=None):
        """Get segments for a file from database, optionally filtered by sizes"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Get segments from experiment training data
            if sizes:
                query = """
                    SELECT DISTINCT ds.segment_id, ds.beginning_index, 
                           ds.beginning_index + ds.segment_length as end_index, 
                           ds.segment_length
                    FROM data_segments ds
                    JOIN experiment_018_segment_training_data est 
                        ON ds.segment_id = est.segment_id
                    WHERE ds.experiment_file_id = %s 
                      AND est.experiment_id = 18
                      AND ds.segment_length = ANY(%s)
                    ORDER BY ds.segment_id
                """
                cursor.execute(query, (file_id, sizes))
            else:
                query = """
                    SELECT DISTINCT ds.segment_id, ds.beginning_index, 
                           ds.beginning_index + ds.segment_length as end_index, 
                           ds.segment_length
                    FROM data_segments ds
                    JOIN experiment_018_segment_training_data est 
                        ON ds.segment_id = est.segment_id
                    WHERE ds.experiment_file_id = %s 
                      AND est.experiment_id = 18
                    ORDER BY ds.segment_id
                """
                cursor.execute(query, (file_id,))
            
            segments = []
            for row in cursor.fetchall():
                segments.append({
                    'segment_id': row[0],
                    'start_sample': row[1],
                    'end_sample': row[2],
                    'length': row[3]
                })
            
            return segments
            
        finally:
            cursor.close()
            conn.close()
    
    def load_file_data(self, file_id, data_type):
        """Load file data based on data type"""
        if data_type == 'RAW':
            # Load from fileset (float64, 2 columns)
            filepath = self.fileset_path / f"{file_id:08d}.npy"
            if not filepath.exists():
                logger.error(f"Fileset file not found: {filepath}")
                return None
            return np.load(filepath)
        else:
            # Load from adc_data (uint32, 4 columns)
            filepath = self.adc_path / f"{file_id:08d}.npy"
            if not filepath.exists():
                logger.error(f"ADC file not found: {filepath}")
                return None
            return np.load(filepath)
    
    def apply_decimation(self, data, decimation):
        """Apply decimation preserving 2^N samples"""
        if decimation == 0:
            return data.copy()
        
        # Keep every (decimation+1)th sample
        step = decimation + 1
        decimated = data[::step]
        
        # Verify result is power of 2
        size = decimated.shape[0]
        if size & (size - 1) != 0:
            logger.warning(f"Decimation produced non-2^N size: {size}")
        
        return decimated
    
    def apply_data_type_conversion(self, data, data_type):
        """Convert data to specified ADC bit depth"""
        if data_type == 'RAW':
            # Already in correct format (float64)
            return data
        
        # Extract ADC columns [0, 1] for voltage and current
        adc_data = data[:, [0, 1]]
        
        if data_type == 'ADC14':
            # Scale UP from 12-bit to 14-bit (left-shift by 2)
            converted = np.clip(adc_data << 2, 0, 16383).astype(np.uint16)
        elif data_type == 'ADC12':
            # Use as-is (already 12-bit)
            converted = np.clip(adc_data, 0, 4095).astype(np.uint16)
        elif data_type == 'ADC10':
            # Right-shift by 2 bits
            converted = (adc_data >> 2).astype(np.uint16)
        elif data_type == 'ADC8':
            # Right-shift by 4 bits
            converted = (adc_data >> 4).astype(np.uint8)
        elif data_type == 'ADC6':
            # Right-shift by 6 bits
            converted = (adc_data >> 6).astype(np.uint8)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return converted
    
    def get_output_path(self, segment_id, file_id, size, decimation, data_type):
        """Generate output path with overwrite protection"""
        # Build directory structure
        size_dir = f"S{size:06d}"
        type_dir = f"T{data_type}"
        dec_dir = f"D{decimation:06d}"
        
        # Create full path
        full_dir = self.base_path / size_dir / type_dir / dec_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{segment_id:08d}_{file_id:08d}_D{decimation:06d}_T{data_type}_S{size:06d}.npy"
        filepath = full_dir / filename
        
        # Check if file exists
        if filepath.exists():
            logger.debug(f"File exists, skipping: {filepath}")
            self.stats['files_skipped'] += 1
            return None
        
        return filepath
    
    def process_single_segment(self, file_data, segment_info, file_id, data_type, decimation):
        """Process one segment with given parameters"""
        # Create unique key for progress tracking
        progress_key = f"{segment_info['segment_id']}_{data_type}_{decimation}"
        
        # Skip if already processed
        if progress_key in self.completed:
            logger.debug(f"Already processed: {progress_key}")
            self.stats['files_skipped'] += 1
            return False
        
        try:
            # Extract segment
            start = segment_info['start_sample']
            end = segment_info['end_sample']
            segment_data = file_data[start:end].copy()
            
            # Apply decimation
            if decimation > 0:
                segment_data = self.apply_decimation(segment_data, decimation)
            
            # Apply data type conversion
            if data_type != 'RAW':
                segment_data = self.apply_data_type_conversion(segment_data, data_type)
            
            # Get output path
            output_path = self.get_output_path(
                segment_info['segment_id'],
                file_id,
                segment_data.shape[0],
                decimation,
                data_type
            )
            
            if output_path:
                # Save file
                np.save(output_path, segment_data)
                logger.info(f"Created: {output_path}")
                self.stats['files_created'] += 1
                
                # Mark as completed
                self.completed.add(progress_key)
                
                # Save progress periodically
                if self.stats['files_created'] % 100 == 0:
                    self.save_progress()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_info['segment_id']}: {e}")
            self.stats['errors'] += 1
            return False
    
    def process_file(self, file_id, decimations, data_types, sizes=None, pbar=None):
        """Process all segments from a file"""
        segments = self.get_file_segments(file_id, sizes)
        
        if not segments:
            logger.warning(f"No segments found for file {file_id}")
            return
        
        logger.info(f"Processing file {file_id} with {len(segments)} segments")
        
        # Process each data type
        for data_type in data_types:
            # Load data once per data type
            file_data = self.load_file_data(file_id, data_type)
            
            if file_data is None:
                continue
            
            # Process each segment
            for segment in segments:
                # Process each decimation
                for decimation in decimations:
                    self.process_single_segment(
                        file_data, segment, file_id, data_type, decimation
                    )
                    
                    if pbar:
                        pbar.update(1)
    
    def generate(self, decimations=None, data_types=None, file_range=None, sizes=None, workers=16):
        """Main generation function"""
        # Use defaults if not specified
        if decimations is None:
            decimations = self.DEFAULT_DECIMATIONS
        if data_types is None:
            data_types = self.DEFAULT_DATA_TYPES
        
        # Record start time
        self.stats['start_time'] = datetime.now()
        
        # Get files to process
        files = self.get_experiment_files(file_range)
        
        if not files:
            logger.error("No files found to process")
            return self.stats
        
        # Get available segment sizes if needed
        if sizes:
            logger.info(f"  Segment sizes: {sizes}")
            # Estimate segments per file based on sizes
            avg_segments = 10  # Approximate based on filtered sizes
        else:
            # Get all unique segment sizes from database for display
            conn = self.connect_db()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT ds.segment_length
                FROM data_segments ds
                JOIN experiment_018_segment_training_data est 
                    ON ds.segment_id = est.segment_id
                WHERE est.experiment_id = 18
                ORDER BY ds.segment_length
            """)
            available_sizes = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            logger.info(f"  Segment sizes: All available - {available_sizes}")
            avg_segments = 13  # Average when processing all sizes
        
        # Calculate total operations
        total_operations = len(files) * avg_segments * len(decimations) * len(data_types)
        
        logger.info(f"Starting generation:")
        logger.info(f"  Files: {len(files)}")
        logger.info(f"  Decimations: {decimations}")
        logger.info(f"  Data types: {data_types}")
        logger.info(f"  Estimated operations: {total_operations}")
        
        # Process with progress bar
        with tqdm(total=total_operations, desc="Processing segments") as pbar:
            # Process files sequentially for now (can parallelize later)
            for file_id in files:
                self.process_file(file_id, decimations, data_types, sizes, pbar)
        
        # Save final progress
        self.save_progress()
        
        # Record end time
        self.stats['end_time'] = datetime.now()
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # Print summary
        print(f"\nGeneration Summary:")
        print(f"  📂 Output location: {self.base_path}")
        print(f"  ✅ Files created: {self.stats['files_created']:,}")
        print(f"  ⏭️  Files skipped: {self.stats['files_skipped']:,}")
        print(f"  ❌ Errors: {self.stats['errors']:,}")
        print(f"  ⏱️  Time elapsed: {elapsed/3600:.1f} hours")
        print(f"  📊 Rate: {self.stats['files_created']/max(1, elapsed):.1f} files/sec")
        
        # Show example files created
        if self.stats['files_created'] > 0:
            print(f"\n📁 Files were created in subdirectories like:")
            print(f"  {self.base_path}/S{262144:06d}/TRAW/D{0:06d}/")
            print(f"  {self.base_path}/S{131072:06d}/TADC*/D{1:06d}/")
            print(f"  etc.")
        
        return self.stats


def test_small_dataset():
    """Test with a small dataset"""
    processor = SegmentFilesetProcessor(experiment_id=18)
    
    # Test with 2 files, 2 decimations, 2 data types, and specific segment size
    stats = processor.generate(
        decimations=[1, 3],
        data_types=['ADC12', 'ADC8'],
        file_range='200-201',
        sizes=[262144],  # Only process 262144-sample segments
        workers=2
    )
    
    return stats


if __name__ == "__main__":
    # Run test if executed directly
    test_small_dataset()