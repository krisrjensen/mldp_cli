#!/usr/bin/env python3
"""
Filename: test_pipeline_direct.py
Author: Kristophor Jensen
Date Created: 20250916_100000
Date Revised: 20250916_100000
File version: 1.0.0.0
Description: Direct test of ML pipeline modules for experiment 41
"""

import sys
import psycopg2
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our modules
from experiment_file_selector import ExperimentFileSelector
from experiment_segment_selector import ExperimentSegmentSelector
from experiment_segment_pair_generator import ExperimentSegmentPairGenerator
from experiment_feature_extractor import ExperimentFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )

def test_file_selection(db_conn, experiment_id=41):
    """Test Step 1: File Selection"""
    print("\n" + "="*70)
    print("STEP 1: FILE SELECTION")
    print("="*70)
    
    try:
        selector = ExperimentFileSelector(experiment_id, db_conn)
        
        # Select files with random strategy, max 50 per label, seed=42
        result = selector.select_files(
            strategy='random',
            max_files_per_label=50,
            seed=42
        )
        
        if result['success']:
            print(f"✅ File selection successful!")
            print(f"   Total selected: {result['total_selected']}")
            print(f"   Strategy: {result['strategy']}")
            print(f"   Seed: {result['seed']}")
            
            if 'statistics' in result and result['statistics']:
                print("\n📊 Label distribution:")
                for label, count in result['statistics']['label_counts'].items():
                    print(f"   {label}: {count} files")
            
            return True
        else:
            print(f"❌ File selection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in file selection: {e}")
        return False

def test_segment_selection(db_conn, experiment_id=41):
    """Test Step 2: Segment Selection"""
    print("\n" + "="*70)
    print("STEP 2: SEGMENT SELECTION")
    print("="*70)
    
    try:
        # Configure the segment selector
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'arc_detection',
            'user': 'kjensen'
        }
        
        # First, make sure experiment config exists in database
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO ml_experiments (
                experiment_id, experiment_name, 
                segments_per_file_per_label, selection_strategy, random_seed
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id) DO UPDATE SET
                segments_per_file_per_label = EXCLUDED.segments_per_file_per_label,
                selection_strategy = EXCLUDED.selection_strategy,
                random_seed = EXCLUDED.random_seed
        """, (experiment_id, f'Experiment {experiment_id}', 3, 'position_balanced_per_file', 42))
        db_conn.commit()
        cursor.close()
        
        selector = ExperimentSegmentSelector(experiment_id, db_config)
        
        # Run selection
        result = selector.run_selection()
        
        if result['success']:
            print(f"✅ Segment selection successful!")
            print(f"   Total files: {result.get('num_files', 'N/A')}")
            print(f"   Total segments: {result.get('num_segments', 'N/A')}")
            
            if 'label_distribution' in result:
                print("\n📊 Label distribution:")
                for label_info in result['label_distribution']:
                    print(f"   {label_info['label_name']}: {label_info['num_segments']} segments from {label_info['num_files']} files")
            
            return True
        else:
            print(f"❌ Segment selection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in segment selection: {e}")
        return False

def test_pair_generation(db_conn, experiment_id=41):
    """Test Step 3: Segment Pair Generation"""
    print("\n" + "="*70)
    print("STEP 3: SEGMENT PAIR GENERATION")
    print("="*70)
    
    try:
        generator = ExperimentSegmentPairGenerator(experiment_id, db_conn)
        
        # Generate pairs with all combinations strategy
        result = generator.generate_pairs(
            strategy='all_combinations',
            seed=42
        )
        
        if result['success']:
            print(f"✅ Pair generation successful!")
            print(f"   Total segments: {result['total_segments']}")
            print(f"   Total pairs: {result['total_pairs']}")
            
            if 'statistics' in result and result['statistics']:
                stats = result['statistics']
                print("\n📊 Pair statistics:")
                print(f"   Same-label pairs: {stats.get('same_label_pairs', 0)}")
                print(f"   Different-label pairs: {stats.get('diff_label_pairs', 0)}")
            
            return True
        else:
            print(f"❌ Pair generation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in pair generation: {e}")
        return False

def test_feature_extraction(db_conn, experiment_id=41):
    """Test Step 5: Feature Extraction (limited test)"""
    print("\n" + "="*70)
    print("STEP 5: FEATURE EXTRACTION (LIMITED TEST)")
    print("="*70)
    
    try:
        extractor = ExperimentFeatureExtractor(experiment_id, db_conn)
        
        # Extract features for a small subset
        result = extractor.extract_features(
            max_segments=5,  # Only test with 5 segments
            use_mpcctl=False  # Use Python extraction
        )
        
        if result['success']:
            print(f"✅ Feature extraction test successful!")
            print(f"   Total segments: {result['total_segments']}")
            print(f"   Total feature sets: {result['total_feature_sets']}")
            print(f"   Total extracted: {result['total_extracted']}")
            
            if result['failed_count'] > 0:
                print(f"\n⚠️  Failed extractions: {result['failed_count']}")
            
            return True
        else:
            print(f"❌ Feature extraction failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return False

def verify_tables(db_conn, experiment_id=41):
    """Verify database tables were created and populated"""
    print("\n" + "="*70)
    print("DATABASE VERIFICATION")
    print("="*70)
    
    cursor = db_conn.cursor()
    
    tables = [
        f'experiment_{experiment_id:03d}_file_training_data',
        f'experiment_{experiment_id:03d}_segment_training_data',
        f'experiment_{experiment_id:03d}_segment_pairs',
        f'experiment_{experiment_id:03d}_feature_fileset'
    ]
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"✅ {table}: {count} records")
        except psycopg2.Error:
            print(f"⚠️  {table}: Not found")
    
    cursor.close()

def main():
    """Run the complete pipeline test"""
    print("\n" + "="*70)
    print("ML PIPELINE DIRECT TEST - EXPERIMENT 41")
    print("="*70)
    
    # Connect to database
    try:
        db_conn = connect_db()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    # Run tests
    results = []
    
    # Step 1: File Selection
    results.append(('File Selection', test_file_selection(db_conn)))
    
    # Step 2: Segment Selection
    results.append(('Segment Selection', test_segment_selection(db_conn)))
    
    # Step 3: Pair Generation
    results.append(('Pair Generation', test_pair_generation(db_conn)))
    
    # Step 5: Feature Extraction (limited test)
    results.append(('Feature Extraction', test_feature_extraction(db_conn)))
    
    # Verify database
    verify_tables(db_conn)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    # Close database connection
    db_conn.close()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)