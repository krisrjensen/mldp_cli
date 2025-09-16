#!/usr/bin/env python3
"""
Filename: test_experiment_41_pipeline.py
Author: Kristophor Jensen
Date Created: 20250916_100000
Date Revised: 20250916_100000
File version: 1.0.0.0
Description: Test complete ML pipeline for experiment 41
"""

import subprocess
import sys
import time
from pathlib import Path
import psycopg2

def run_command(cmd):
    """Run a shell command and capture output"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("Output:")
        print(result.stdout)
    
    if result.stderr:
        print("Error output:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"⚠️  Command failed with exit code {result.returncode}")
        return False
    
    print("✅ Command completed successfully")
    return True

def check_database_connection():
    """Verify database connection"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        conn.close()
        print("✅ Database connection verified")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_pipeline():
    """Test the complete ML pipeline for experiment 41"""
    
    print("\n" + "="*70)
    print("TESTING ML PIPELINE FOR EXPERIMENT 41")
    print("="*70)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # 1. Check database connection
    if not check_database_connection():
        print("❌ Cannot proceed without database connection")
        return False
    
    # 2. Check mldp CLI is available
    mldp_path = Path(__file__).parent / "mldp"
    if not mldp_path.exists():
        print(f"❌ MLDP CLI not found at {mldp_path}")
        return False
    
    print(f"✅ MLDP CLI found at {mldp_path}")
    
    # Test each step of the pipeline
    steps = [
        {
            'name': 'Step 1: Select Training Files',
            'cmd': f'{mldp_path} shell -c "set experiment 41; select-files --strategy random --max-files 50 --seed 42"',
            'critical': True
        },
        {
            'name': 'Step 2: Generate Segment Training Data',
            'cmd': f'{mldp_path} shell -c "set experiment 41; generate-segment-training-data --segments-per-file 3 --strategy balanced --seed 42"',
            'critical': True
        },
        {
            'name': 'Step 3: Generate Segment Pairs',
            'cmd': f'{mldp_path} shell -c "set experiment 41; generate-segment-pairs --strategy all_combinations"',
            'critical': True
        },
        {
            'name': 'Step 4a: Generate Segment Fileset (test with small subset)',
            'cmd': f'{mldp_path} shell -c "set experiment 41; segment-generate exp18 --files 200-202 --sizes 8192"',
            'critical': False
        },
        {
            'name': 'Step 5: Generate Feature Fileset (test with max 10 segments)',
            'cmd': f'{mldp_path} shell -c "set experiment 41; generate-feature-fileset --max-segments 10"',
            'critical': False
        }
    ]
    
    results = []
    
    for i, step in enumerate(steps, 1):
        print(f"\n{'='*70}")
        print(f"EXECUTING {step['name']}")
        print('='*70)
        
        success = run_command(step['cmd'])
        results.append({
            'step': step['name'],
            'success': success,
            'critical': step['critical']
        })
        
        if not success and step['critical']:
            print(f"\n❌ Critical step failed: {step['name']}")
            print("Stopping pipeline test")
            break
        
        # Small delay between steps
        time.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        critical = " [CRITICAL]" if result['critical'] else ""
        print(f"{status} {result['step']}{critical}")
    
    all_passed = all(r['success'] for r in results)
    critical_passed = all(r['success'] for r in results if r['critical'])
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    elif critical_passed:
        print("⚠️  All critical steps passed, but some optional steps failed")
    else:
        print("❌ PIPELINE TEST FAILED - Critical steps did not complete")
    print("="*70)
    
    return all_passed

def verify_database_tables():
    """Verify that expected tables were created"""
    print("\n📊 Verifying database tables...")
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        cursor = conn.cursor()
        
        tables_to_check = [
            'experiment_041_file_training_data',
            'experiment_041_segment_training_data',
            'experiment_041_segment_pairs',
            'experiment_041_feature_fileset'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = %s
            """, (table,))
            
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✅ {table}: {count} records")
            else:
                print(f"  ⚠️  {table}: Table not found")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")

if __name__ == "__main__":
    # Run the test
    success = test_pipeline()
    
    # Verify results
    verify_database_tables()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)