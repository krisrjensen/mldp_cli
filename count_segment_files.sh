#!/bin/bash
# Count segment files for experiment 041 by size, type, and decimation

echo "Experiment 041 Segment File Statistics"
echo "======================================"
echo ""

base_path="/Volumes/ArcData/V3_database/experiment041/segment_files"

# Total count
total=$(find "$base_path" -name "*.npy" -type f | wc -l | tr -d ' ')
echo "Total files: $total"
echo ""

# By segment size
echo "Files by segment size:"
for size_dir in "$base_path"/S*/; do
    if [ -d "$size_dir" ]; then
        size=$(basename "$size_dir")
        count=$(find "$size_dir" -name "*.npy" -type f | wc -l | tr -d ' ')
        echo "  $size: $count files"
    fi
done
echo ""

# By data type (within first size dir as sample)
echo "Data types found (sample from S008192):"
sample_dir="$base_path/S008192"
if [ -d "$sample_dir" ]; then
    for type_dir in "$sample_dir"/T*/; do
        if [ -d "$type_dir" ]; then
            dtype=$(basename "$type_dir")
            echo "  $dtype"
        fi
    done
fi
echo ""

# By decimation (within first size/type dir as sample)
echo "Decimation factors found (sample from S008192/TRAW):"
sample_dir="$base_path/S008192/TRAW"
if [ -d "$sample_dir" ]; then
    for dec_dir in "$sample_dir"/D*/; do
        if [ -d "$dec_dir" ]; then
            dec=$(basename "$dec_dir")
            count=$(find "$dec_dir" -name "*.npy" -type f | wc -l | tr -d ' ')
            echo "  $dec: $count files"
        fi
    done
fi
echo ""

# Disk usage
echo "Disk usage:"
du -sh "$base_path"
