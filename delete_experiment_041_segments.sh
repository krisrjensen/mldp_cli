#!/bin/bash
# Delete experiment 041 segment files (2-column files that need regeneration)
# Author: Kristophor Jensen
# Date: 20251005_132600
# Version: 1.0.0.0

set -euo pipefail

# Configuration
EXPERIMENT_ID="041"
BASE_PATH="/Volumes/ArcData/V3_database/experiment${EXPERIMENT_ID}/segment_files"
BACKUP_DIR="/Users/kjensen/Documents/GitHub/mldp/mldp_cli/deletion_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default mode
DRY_RUN=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --help)
            echo "Usage: $0 [--execute] [--help]"
            echo ""
            echo "Options:"
            echo "  --execute    Actually delete files (default is dry-run)"
            echo "  --help       Show this help message"
            echo ""
            echo "By default, runs in dry-run mode (no files deleted)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Safety checks
echo "Experiment 041 Segment File Deletion Script"
echo "==========================================="
echo ""

if [ ! -d "$BASE_PATH" ]; then
    echo -e "${RED}ERROR: Segment path does not exist: $BASE_PATH${NC}"
    exit 1
fi

# Count files
TOTAL_FILES=$(find "$BASE_PATH" -name "*.npy" -type f | wc -l | tr -d ' ')
DISK_USAGE=$(du -sh "$BASE_PATH" | awk '{print $1}')

echo "Target directory: $BASE_PATH"
echo "Files to delete: $TOTAL_FILES"
echo "Disk space to free: $DISK_USAGE"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}MODE: DRY RUN (no files will be deleted)${NC}"
    echo -e "${YELLOW}Use --execute flag to actually delete files${NC}"
else
    echo -e "${RED}MODE: EXECUTE (files WILL BE DELETED)${NC}"
    echo -e "${RED}This action cannot be undone!${NC}"
    echo ""
    echo -n "Type 'DELETE' to confirm: "
    read confirmation
    if [ "$confirmation" != "DELETE" ]; then
        echo "Deletion cancelled"
        exit 0
    fi
fi

echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Export file list
FILE_LIST="$BACKUP_DIR/experiment_${EXPERIMENT_ID}_file_list_${TIMESTAMP}.txt"
echo "Exporting file list to: $FILE_LIST"
find "$BASE_PATH" -name "*.npy" -type f > "$FILE_LIST"
echo "  Saved $TOTAL_FILES file paths"
echo ""

# Create deletion log
LOG_FILE="$BACKUP_DIR/experiment_${EXPERIMENT_ID}_deletion_log_${TIMESTAMP}.txt"
echo "Deletion log: $LOG_FILE"
{
    echo "Experiment 041 Segment File Deletion"
    echo "Timestamp: $TIMESTAMP"
    echo "Mode: $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'EXECUTE')"
    echo "Base path: $BASE_PATH"
    echo "Total files: $TOTAL_FILES"
    echo "Disk usage: $DISK_USAGE"
    echo ""
} > "$LOG_FILE"

# Delete files
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would delete the following:"
    echo "  - All *.npy files in $BASE_PATH"
    echo "  - Keep directory structure intact"
    echo ""
    echo "Sample files that would be deleted:"
    find "$BASE_PATH" -name "*.npy" -type f | head -10
    echo "  ... and $(($TOTAL_FILES - 10)) more files"
else
    echo "Deleting files..."
    deleted_count=0
    failed_count=0

    while IFS= read -r file; do
        if rm "$file" 2>> "$LOG_FILE"; then
            ((deleted_count++))
            if [ $((deleted_count % 1000)) -eq 0 ]; then
                echo "  Deleted $deleted_count/$TOTAL_FILES files..."
            fi
        else
            ((failed_count++))
            echo "ERROR: Failed to delete $file" | tee -a "$LOG_FILE"
        fi
    done < "$FILE_LIST"

    echo ""
    echo -e "${GREEN}Deletion complete${NC}"
    echo "  Deleted: $deleted_count files"
    echo "  Failed: $failed_count files"
    echo ""

    # Report final disk usage
    NEW_DISK_USAGE=$(du -sh "$BASE_PATH" | awk '{print $1}')
    echo "Disk usage after deletion: $NEW_DISK_USAGE"

    # Log final stats
    {
        echo ""
        echo "Deletion Results:"
        echo "  Deleted: $deleted_count files"
        echo "  Failed: $failed_count files"
        echo "  Disk usage after: $NEW_DISK_USAGE"
        echo "  Completion time: $(date +%Y%m%d_%H%M%S)"
    } >> "$LOG_FILE"
fi

echo ""
echo "File list saved: $FILE_LIST"
echo "Log file saved: $LOG_FILE"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}This was a DRY RUN - no files were deleted${NC}"
    echo -e "${YELLOW}Run with --execute flag to actually delete files${NC}"
fi
