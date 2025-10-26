#!/bin/bash
#
# Filename: recover_experiment_018_distances.sh
# Author: Kristophor Jensen
# Date Created: 20251009_110000
# Description: Recover experiment_018 distance tables from pg_basebackup
#

set -e  # Exit on error

# PostgreSQL paths
PG_BIN="/opt/homebrew/opt/postgresql@15/bin"
export PATH="$PG_BIN:$PATH"

BACKUP_DIR="/Users/kjensen/Library/CloudStorage/GoogleDrive-kris.r.jensen@gmail.com/My Drive/V3_database/20251005_131837/postgresql_basebackup"
TEMP_DIR="/Volumes/ArcData/temp_recovery_$$"
TEMP_PORT=5433
LIVE_PORT=5432
DATABASE="arc_detection"
USER="kjensen"

echo "🔄 Starting recovery of experiment_018 distance tables..."
echo ""
echo "📁 Backup source: $BACKUP_DIR"
echo "📁 Temp directory: $TEMP_DIR"
echo "🔌 Temp PostgreSQL port: $TEMP_PORT"
echo ""

# Step 1: Create temporary directory
echo "📦 Step 1: Creating temporary directory..."
mkdir -p "$TEMP_DIR/data"
mkdir -p "$TEMP_DIR/dumps"

# Step 2: Extract base backup
echo "📦 Step 2: Extracting base backup (this may take 5-10 minutes)..."
cd "$TEMP_DIR/data"
tar -xzf "$BACKUP_DIR/base.tar.gz"

# Step 3: Extract WAL backup
echo "📦 Step 3: Extracting WAL backup..."
mkdir -p pg_wal
cd pg_wal
tar -xzf "$BACKUP_DIR/pg_wal.tar.gz"
cd ..

# Step 4: Start temporary PostgreSQL instance
echo "🚀 Step 4: Starting temporary PostgreSQL instance on port $TEMP_PORT..."

# Stop any existing temp instance first
pg_ctl -D "$TEMP_DIR/data" stop -m fast 2>/dev/null || true

# Start temp instance
pg_ctl -D "$TEMP_DIR/data" -o "-p $TEMP_PORT" -l "$TEMP_DIR/logfile" start

# Wait for it to start
echo "⏳ Waiting for temporary PostgreSQL to start..."
sleep 5

# Verify it's running
if ! pg_isready -p $TEMP_PORT -h localhost; then
    echo "❌ Failed to start temporary PostgreSQL instance"
    echo "📋 Check log: $TEMP_DIR/logfile"
    exit 1
fi

echo "✅ Temporary PostgreSQL started"

# Step 5: Dump specific tables
echo "💾 Step 5: Dumping experiment_018 distance tables..."

TABLES=(
    "experiment_018_distance_l1"
    "experiment_018_distance_l2"
    "experiment_018_distance_cosine"
    "experiment_018_distance_pearson"
)

for TABLE in "${TABLES[@]}"; do
    echo "   Dumping $TABLE..."

    # Check if table exists and has data
    ROW_COUNT=$(psql -h localhost -p $TEMP_PORT -d $DATABASE -U $USER -t -c "SELECT COUNT(*) FROM $TABLE" 2>/dev/null || echo "0")
    ROW_COUNT=$(echo $ROW_COUNT | xargs)  # Trim whitespace

    if [ "$ROW_COUNT" = "0" ]; then
        echo "   ⚠️  Warning: $TABLE is empty or doesn't exist in backup"
        continue
    fi

    echo "   📊 Found $ROW_COUNT records"

    # Dump table data only (no schema, we'll truncate existing)
    pg_dump -h localhost -p $TEMP_PORT -d $DATABASE -U $USER \
        --table=$TABLE \
        --data-only \
        --no-owner \
        --no-privileges \
        --format=custom \
        --file="$TEMP_DIR/dumps/${TABLE}.dump"

    echo "   ✅ Dumped to ${TABLE}.dump"
done

# Step 6: Stop temporary PostgreSQL
echo "🛑 Step 6: Stopping temporary PostgreSQL instance..."
pg_ctl -D "$TEMP_DIR/data" stop -m fast

# Step 7: Restore to live database
echo "🔄 Step 7: Restoring tables to live database (port $LIVE_PORT)..."

for TABLE in "${TABLES[@]}"; do
    DUMP_FILE="$TEMP_DIR/dumps/${TABLE}.dump"

    if [ ! -f "$DUMP_FILE" ]; then
        echo "   ⏭️  Skipping $TABLE (no dump file)"
        continue
    fi

    echo "   Restoring $TABLE..."

    # First, truncate the existing table
    echo "      Truncating existing table..."
    psql -h localhost -p $LIVE_PORT -d $DATABASE -U $USER -c "TRUNCATE TABLE $TABLE CASCADE" || {
        echo "      ❌ Failed to truncate $TABLE"
        continue
    }

    # Restore the data
    echo "      Restoring data..."
    pg_restore -h localhost -p $LIVE_PORT -d $DATABASE -U $USER \
        --data-only \
        --no-owner \
        --no-privileges \
        "$DUMP_FILE" || {
        echo "      ❌ Failed to restore $TABLE"
        continue
    }

    # Verify row count
    RESTORED_COUNT=$(psql -h localhost -p $LIVE_PORT -d $DATABASE -U $USER -t -c "SELECT COUNT(*) FROM $TABLE")
    RESTORED_COUNT=$(echo $RESTORED_COUNT | xargs)

    echo "      ✅ Restored $RESTORED_COUNT records"
done

# Step 8: Cleanup
echo "🧹 Step 8: Cleaning up temporary files..."
echo ""
echo "⚠️  Temporary files are in: $TEMP_DIR"
echo "   You can delete this directory after verifying the recovery:"
echo "   rm -rf $TEMP_DIR"
echo ""

# Step 9: Verification
echo "✅ Recovery complete!"
echo ""
echo "📊 Verification query:"
echo ""
echo "psql -h localhost -p $LIVE_PORT -d $DATABASE -c \""
echo "SELECT 'L1' as metric, COUNT(*) as records FROM experiment_018_distance_l1"
echo "UNION ALL SELECT 'L2', COUNT(*) FROM experiment_018_distance_l2"
echo "UNION ALL SELECT 'cosine', COUNT(*) FROM experiment_018_distance_cosine"
echo "UNION ALL SELECT 'pearson', COUNT(*) FROM experiment_018_distance_pearson"
echo "\""
echo ""
