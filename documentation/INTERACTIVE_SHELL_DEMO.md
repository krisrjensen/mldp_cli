# MLDP Interactive Shell Demo

## Launch the Shell

```bash
$ mldp

╔══════════════════════════════════════════════════════════════════════════════╗
║                         MLDP Interactive Shell v2.0                          ║
║                  Machine Learning Data Processing Platform                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Tab completion and auto-suggestions available                             ║
║  • Type 'help' for commands or 'help <command>' for details                  ║
║  • Current settings shown in prompt: mldp[exp18:l2]>                         ║
║  • Type 'exit' or Ctrl-D to leave                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ Connected to MLDP ecosystem at: /Users/kjensen/Documents/GitHub/mldp

mldp[exp18:l2]> 
```

## Example Session

### 1. Connect to Database
```
mldp[exp18:l2]> connect
✅ Connected to arc_detection@localhost:5432
```

### 2. Query Database
```
mldp[exp18:l2]> query SELECT COUNT(*) FROM segments
┌────────┐
│ count  │
├────────┤
│ 245678 │
└────────┘
📊 1 rows returned
```

### 3. Show Statistics
```
mldp[exp18:l2]> stats

📊 Statistics for experiment_018_distance_l2:
──────────────────────────────────────────────
  Total records:  1,234,567
  Min distance:   0.001234
  Q1 (25%):       0.234567
  Median (50%):   0.456789
  Q3 (75%):       0.678901
  Max distance:   0.999999
  Mean distance:  0.500000
  Std deviation:  0.123456
```

### 4. Find Closest Pairs
```
mldp[exp18:l2]> closest 5

🔍 Top 5 closest pairs (l2 distance):
┌──────────┬──────────┬──────────┬────────┬────────┐
│ Segment 1│ Segment 2│ Distance │ File 1 │ File 2 │
├──────────┼──────────┼──────────┼────────┼────────┤
│    12345 │    67890 │ 0.001234 │    123 │    456 │
│    23456 │    78901 │ 0.002345 │    234 │    567 │
│    34567 │    89012 │ 0.003456 │    345 │    678 │
│    45678 │    90123 │ 0.004567 │    456 │    789 │
│    56789 │    01234 │ 0.005678 │    567 │    890 │
└──────────┴──────────┴──────────┴────────┴────────┘
```

### 5. Change Settings
```
mldp[exp18:l2]> set distance cosine
✅ Current distance type set to: cosine

mldp[exp18:cosine]> show

⚙️  Current Settings:
────────────────────────────────────────
  Experiment ID:  18
  Distance Type:  cosine
  MLDP Root:      /Users/kjensen/Documents/GitHub/mldp
  Database:       ✅ Connected
  DB Name:        arc_detection
  DB User:        kjensen
```

### 6. Generate Visualization
```
mldp[exp18:cosine]> heatmap --version 7
🎨 Generating cosine heatmap (v7)...
✅ Heatmap generated!

mldp[exp18:cosine]> histogram --bins 100
📊 Generating cosine histogram...
✅ Histogram generated!
```

### 7. Export Query Results
```
mldp[exp18:cosine]> query SELECT * FROM segments WHERE file_id = 123 LIMIT 10
[...table output...]
📊 10 rows returned

mldp[exp18:cosine]> export segments_file123.csv
✅ Exported 10 rows to segments_file123.csv
```

### 8. Verify Tools
```
mldp[exp18:cosine]> verify

🔍 Verifying MLDP tools...
──────────────────────────────────────────────
  ✅ Distance Calculator      Found
  ✅ Distance DB Insert       Found
  ✅ Segment Visualizer       Found
  ✅ Database Browser         Found
  ✅ Experiment Generator     Found
  ✅ Segment Verifier         Found
  ✅ Data Cleaning Tool       Found
──────────────────────────────────────────────
Summary: 7 found, 0 missing
✅ All tools verified successfully!
```

## Key Features

### Tab Completion
Type part of a command and press Tab:
```
mldp[exp18:l2]> calc<TAB>
mldp[exp18:l2]> calculate
```

### Command History
Use ↑/↓ arrows to navigate previous commands:
```
mldp[exp18:l2]> ↑
mldp[exp18:l2]> query SELECT COUNT(*) FROM segments
```

### Auto-suggestions
Start typing and see suggestions from history:
```
mldp[exp18:l2]> que
                 ry SELECT COUNT(*) FROM segments  [gray suggestion]
```

### Dynamic Prompt
The prompt shows current configuration:
- `mldp[exp18:l2]>` - Experiment 18, L2 distance
- `mldp[exp17:cosine]>` - Experiment 17, Cosine distance

### SQL Support
Full PostgreSQL syntax supported:
```
mldp[exp18:l2]> query SELECT 
    file_id,
    COUNT(*) as segment_count,
    MIN(segment_id) as first_segment,
    MAX(segment_id) as last_segment
FROM segments
WHERE segment_size = 8192
GROUP BY file_id
ORDER BY segment_count DESC
LIMIT 10
```

### Export Formats
Export query results to CSV or JSON:
```
mldp[exp18:l2]> export results.csv    # CSV format
mldp[exp18:l2]> export results.json   # JSON format
```

## Comparison with Command-Line Mode

### Interactive Shell
```bash
$ mldp
mldp[exp18:l2]> connect
mldp[exp18:l2]> query SELECT * FROM segments LIMIT 5
mldp[exp18:l2]> calculate --segment-size 8192
mldp[exp18:l2]> exit
```

### Command-Line Mode
```bash
$ mldp database query --table segments --limit 5
$ mldp distance calculate --segment-size 8192 --distance-type l2
```

## Benefits of Interactive Shell

1. **Persistent Session**: Database connection maintained across commands
2. **Faster Workflow**: No need to type `mldp` prefix repeatedly
3. **Context Awareness**: Current settings visible in prompt
4. **Exploration**: Tab completion helps discover commands
5. **Data Analysis**: Query and immediately export results
6. **Efficiency**: Command history for repeating operations

## Installation

```bash
# Install with all features
pip install -r requirements.txt

# Required for advanced shell
pip install prompt-toolkit

# Basic shell works without prompt-toolkit
# But you'll miss tab completion and auto-suggestions
```

---

The interactive shell transforms MLDP from a collection of commands into a cohesive data analysis environment.