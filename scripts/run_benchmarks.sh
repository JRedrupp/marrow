#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
BENCH_OUTPUT="$REPO_ROOT/.benchmarks/bench_output.json"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$RESULTS_DIR" "$(dirname "$BENCH_OUTPUT")"

# Collect git metadata for the result envelope.
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
REF="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"

echo "Running benchmarks for commit ${COMMIT:0:7} (${REF})..."

cd "$REPO_ROOT"

# Run the full benchmark suite via pytest.
pixi run pytest --benchmark -v \
    --benchmark-json "$BENCH_OUTPUT"

if [ ! -f "$BENCH_OUTPUT" ]; then
    echo "::error::pytest produced no benchmark JSON output"
    exit 1
fi

# Transform pytest-benchmark JSON into the marrow result envelope format.
OUT_FILE="$RESULTS_DIR/${COMMIT}.json"
python3 - "$BENCH_OUTPUT" "$COMMIT" "$TIMESTAMP" "$REF" "$OUT_FILE" <<'PYEOF'
import json
import sys

bench_json_path, commit, timestamp, ref, out_file = sys.argv[1:]

with open(bench_json_path) as f:
    data = json.load(f)

results = []
for b in data.get("benchmarks", []):
    name = b["name"]
    stats = b.get("stats", {})
    # pytest-benchmark reports mean in seconds; convert to nanoseconds.
    mean_s = stats.get("mean", 0.0)
    mean_ns = mean_s * 1e9
    # Throughput is stored in extra_info by the marrow conftest.py integration.
    throughput = b.get("extra_info", {}).get("throughput (GElems/s)")
    results.append({
        "name": name,
        "mean_ns": mean_ns,
        "throughput_gelems_s": throughput,
    })

envelope = {
    "commit": commit,
    "timestamp": timestamp,
    "ref": ref,
    "results": results,
}

with open(out_file, "w") as f:
    json.dump(envelope, f, indent=2)
    f.write("\n")

print(f"Written {out_file} ({len(results)} entries)")
PYEOF

cp "$OUT_FILE" "$RESULTS_DIR/latest.json"
echo "Written $RESULTS_DIR/latest.json"
echo "Benchmark run complete."
