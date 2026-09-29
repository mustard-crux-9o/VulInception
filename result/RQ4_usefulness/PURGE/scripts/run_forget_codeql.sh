#!/usr/bin/env bash
# CodeQL on a flat tree of forget-set .py files (same query packs as SecurityEval).
#
# Usage:
#   export CODEQL=.../SecurityEval/tools/codeql/codeql   # optional if default exists
#   export CODEQL_REPO=.../SecurityEval/tools/codeql-repo
#   ./run_forget_codeql.sh /abs/path/to/py_dir run_id
#
# Outputs under purge/eval_outputs/forget_codeql/{db,csv}/<run_id>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SE_ROOT="${SE_ROOT:-${PURGE_ROOT}/../SecurityEval}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source_root_with_py_files> <run_id>"
  exit 1
fi

SOURCE_ROOT="$(cd "$1" && pwd)"
RUN_ID="$2"

CODEQL="${CODEQL:-${SE_ROOT}/tools/codeql/codeql}"
CODEQL_REPO="${CODEQL_REPO:-${SE_ROOT}/tools/codeql-repo}"

if [[ ! -x "$CODEQL" && ! -f "$CODEQL" ]]; then
  echo "CODEQL must point to the codeql executable (got: $CODEQL)"
  exit 1
fi

PACK_BASE="${CODEQL_REPO}/python/ql/src/Security"
if [[ ! -d "$PACK_BASE" ]]; then
  echo "Missing pack dir: $PACK_BASE (set CODEQL_REPO)"
  exit 1
fi

OUT_BASE="${PURGE_ROOT}/eval_outputs/forget_codeql"
DB_DIR="${OUT_BASE}/db/${RUN_ID}"
RESULT_DIR="${OUT_BASE}/csv/${RUN_ID}"
mkdir -p "$RESULT_DIR" "${OUT_BASE}/db"

if [[ "${SKIP_DB_CREATE:-}" == "1" ]]; then
  echo "[1/2] SKIP_DB_CREATE=1 — reusing database: $DB_DIR"
  if [[ ! -f "$DB_DIR/codeql-database.yml" ]]; then
    echo "Database missing; run without SKIP_DB_CREATE first."
    exit 1
  fi
else
  echo "[1/2] Creating database: $DB_DIR"
  "$CODEQL" database create "$DB_DIR" --language=python --source-root="$SOURCE_ROOT" --overwrite
fi

echo "[2/2] Running Security packs (same set as SecurityEval run_securityeval_codeql.sh)..."
run_pack() {
  local pack="$1"
  local outfile="$2"
  echo "  analyze $pack -> $outfile"
  "$CODEQL" database analyze --search-path="$CODEQL_REPO" "$DB_DIR" "${PACK_BASE}/${pack}" \
    --format=csv --output="${RESULT_DIR}/${outfile}"
}

run_pack "CWE-020" "results_cwe_020.csv"
run_pack "CWE-022" "results_cwe_022.csv"
run_pack "CWE-078" "results_cwe_078.csv"
run_pack "CWE-079" "results_cwe_079.csv"
run_pack "CWE-089" "results_cwe_089.csv"
run_pack "CWE-090" "results_cwe_090.csv"
run_pack "CWE-094" "results_cwe_094.csv"
run_pack "CWE-116" "results_cwe_116.csv"
run_pack "CWE-117" "results_cwe_117.csv"
run_pack "CWE-209" "results_cwe_209.csv"
run_pack "CWE-215" "results_cwe_215.csv"
run_pack "CWE-285" "results_cwe_285.csv"
run_pack "CWE-295" "results_cwe_295.csv"
run_pack "CWE-312" "results_cwe_312.csv"
run_pack "CWE-326" "results_cwe_326.csv"
run_pack "CWE-327" "results_cwe_327.csv"
run_pack "CWE-352" "results_cwe_352.csv"
run_pack "CWE-377" "results_cwe_377.csv"
run_pack "CWE-502" "results_cwe_502.csv"
run_pack "CWE-601" "results_cwe_601.csv"
run_pack "CWE-611" "results_cwe_611.csv"
run_pack "CWE-643" "results_cwe_643.csv"
run_pack "CWE-730" "results_cwe_730.csv"
run_pack "CWE-732" "results_cwe_732.csv"
run_pack "CWE-776" "results_cwe_776.csv"
run_pack "CWE-798" "results_cwe_798.csv"
run_pack "CWE-918" "results_cwe_918.csv"

echo "Done. CSVs: $RESULT_DIR"
