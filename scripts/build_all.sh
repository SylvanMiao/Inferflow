#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-4}"
CLEAN=0
BUILD_TESTS=1
ENABLE_CUDA=0

usage() {
  cat <<'USAGE'
Usage: scripts/build_all.sh [options]

Options:
  --clean        Remove existing build directories before building.
  --no-tests     Do not build Boost/jsoncpp smoke test targets in httpserver.
  --cuda         Build llama-engine with optional CUDA backend tests.
  -j, --jobs N   Parallel build jobs. Defaults to JOBS env or 4.
  -h, --help     Show this help.

Examples:
  scripts/build_all.sh
  scripts/build_all.sh --clean -j 8
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --no-tests)
      BUILD_TESTS=0
      shift
      ;;
    --cuda)
      ENABLE_CUDA=1
      shift
      ;;
    -j|--jobs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing: $path" >&2
    echo "$message" >&2
    exit 1
  fi
}

echo "[InferFlow] root: $ROOT_DIR"
echo "[InferFlow] jobs: $JOBS"

require_path "$ROOT_DIR/llama-engine/third_party/eigen3/Eigen/Core" \
  "Eigen is required under llama-engine/third_party/eigen3."
require_path "$ROOT_DIR/third_party/boost_1_84_0/stage/lib/cmake/Boost-1.84.0/BoostConfig.cmake" \
  "Boost is required under third_party/. Recreate it before building httpserver."
require_path "$ROOT_DIR/third_party/jsoncpp-1.9.6/build/lib/libjsoncpp.a" \
  "jsoncpp is required under third_party/. Recreate it before building httpserver."
require_path "$ROOT_DIR/third_party/mysqlclient/usr/lib/x86_64-linux-gnu/libmysqlclient.so" \
  "mysqlclient is required under third_party/. Recreate it or install default-libmysqlclient-dev."

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[InferFlow] cleaning build directories"
  rm -rf "$ROOT_DIR/llama-engine/build"
  rm -rf "$ROOT_DIR/notix/httpserver/build-http"
fi

echo "[InferFlow] configuring llama-engine"
cmake -S "$ROOT_DIR/llama-engine" \
  -B "$ROOT_DIR/llama-engine/build" \
  -DINFERFLOW_ENABLE_CUDA="$ENABLE_CUDA"

echo "[InferFlow] building llama-engine"
cmake --build "$ROOT_DIR/llama-engine/build" -j "$JOBS"

echo "[InferFlow] configuring httpserver"
cmake -S "$ROOT_DIR/notix/httpserver" \
  -B "$ROOT_DIR/notix/httpserver/build-http" \
  -DBUILD_HTTP_SERVER=ON \
  -DBUILD_LIBTESTS="$BUILD_TESTS"

echo "[InferFlow] building httpserver"
cmake --build "$ROOT_DIR/notix/httpserver/build-http" -j "$JOBS"

echo "[InferFlow] done"
echo "Run HTTP server:"
echo "  cd $ROOT_DIR/notix/httpserver"
echo "  ./build-http/http_server"
