#!/usr/bin/env bash
# Build lal + lalsimulation from the custom fork and produce local wheels.
#
# Usage (from repo root):
#   bash scripts/build_lal_wheels.sh
#
# After this succeeds, run the full test suite with:
#   uv run --group test pytest tests/ -v
#
# What this script does:
#   1. Checks / installs build dependencies (cmake, swig, gsl, fftw)
#   2. Clones neha2023sharma/lalsuite at the pinned commit
#   3. Builds lal + lalsimulation C libraries with cmake
#   4. Builds Python wheels for lal and lalsimulation
#   5. Places wheels in ./lal-wheels/ (referenced by pyproject.toml)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAL_FORK="https://github.com/neha2023sharma/lalsuite.git"
LAL_COMMIT="a13e410022d5db89c64983c2bf9c1c1da54f7cdb"
BUILD_DIR="$REPO_ROOT/.lal-build"
INSTALL_PREFIX="$BUILD_DIR/install"
WHEEL_DIR="$REPO_ROOT/lal-wheels"
CLONE_DIR="$BUILD_DIR/lalsuite"
NCPU=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# ── 1. System dependencies ──────────────────────────────────────────────────

echo "==> Checking build dependencies..."

missing=()
command -v cmake   &>/dev/null || missing+=(cmake)
command -v swig    &>/dev/null || missing+=(swig)
command -v gsl-config &>/dev/null || missing+=(gsl)
command -v pkg-config &>/dev/null || missing+=(pkg-config)
# fftw3 — check via pkg-config
pkg-config --exists fftw3 2>/dev/null || missing+=(fftw)

if [ ${#missing[@]} -gt 0 ]; then
    echo "==> Installing missing dependencies via Homebrew: ${missing[*]}"
    brew install "${missing[@]}"
fi

echo "==> cmake:  $(cmake --version | head -1)"
echo "==> swig:   $(swig -version 2>&1 | head -1)"
echo "==> gsl:    $(gsl-config --version)"
echo "==> fftw3:  $(pkg-config --modversion fftw3)"

# ── 2. Clone ─────────────────────────────────────────────────────────────────

mkdir -p "$BUILD_DIR"
if [ -d "$CLONE_DIR/.git" ]; then
    echo "==> lalsuite already cloned; checking out pinned commit..."
    git -C "$CLONE_DIR" fetch origin
else
    echo "==> Cloning lalsuite fork..."
    git clone "$LAL_FORK" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" checkout "$LAL_COMMIT"
echo "==> At commit: $(git -C "$CLONE_DIR" rev-parse HEAD)"

# ── 3. CMake configure ───────────────────────────────────────────────────────

echo "==> Configuring with cmake..."
cmake -B "$BUILD_DIR/cmake-build" -S "$CLONE_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLAL_ENABLE_SWIG=ON \
    -DLAL_ENABLE_SWIG_PYTHON=ON \
    -Dlal_ENABLE_SWIG_PYTHON=ON \
    -Dlalsimulation_ENABLE_SWIG_PYTHON=ON \
    -DLAL_PYTHON=ON \
    -Dlal_PYTHON=ON \
    -Dlalsimulation_PYTHON=ON \
    -DENABLE_OPENMP=OFF \
    -DLALFRAME=OFF \
    -DLALMETAIO=OFF \
    -DLALBURST=OFF \
    -DLALINSPIRAL=OFF \
    -DLALPULSAR=OFF \
    -DLALINFERENCE=OFF \
    -DLALDETCHAR=OFF \
    -DLALSTOCHASTIC=OFF \
    -DLALXML=OFF \
    2>&1 | tail -20

# ── 4. Build ─────────────────────────────────────────────────────────────────

echo "==> Building lal and lalsimulation (using $NCPU cores)..."
cmake --build "$BUILD_DIR/cmake-build" \
    --target lal lalsimulation \
    --parallel "$NCPU"

echo "==> Installing to $INSTALL_PREFIX..."
cmake --install "$BUILD_DIR/cmake-build" --component lal
cmake --install "$BUILD_DIR/cmake-build" --component lalsimulation

# ── 5. Build Python wheels ───────────────────────────────────────────────────

echo "==> Building Python wheels..."
mkdir -p "$WHEEL_DIR"

# The Python packages live under the install prefix's site-packages.
# We build wheels from the source Python directories (which cmake populates
# with the compiled _swig extension .so files after install).
PYTHON_PKGS=(
    "$CLONE_DIR/lal/python/lal"
    "$CLONE_DIR/lalsimulation/python/lalsimulation"
)

# Add the install prefix to the library search path so the extension modules
# can find libla*.dylib at wheel-build time.
export DYLD_LIBRARY_PATH="$INSTALL_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

for pkg_src in "${PYTHON_PKGS[@]}"; do
    if [ ! -d "$pkg_src" ]; then
        echo "ERROR: Expected Python package directory not found: $pkg_src"
        echo "  The cmake build may place it elsewhere — check $CLONE_DIR"
        exit 1
    fi
    echo "==> Building wheel for $(basename "$pkg_src")..."
    uv pip wheel "$pkg_src" \
        --wheel-dir "$WHEEL_DIR" \
        --no-deps \
        --config-setting="--build-option=--library-dir=$INSTALL_PREFIX/lib"
done

echo ""
echo "✓ Wheels written to $WHEEL_DIR:"
ls "$WHEEL_DIR"/*.whl

# ── 6. Install wheels + integration deps into the uv venv ───────────────────

echo ""
echo "==> Installing lal wheels into uv venv..."
cd "$REPO_ROOT"
uv sync --group test-integration
uv pip install "$WHEEL_DIR"/*.whl

echo ""
echo "✓ Done. Run the full test suite with:"
echo "  uv run --group test-integration pytest tests/test_likelihood.py -v"
