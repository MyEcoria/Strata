#!/usr/bin/env bash
# Installs what is needed to build the Strata engine from source for an AMD Radeon card (HIP, gfx1201)
# on Ubuntu 26.10. Run it yourself: it uses sudo.   Usage: bash tools/install_amd_build_deps_ubuntu.sh
# Packages come from the Ubuntu archive (ROCm 7.2.4 hipcc, 10.0 libraries). The released Strata packages
# are built with other ROCm versions, so a source build may need small adjustments.
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

PKGS=(
  build-essential cmake ninja-build git python3-venv python3-pip pkg-config
  clang lld
  hipcc libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev
  rocm-cmake rocminfo rocm-smi
  libhiprand-dev librocprim-dev libhipcub-dev
)
# Not listed on purpose: libamd-comgr-dev. It conflicts with libamd-comgr-rocm-dev, which libamdhip64-dev needs.
# Optional, installed when the archive has them (not all do on this release).
OPTIONAL=(rocthrust-dev rocm-device-libs libhipsparse-dev libhipfft-dev)

echo "== apt update"
$SUDO apt-get update

echo "== required packages"
$SUDO apt-get install -y "${PKGS[@]}"

echo "== optional packages"
for p in "${OPTIONAL[@]}"; do
  if apt-cache policy "$p" 2>/dev/null | grep -q 'Candidate: [^(]'; then
    $SUDO apt-get install -y "$p" || echo "skipped $p (install failed)"
  else
    echo "skipped $p (not in the archive)"
  fi
done

echo "== GPU access (groups render and video)"
for g in render video; do
  if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then echo "ok: $USER is in $g"
  else $SUDO usermod -aG "$g" "$USER" && echo "added $USER to $g: log out and in again"; fi
done

echo "== check"
for t in hipcc cmake ninja g++ clang rocminfo; do
  if command -v "$t" >/dev/null; then echo "ok  $t: $(command -v "$t")"; else echo "MISSING $t"; fi
done
if command -v rocminfo >/dev/null; then
  out="$(rocminfo 2>/dev/null || true)"
  if grep -q 'gfx' <<<"$out"; then grep -E 'Name:.*gfx|Marketing' <<<"$out" | head -3
  else echo "rocminfo sees no GPU (check /dev/kfd and the groups)"; fi
fi
echo "done."
