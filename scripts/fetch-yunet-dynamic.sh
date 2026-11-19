#!/usr/bin/env bash
# Fetch the upstream dynamic-input YuNet ONNX, not the 131-byte Git LFS pointer.
# No camera pixels or user data are transferred by this operation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$repo_root/models/yunet-dynamic.onnx}"
expected_sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"

mkdir -p "$(dirname "$destination")"
if [[ -f "$destination" ]] && printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --status -c -; then
  printf 'Verified existing dynamic YuNet: %s\n' "$destination"
  exit 0
fi

tmp="$(mktemp "$(dirname "$destination")/.yunet-dynamic.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
curl --fail --show-error --location --retry 3 --connect-timeout 15 --max-time 180 \
  --output "$tmp" "$url"
printf '%s  %s\n' "$expected_sha256" "$tmp" | sha256sum -c -
mv -f "$tmp" "$destination"
printf 'Verified dynamic YuNet: %s\n' "$destination"
