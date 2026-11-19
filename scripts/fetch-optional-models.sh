#!/usr/bin/env bash
# Download optional research models. This does not enable inference.
set -euo pipefail

root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
dest="$root/models/experimental"
mkdir -p -- "$dest"

verify() {
  local file="$1" expected_bytes="$2" expected_sha="$3"
  if [[ ! -f "$file" || -L "$file" ]]; then
    printf 'Refusing missing or symlink model: %s\n' "$file" >&2
    return 1
  fi
  if [[ -n "$expected_bytes" ]]; then
    local actual_bytes
    actual_bytes="$(stat -c '%s' -- "$file")"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
      printf 'Size mismatch for %s: got %s, expected %s\n' \
        "$file" "$actual_bytes" "$expected_bytes" >&2
      return 1
    fi
  fi
  if [[ -n "$expected_sha" ]]; then
    if ! printf '%s  %s\n' "$expected_sha" "$file" | sha256sum --check --status; then
      printf 'SHA-256 mismatch for %s\n' "$file" >&2
      return 1
    fi
  else
    printf 'WARNING: upstream does not publish a SHA-256 for %s; size check is NOT cryptographic verification.\n' "$file" >&2
    sha256sum -- "$file"
  fi
}

fetch() {
  local filename="$1" url="$2" expected_bytes="$3" expected_sha="$4"
  local output="$dest/$filename"
  if [[ -e "$output" || -L "$output" ]]; then
    verify "$output" "$expected_bytes" "$expected_sha"
    printf 'Already present: %s\n' "$output"
    return
  fi
  local temporary
  temporary="$(mktemp "$dest/.model-XXXXXXXX")"
  if ! curl --fail --location --retry 3 --show-error --silent \
    --proto '=https' --proto-redir '=https' --output "$temporary" "$url"; then
    rm -f -- "$temporary"
    printf 'Download failed: %s\n' "$url" >&2
    return 1
  fi
  if ! verify "$temporary" "$expected_bytes" "$expected_sha"; then
    rm -f -- "$temporary"
    return 1
  fi
  mv -- "$temporary" "$output"
  printf 'Downloaded: %s\n' "$output"
}

fetch_nanodet() {
  # NanoDet upstream v1.0.0-alpha-1: 4,793,615 bytes, no published SHA-256.
  fetch "nanodet-plus-m_320.onnx" \
    "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx" \
    "4793615" ""
}

fetch_vad() {
  # SHA-256 from Hugging Face's ONNX Community model object.
  fetch "silero-vad.onnx" \
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx" \
    "" "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808"
}

if [[ "$#" -ne 1 ]]; then
  printf 'Usage: bash scripts/fetch-optional-models.sh {nanodet|vad|all}\n' >&2
  exit 2
fi
case "$1" in
  nanodet) fetch_nanodet ;;
  vad) fetch_vad ;;
  all) fetch_nanodet; fetch_vad ;;
  *) printf 'Usage: bash scripts/fetch-optional-models.sh {nanodet|vad|all}\n' >&2; exit 2 ;;
esac
