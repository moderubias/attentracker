# Optional experimental models (not integrated)

This directory is intentionally not tracked in Git. Downloading an ONNX
weight file does not enable a detector, postprocessor, microphone observer,
or a new activity state. The application still loads only YuNet.

## Sources

NanoDet-Plus-m 320:
- Project: https://github.com/RangiLyu/nanodet
- Asset: https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx
- Release: v1.0.0-alpha-1, size 4,793,615 bytes
- License: Apache-2.0, subject to upstream conditions
- Integrity: size checked, NOT SHA-256 pinned (no release digest published
  in the GitHub asset metadata). The script prints the actual digest.
- Model output requires NanoDet-Plus class-score and bbox-distribution decoding,
  followed by non-maximum suppression. COCO "cell phone" detection alone
  cannot establish that the user is using a phone.

Silero VAD:
- ONNX: https://huggingface.co/onnx-community/silero-vad/blob/main/onnx/model.onnx
- Source project: https://github.com/snakers4/silero-vad
- License: MIT, subject to upstream conditions
- SHA-256:
  a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808
- Audio input and any audio model remain disabled in Attentracker. Voice
  activity detection is not speaker identification or proof of a conversation.

## Download on the build server

Run from local repository root, after pulling current GitHub changes:

    git pull --ff-only
    rsync -az --exclude='.git/' --exclude='target/' \
      --exclude='node_modules/' --exclude='.env' --exclude='.env.*' \
      ./ kiyoromi:~/attentracker-build/
    ssh kiyoromi 'cd ~/attentracker-build && bash scripts/fetch-optional-models.sh all'
    mkdir -p models/experimental
    rsync -av --progress \
      kiyoromi:~/attentracker-build/models/experimental/ \
      models/experimental/
    sha256sum models/experimental/*.onnx

This performs downloads, not compilation. No additional models are loaded by
the current monitoring worker, and no automatic microphone activation is added.

## Before enabling inference

1. Validate model outputs and supported ONNX operators under Tract on
   supported Windows/Linux targets, using a reproducible golden image/audio
   test with expected tensors and runtime measurements.
2. Implement decoding, NMS and calibrated confidence for NanoDet; do not
   classify "phone use" merely from a detected phone bounding box.
3. Obtain explicit microphone opt-in and ephemeral audio-buffer handling before
   enabling VAD. Verify the actual model state/input protocol.
4. Check upstream model licensing and verify artifact hashes independently.
5. Add dedicated opt-in switches for each non-YuNet model and benchmark their
   CPU/memory impact in parallel with the existing YuNet pipeline.
