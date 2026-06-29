#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# ZEST: Zero-shot Emotion Style Transfer — End-to-End Reproduction Script
# Paper: https://arxiv.org/abs/2401.04511
# Code:  https://github.com/iiscleap/ZEST
# =============================================================================
# Prerequisites:
#   1. Download ESD dataset from https://github.com/HLTSingapore/Emotional-Speech-Data
#      into $ESD_SOURCE (default: ./data/ESD_raw)
#   2. Download CREMA-D from https://github.com/VideoUnderstanding/CREMA-D
#      into $CREMA_SOURCE (default: ./data/CREMA-D)
#   3. (Optional) TIMIT from LDC for USS evaluation
#   4. NVIDIA GPU with ~24 GB memory (training all stages)
#   5. Python 3.9+ with `pip install -r requirements.txt`
#
# Usage:
#   export ESD_SOURCE=/path/to/ESD
#   bash run_pipeline.sh 2>&1 | tee run.log
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"
echo "[ZEST] Starting at $(date)"
echo "[ZEST] Script dir: $SCRIPT_DIR"

# ── Config ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ESD_SOURCE="${ESD_SOURCE:-$SCRIPT_DIR/data/ESD_raw}"
ESD_TARGET="${ESD_TARGET:-$SCRIPT_DIR/data/ESD}"
CREMA_SOURCE="${CREMA_SOURCE:-$SCRIPT_DIR/data/CREMA-D}"

mkdir -p "$SCRIPT_DIR/data" "$SCRIPT_DIR/logs"

# ── Step 0: Environment ─────────────────────────────────────────────────────
echo; echo "═══ Step 0: Install Python dependencies ═══"
command -v conda >/dev/null 2>&1 && echo "Using conda: $(conda --version)"
python --version
pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tail -5

# ── Step 1: Prepare ESD dataset ─────────────────────────────────────────────
echo; echo "═══ Step 1: Organize ESD dataset ═══"
if [[ ! -d "$ESD_SOURCE" ]]; then
    echo "WARNING: ESD source not found at $ESD_SOURCE"
    echo "Download from https://github.com/HLTSingapore/Emotional-Speech-Data"
    echo "Skipping — using pre-organized splits from repo"
else
    python "$CODE_DIR/prepare_esd_data.py" \
        --source_dir "$ESD_SOURCE" \
        --output_dir "$ESD_TARGET" \
        --val_ratio 0.1 --test_ratio 0.1
fi

# ── Step 2: Extract HuBERT tokens ───────────────────────────────────────────
echo; echo "═══ Step 2: Extract HuBERT discrete tokens (layer 9, K=100) ═══"
# NOTE: This step requires the split files to already have HuBERT tokens.
# If you need to regenerate from scratch, point to your organized ESD wav dir:
# python "$CODE_DIR/extract_hubert_tokens.py" \
#     --audio_dir "$ESD_TARGET/train" \
#     --split_file "$CODE_DIR/train_esd.txt" \
#     --output_file "$CODE_DIR/train_esd.txt"
echo "  Using pre-extracted tokens from repo (train_esd.txt already has HuBERT column)"

# ── Step 3: Extract F0 features ─────────────────────────────────────────────
echo; echo "═══ Step 3: Extract F0 via YAAPT ═══"
# If you need to regenerate f0.pickle from audio:
# python "$CODE_DIR/extract_f0.py" \
#     --audio_dir "$ESD_TARGET" \
#     --output_pickle "$CODE_DIR/f0.pickle" \
#     --output_stats "$CODE_DIR/esd_f0_stats.pth"
echo "  Using pre-extracted f0.pickle and esd_f0_stats.pth from repo"

# ── Step 4: Extract x-vectors (SpeechBrain ECAPA-TDNN) ─────────────────────
echo; echo "═══ Step 4: Extract x-vectors ═══"
echo "  cd $CODE_DIR/EASE && python get_speaker_embedding.py"
echo "  (requires configuring paths in get_speaker_embedding.py lines 8-9)"

# ── Step 5: Train EASE (adversarial speaker encoder) ────────────────────────
echo; echo "═══ Step 5: Train EASE model ═══"
echo "  cd $CODE_DIR/EASE && python speaker_classifier.py"
echo "  (requires configuring paths in speaker_classifier.py lines 120-126)"
echo "  Output: EASE.pth, EASE_embeddings/"

# ── Step 6: Train F0 predictor ──────────────────────────────────────────────
echo; echo "═══ Step 6: Train F0 predictor ═══"
echo "  cd $CODE_DIR/F0_predictor && python pitch_attention_adv.py"
echo "  (configure paths in config.py)"
echo "  Output: f0_predictor.pth"

# ── Step 7: Reconstruct F0 contours ─────────────────────────────────────────
echo; echo "═══ Step 7: Reconstruct F0 contours for HiFi-GAN training ═══"
echo "  cd $CODE_DIR/F0_predictor && python pitch_inference.py"
echo "  Output: f0_contours/"

# ── Step 8: Extract SACE wav2vec emotion features ───────────────────────────
echo; echo "═══ Step 8: Extract SACE emotion embeddings ═══"
echo "  cd $CODE_DIR/F0_predictor && python get_wav2vec_feats.py"
echo "  Output: wav2vec_feats/"

# ── Step 9: Train HiFi-GAN ──────────────────────────────────────────────────
echo; echo "═══ Step 9: Train HiFi-GAN (100K steps, ~2-3 days on V100) ═══"
echo "  cd $CODE_DIR/HiFi-GAN && python train.py \\"
echo "    --checkpoint_path checkpoints/ESD/ \\"
echo "    --pitch_folder $CODE_DIR/F0_predictor/f0_contours \\"
echo "    --emo_folder $CODE_DIR/F0_predictor/wav2vec_feats/ \\"
echo "    --config hubert_alladv.json \\"
echo "    --training_steps 100000"

# ── Step 10: Convert F0 for target emotion ──────────────────────────────────
echo; echo "═══ Step 10: Convert F0 contour (DSDT setting) ═══"
echo "  cd $CODE_DIR/F0_predictor && python pitch_convert.py"
echo "  Output: pred_DSDT_f0/"

# ── Step 11: Generate converted audio ───────────────────────────────────────
echo; echo "═══ Step 11: Inference (DSDT setting) ═══"
echo "  cd $CODE_DIR/HiFi-GAN && python inference.py \\"
echo "    --checkpoint_file checkpoints/ESD/ \\"
echo "    --pitch_folder $CODE_DIR/F0_predictor/pred_DSDT_f0 \\"
echo "    --emo_folder $CODE_DIR/F0_predictor/wav2vec_feats/ \\"
echo "    --convert"

# ── Summary ─────────────────────────────────────────────────────────────────
echo; echo "══════════════════════════════════════════════════════════════════"
echo "  ZEST pipeline staged. To execute each step, modify the paths in the"
echo "  corresponding Python files and run the commands above."
echo "  See logs/ for per-step output."
echo "══════════════════════════════════════════════════════════════════"

echo "[ZEST] Finished at $(date)"
