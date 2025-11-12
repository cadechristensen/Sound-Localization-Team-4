# Training Guide: Pushing to 95%+ Accuracy

## Current Performance (Baseline)
- **Test Accuracy**: 94.01%
- **Precision**: 94.09%
- **Recall**: 94.01%
- **F1 Score**: 94.01%
- **ROC AUC**: 98.42%

**Confusion Matrix Analysis:**
- True Negatives: 867 (correct non-cry)
- True Positives: 875 (correct cry)
- False Positives: 75 (over-detection of cries) ← Main issue
- False Negatives: 36 (missed cries)

## Recommended Configuration Changes

### Option 1: Conservative Improvements (Safest)
Update these parameters in `src/config.py`:

```python
# Data Augmentation (Stronger robustness)
NOISE_FACTOR = 0.01  # From 0.005
TIME_STRETCH_RATE = [0.75, 1.25]  # From [0.8, 1.2]
PITCH_SHIFT_STEPS = [-3, 3]  # From [-2, 2]

# Training Schedule (More epochs)
NUM_EPOCHS = 80  # From 60
PATIENCE = 20  # From 15
WARMUP_EPOCHS = 5  # From 3

# Regularization (Less aggressive)
CNN_DROPOUT = 0.15  # From 0.2
TRANSFORMER_DROPOUT = 0.08  # From 0.1

# Class Balance (Reduce false positives)
CRY_WEIGHT_MULTIPLIER = 1.1  # From 1.3
```

**Expected Result**: 94.5-95% accuracy
**Training Time**: +10-15% longer
**Risk**: Very low

---

### Option 2: Aggressive Improvements (Maximum Performance)
All changes from Option 1, PLUS:

```python
# Model Capacity (More parameters)
D_MODEL = 384  # From 256
N_HEADS = 12  # From 8
N_LAYERS = 5  # From 4

# Batch Size (If GPU memory allows)
BATCH_SIZE = 256  # From 128

# Learning Rate (Finer optimization)
LEARNING_RATE = 5e-5  # From 1e-4

# Advanced (Add these new parameters)
FOCAL_LOSS_GAMMA = 2.0  # From 2.5 (less aggressive focusing)
```

**Expected Result**: 95-96% accuracy
**Training Time**: +30-40% longer
**GPU Memory**: ~2x usage (due to larger model + batch size)
**Risk**: Low-Medium (may need GPU memory adjustment)

---

## Step-by-Step Training Instructions

### 1. Backup Current Configuration
```bash
cp src/config.py src/config.py.backup
```

### 2. Update Configuration
Edit `src/config.py` with your chosen option parameters.

### 3. Start Training
```bash
python training/main.py train
```

### 4. Monitor Training
Watch for:
- **Val Accuracy** should reach >94% within 20-30 epochs
- **Val Loss** should decrease smoothly (no spikes)
- **Early Stopping** should trigger around epoch 50-70

If you see:
- **Overfitting** (train acc >> val acc): Increase dropout back to 0.2/0.1
- **Underfitting** (both accuracies plateau low): Increase model capacity more
- **OOM Error**: Reduce BATCH_SIZE to 128 or 64

### 5. Evaluate Results
```bash
python training/main.py evaluate --model-path results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth
```

### 6. Compare to Baseline
Check if:
- Test accuracy increased by 0.5%+ ✓
- False Positives decreased (was 75) ✓
- ROC AUC stayed above 98% ✓

---

## GPU Memory Requirements

### Current Model (256-dim)
- Batch 128: ~4-6 GB GPU
- Batch 256: ~8-10 GB GPU

### Larger Model (384-dim)
- Batch 128: ~6-8 GB GPU
- Batch 256: ~12-16 GB GPU

**If you have <16GB GPU:**
Use Option 1 (conservative) OR use Option 2 with `BATCH_SIZE = 128`

---

## Advanced: SpecAugment (For 96%+ Target)

If you want to push beyond 95%, implement SpecAugment:

1. Add to `src/config.py`:
```python
USE_SPECAUGMENT = True
FREQ_MASK_PARAM = 15  # Frequency masking width
TIME_MASK_PARAM = 35  # Time masking width
NUM_MASKS = 2  # Number of masks per spectrogram
```

2. Implement in `src/dataset.py` (requires code changes - ask Claude for help)

**Expected Gain**: Additional 0.5-1% accuracy

---

## Quantization for Raspberry Pi

**IMPORTANT**: After training with the new configuration:

1. **DO NOT use the auto-quantized model** from training
   - Has compatibility issues with PyTorch 2.8

2. **Use the non-quantized model directly**:
   ```bash
   # Copy to deployment
   cp results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth deployment/raspberry_pi/
   ```

3. **OR manually re-quantize** (if needed for size):
   ```bash
   # Ask Claude for a fresh quantization script
   # Must be done with PyTorch 2.8 to avoid compatibility issues
   ```

---

## Troubleshooting

### Training is too slow
- Reduce `BATCH_SIZE` to 64
- Reduce `D_MODEL` to 320
- Reduce `NUM_EPOCHS` to 60

### Out of Memory
- Reduce `BATCH_SIZE` to 64
- Use Option 1 (smaller model)
- Enable gradient checkpointing (ask Claude)

### Accuracy not improving
- Check data quality (look at misclassified files)
- Increase augmentation more aggressively
- Try different random seed: `torch.manual_seed(42)` → `torch.manual_seed(123)`

### Model overfitting
- Increase dropout: `CNN_DROPOUT = 0.25`, `TRANSFORMER_DROPOUT = 0.15`
- Add more augmentation
- Reduce model capacity slightly

---

## Expected Timeline

**Conservative (Option 1)**:
- Setup: 5 minutes
- Training: 2-3 hours (on GPU)
- Evaluation: 10 minutes
- **Total**: ~3 hours

**Aggressive (Option 2)**:
- Setup: 5 minutes
- Training: 4-5 hours (on GPU)
- Evaluation: 10 minutes
- **Total**: ~5 hours

---

## Success Criteria

Target metrics for 95% accuracy:
- ✓ Test Accuracy ≥ 95.0%
- ✓ False Positives ≤ 50 (was 75)
- ✓ False Negatives ≤ 40 (was 36)
- ✓ ROC AUC ≥ 98.5%
- ✓ Generalization: Train/Val/Test accuracies within 2% of each other

---

## After Training

1. **Commit the new model info**:
```bash
git add src/config.py
git commit -m "feat: Update config for 95% accuracy training run"
```

2. **Document results**:
   - Save confusion matrix plots
   - Record final metrics
   - Note any interesting findings

3. **Deploy**:
   - Use `model_best.pth` (non-quantized)
   - Test on Raspberry Pi
   - Monitor real-world performance

Good luck with your training! 🚀
