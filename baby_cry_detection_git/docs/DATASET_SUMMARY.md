# Baby Cry Detection Dataset Summary

## Current Dataset Composition

**Total training samples: 9,264**
**Class distribution ratio: 0.968:1 (cry:non-cry)**

### Cry Samples: 4,557

#### 1. Original Cry Samples (822 samples)
- **Source**:
  - Donate-a-Cry corpus (696 files)
  - mahmudulhasan01/baby_crying_sound (Hugging Face) (69 unique files)
  - Baby Cry Sense Dataset (Kaggle - mennaahmed23) (57 unique files)
- **Location**: `data/cry/`
- **Format**: .wav, .ogg, .mp3, .3gp, .m4a files
- **Content**: Verified baby cry recordings from multiple sources
- **Categories**: Various cry types (hunger, pain, discomfort, tired, belly pain, burping, scared, lonely, cold/hot)

#### 2. ICSD Real Recordings (2,312 samples)
- **Source**: ICSD (Infant Cry Sound Dataset)
- **Location**: `data/cry_ICSD/`
- **Format**: .wav files
- **Content**: Real baby cry recordings with strong and weak labels
- **Composition**:
  - 424 strong label files (precise onset/offset timestamps)
  - 1,888 weak label files (clip-level annotations)
- **Purpose**: High-quality real recordings for binary classification and sound localization
- **Filtering**: Excluded synthetic audio (synth_*) and snoring files to maintain realistic acoustic properties
- **Note**: Both strong and weak labels are real recordings - "weak" just means no precise timing, which is fine for binary classification

#### 3. CryCeleb Dataset (1,423 samples)
- **Source**: CryCeleb2023 dataset
- **Location**: `data/cry_crycaleb/`
- **Format**: .wav files
- **Content**: Baby cry recordings from CryCeleb dataset
- **Purpose**: Binary classification and sound localization

### Non-Cry Samples: 4,707

#### 1. Baby Non-Cry Sounds (356 samples)
- **Location**: `data/baby_noncry/`
- **Format**: .wav files
- **Content**: Baby babbling, laughing, cooing, silence
- **Purpose**: Distinguish cries from other baby sounds

#### 2. Adult Speech (2,241 samples)
- **Source**: LibriSpeech train-clean-100 dataset (balanced subset)
- **Location**: `data/adult_speech/`
- **Format**: .flac files
- **Content**: Clean English speech recordings
- **Purpose**: Teach model to distinguish baby cries from adult conversation
- **Note**: Reduced from 4,481 to 2,241 files (50%) for better class balance

#### 3. Environmental Sounds (2,110 samples)
- **Source**: ESC-50 dataset
- **Location**: `data/environmental/`
- **Format**: .wav files
- **Content**: Common household sounds (vacuum, door slam, dishes, appliances, etc.)
- **Purpose**: Reduce false positives from home environment noises

### Background Noise for Augmentation: 1,960 samples
- **Location**: `data/noise/`
- **Source**: ESC-50 environmental sounds
- **Format**: .wav files
- **Purpose**: Mixed into cry samples during training for robustness
- **Note**: NOT used as training labels, only for augmentation

## Class Imbalance

**Current ratio: 0.968:1 (cry:non-cry)** - Well-balanced with minimal class imbalance

This well-balanced dataset is further optimized through:
1. WeightedRandomSampler during training (optional, may provide marginal improvement)
2. Class weights in loss function
3. On-the-fly data augmentation for cry samples
4. Stratified train/val/test splitting

### Balancing Options

**Current Status: Well-balanced (0.968:1 ratio)**

With the expanded dataset including CryCeleb and ICSD recordings, the dataset is well-balanced. Options:

**Option 1: Train with current balance (Recommended)**
- Use current 4,557 cry vs 4,707 non-cry ratio
- Excellent 0.968:1 class balance
- Pro: Perfect balance, all real recordings
- Pro: Optimal for sound localization (no synthetic audio)
- Pro: Realistic household acoustic properties
- Pro: Maximum dataset diversity

**Option 2: Reduce dataset size for faster training**
```bash
python scripts/balance_dataset.py --ratio 1.0 --max-samples 3000
```
Creates: 3,000 cry vs 3,000 non-cry (1:1 ratio)
- Pro: Faster training
- Con: Discards ~34% of samples
- Con: Less exposure to cry and environmental sound diversity

**Option 3: Collect more samples for specific scenarios**
- Record cry samples in YOUR specific home environment
- Add more household sounds from your setting
- Pro: Optimizes for your deployment environment
- Con: Time investment

## Data Splits

The dataset is split using stratified sampling:
- **Training**: 60% (5,468 samples)
- **Validation**: 20% (1,823 samples)
- **Test**: 20% (1,823 samples)

Stratified splitting ensures each split maintains the 1:1 class ratio.

## Data Augmentation

### On-the-Fly Augmentation (Training Only)

Applied randomly during training to cry samples:
- **Gaussian Noise**: Adds random noise (50% probability)
- **Time Stretching**: 0.8x to 1.2x speed (30% probability)
- **Pitch Shifting**: -2 to +2 semitones (30% probability)
- **Background Noise Mixing**: Mixes with noise/ files (70% probability)

### Benefits of On-the-Fly Augmentation:
1. No data leakage (applied after train/val/test split)
2. Different augmentation each epoch (more variety)
3. No storage overhead
4. Validation/test sets remain unaugmented (true performance)

### Why Pre-Augmented Data Was Removed:
The original `cry_aug/` directory (1,968 files) was deleted because:
- Pre-augmented versions could leak into validation/test sets
- Original cry + augmented version might appear in different splits
- Inflated evaluation metrics artificially
- On-the-fly augmentation is the industry standard

## Audio Preprocessing Pipeline

All audio files undergo:
1. **Loading**: torchaudio (primary) or librosa (fallback)
2. **Resampling**: Convert to 16kHz sample rate
3. **Mono Conversion**: Stereo files converted to mono
4. **Duration Normalization**: Pad or truncate to 3 seconds
5. **Feature Extraction**: Convert to log-mel spectrogram
   - 128 mel-frequency bins
   - 0-8000 Hz frequency range
   - 2048 FFT window size
   - 512 hop length
6. **Normalization**: Z-score normalization

## Supported Audio Formats

- .wav (primary format)
- .ogg
- .mp3
- .flac
- .m4a
- .3gp
- .webm
- .mp4

All formats are automatically converted to the standard preprocessing pipeline.

## Data Collection History

### Successful Additions:
1. **Original cry samples**: 696 files from Donate-a-Cry corpus
2. **Original non-cry**: Baby sounds from Freesound.org community
3. **LibriSpeech speech**: 4,481 adult speech samples (January 2025)
4. **ESC-50 environmental**: 1,960 environmental sounds (January 2025)
5. **Hugging Face baby cries**: 69 unique files from mahmudulhasan01/baby_crying_sound (October 2025)
6. **Kaggle Baby Cry Sense**: 57 unique files from mennaahmed23 dataset (October 2025)
7. **ICSD real recordings**: 2,312 files (424 strong + 1,888 weak) from ICSD dataset (October 2025)

### Failed/Filtered Attempts:
1. **AudioSet baby cries**: Downloaded 75 samples, but manual inspection revealed they were NOT baby cries (deleted)
2. **FSD50K baby cries**: Dataset search found 0 baby cry samples
3. **Hugging Face duplicates**: Removed 696 duplicates and 216 mislabeled files (baby laughs/silence)
4. **Kaggle duplicates**: Removed 1,047 duplicates from 1,054 downloaded files

## Dataset Quality Notes

### Cry Samples (4,557)
- **Quality**: High - manually verified and professionally annotated
- **Diversity**: Multiple babies, ages, cry types, and sources
- **Categories**: Hunger, pain, discomfort, tired, belly pain, burping, scared, lonely, cold/hot
- **Sources**: Original (822) + ICSD real recordings (2,312: strong + weak) + CryCeleb (1,423)
- **Improvement**: +554% more cry samples than original (696 -> 4,557)
- **Acoustic Quality**: All real recordings, no synthetic audio
- **Filtering Applied**: Removed 883 snoring files and 500 synthetic files from ICSD for optimal quality
- **Label Types**: Mix of strong labels (precise timing) and weak labels (clip-level) - all are real recordings

### Non-Cry Samples (4,557)
- **Quality**: High - professional datasets (LibriSpeech, ESC-50, Freesound.org)
- **Diversity**: Excellent - baby sounds, speech, household, environmental
- **Coverage**: Well-represents home environment sounds

## Recommendations

### Short-term:
1. Train with current perfectly balanced dataset (1:1 ratio) - optimal for all use cases
2. No need for WeightedRandomSampler with perfect balance
3. Monitor precision/recall to ensure both classes perform well
4. All real recordings make this optimal for sound localization applications

### Long-term:
1. Record cry samples in YOUR specific home environment for best localization accuracy
2. Test model performance with real-world audio from your deployment setting
3. Consider adding more household sounds specific to your environment
4. Validate sound localization performance with spatial audio recordings

## Verification Commands

Count all training data:
```bash
python scripts/count_training_data.py
```

Create balanced subset:
```bash
python scripts/balance_dataset.py --ratio 5.0
```

Analyze dataset:
```bash
python main.py analyze
```

## File Structure

```
data/
+-- cry/                   # 822 baby cry samples (TRAINING)
+-- cry_ICSD/              # 2,312 ICSD real baby cry samples (TRAINING)
                           # - 424 strong label files
                           # - 1,888 weak label files
                           # - Excludes snoring (883 removed)
                           # - Excludes synthetic (500 removed)
+-- cry_crycaleb/          # 1,423 CryCeleb baby cry samples (TRAINING)
+-- baby_noncry/           # 356 non-cry baby sounds (TRAINING)
+-- adult_speech/          # 2,241 adult speech samples (TRAINING)
+-- environmental/         # 2,110 environmental sounds (TRAINING)
+-- noise/                 # 1,960 sounds (AUGMENTATION ONLY)
+-- ICSD_cry/              # Original ICSD dataset (archived, not used)
```

**Important**: Only the first 6 directories are used as training labels. The `noise/` directory is exclusively for augmentation. The `ICSD_cry/` folder contains the original dataset structure (archived).

## Known Issues

1. **Class balance**: 1:1 ratio (perfectly balanced)
2. **ICSD filtering**: Removed 1,383 files (883 snoring + 500 synthetic) to maintain acoustic realism
3. **Multiple sources**: Cry samples from 5 different sources may have quality variations
4. **Format variety**: .wav (primary), .3gp, .mp3, .ogg, .m4a files
5. **Reduced adult speech**: Adult speech reduced to 50% for better balance
6. **Mixed label types**: Combines strong and weak labels (all are real recordings, just different annotation detail)

## Change Log

**October 2025:**
- Added 2,312 real cry samples from ICSD (424 strong + 1,888 weak labels)
- Created cry_ICSD folder with real recordings only
- Filtered out 883 snoring files from ICSD (not baby cries)
- Filtered out 500 synthetic files from ICSD (maintain acoustic realism for sound localization)
- Included weak label files (real recordings, clip-level annotations work fine for binary classification)
- Added 69 unique cry samples from Hugging Face (mahmudulhasan01/baby_crying_sound)
- Added 57 unique cry samples from Kaggle (Baby Cry Sense Dataset - mennaahmed23)
- Added 1,423 cry samples from CryCeleb2023 dataset
- Added .3gp audio format support
- Removed 1,743 duplicate files and 216 mislabeled files during dataset expansion
- Updated total cry samples: 696 -> 4,557 (+554%)
- Reduced adult_speech from 4,481 to 2,241 files (50% reduction for better balance)
- Updated class imbalance: 1:9.77 -> 1:1 (perfectly balanced)
- Consolidated cry_1 and cry_2 folders into main cry folder
- Total dataset: 5,379 -> 9,264 samples (all real recordings)

**January 2025:**
- Added 4,481 LibriSpeech adult speech samples
- Added 1,960 ESC-50 household environmental sounds
- Removed 1,968 pre-augmented cry samples (prevent data leakage)
- Updated audio format support (.webm, .mp4)
- Attempted AudioSet/FSD50K downloads (failed - no valid baby cries)
- Organized project structure (scripts/, docs/ folders)
