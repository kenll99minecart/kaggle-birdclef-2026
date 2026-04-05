# BirdCLEF 2026 Inference Pipeline

## Overview
The inference pipeline processes new audio data and generates species predictions in the competition submission format.

## Files
- `inference.py` - Main inference script
- `test_inference.py` - Test script using existing training data

## Usage

### For New Test Data
```python
from inference import BirdCLEFInference

# Initialize inference
inference = BirdCLEFInference()

# Run inference on test data
submission_df = inference.run_inference(
    test_audio_dir="path/to/test_soundscapes",
    output_path="submission.csv"
)
```

### Command Line
```bash
python inference.py
```

## Input Format
- Audio files in `.ogg` format (similar to `train_soundscapes/`)
- Files should be in a directory structure like:
  ```
  test_soundscapes/
  ├── audio_file_1.ogg
  ├── audio_file_2.ogg
  └── ...
  ```

## Output Format
- CSV file matching `sample_submission.csv` format
- Columns: `row_id` + 235 species probability columns
- Row IDs: `{filename}_{segment_start_time}`
- Probabilities: Float values between 0 and 1

## Processing Steps

1. **Model Loading**: Loads trained ensemble models and metadata
2. **Audio Segmentation**: Uses denoising to extract bird sound segments
   - Bandpass filtering (2kHz-10kHz)
   - Energy-based bird sound detection
   - Smart segment extraction with padding
3. **Embedding Extraction**: Uses Perch model (or MFCC fallback) on denoised segments
4. **Species Prediction**: Ensemble averaging across models per species
5. **Submission Generation**: Formats predictions for competition

## Denoising Pipeline

The inference pipeline includes advanced denoising to improve accuracy:

### **Audio Processing**
- **Bandpass Filter**: 2kHz-10kHz range (optimal for bird vocalizations)
- **Energy Detection**: RMS-based bird sound identification
- **Threshold**: 0.1 (configurable) for noise rejection
- **Minimum Duration**: 0.5s to avoid false positives
- **Smart Merging**: Combines segments within 0.5s of each other
- **Padding**: 0.2s padding around detected segments

### **Benefits**
- **Noise Reduction**: Removes background noise and non-bird sounds
- **Targeted Processing**: Only processes actual bird vocalizations
- **Improved Accuracy**: Higher quality embeddings from clean audio
- **Efficient**: Reduces processing of silent/empty segments

### **Fallback Handling**
- If no bird sounds detected, creates a fallback segment
- Ensures all audio files are processed
- Maintains submission format compatibility

## Features

- **Species-Level Classification**: 235 bird species
- **Ensemble Models**: Multiple models per species with averaging
- **Advanced Denoising**: Bandpass filtering + energy detection
- **Smart Segmentation**: Extracts only bird vocalizations
- **Flexible Embedding**: Perch model with MFCC fallback
- **Memory Efficient**: Processes files in batches
- **Complete Coverage**: Handles all species in competition format
- **Robust Fallbacks**: Handles silent/noisy audio gracefully

## Model Structure

Models are stored as a dictionary:
```python
{
    "species_id": [model1, model2, ...],  # Multiple models per species
    ...
}
```

## Performance

- **Embedding Dimension**: 1536 (Perch) or padded MFCC
- **Processing**: Smart bird sound detection + 5-second segments
- **Memory**: Efficient batch processing
- **Accuracy**: Species-level predictions with confidence scores
- **Denoising**: Improves embedding quality and prediction accuracy

## Notes

- Untrained species get default probability (0.004273504273504274)
- Audio is resampled to 32kHz and normalized to 5-second segments
- Ensemble averaging improves prediction robustness
- Handles missing species gracefully
