# BirdCLEF 2026 Pipeline

A complete pipeline for identifying species (birds, amphibians, mammals, reptiles, insects) in audio recordings from the Brazilian Pantanal.

## Overview

This pipeline implements the following steps:

1. **Audio-Label Matching**: Matches audio files to species labels in both `train_soundscapes_labels.csv` and `train.csv`
2. **Denoising**: Uses the existing `denoising.py` to extract important audio segments from soundtracks
3. **Embedding Generation**: Uses Google's Perch model (with MFCC fallback) to convert audio into embeddings
4. **Dataset Splitting**: Divides data into training and validation sets
5. **Bootstrapping**: Generates additional training data through bootstrapping
6. **Ensemble Training**: Trains multiple logistic regression models and averages their logits
7. **Evaluation**: Evaluates performance on validation datasets

## Dataset Structure

The dataset contains recordings from the Brazilian Pantanal with the following files:

- `train.csv`: Individual training audio files with species labels
- `train_soundscapes_labels.csv`: Labels for continuous soundscape recordings with time segments
- `taxonomy.csv`: Species ID to name/class mapping
- `train_audio/`: Individual audio files (.ogg format)
- `train_soundscapes/`: Continuous soundscape recordings

## Species Classes

The dataset includes 5 main classes:
- **Aves** (Birds)
- **Amphibia** (Amphibians) 
- **Mammalia** (Mammals)
- **Reptilia** (Reptiles)
- **Insecta** (Insects)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset is in the `birdclef-2026/` directory

## Usage

### Basic Usage

```python
from birdclef_pipeline import BirdCLEFPipeline

# Initialize pipeline
pipeline = BirdCLEFPipeline("birdclef-2026")

# Run complete pipeline
results, models = pipeline.run_complete_pipeline()
```

### Step-by-Step Usage

```python
# 1. Match audio to labels
train_df, soundscape_df = pipeline.match_audio_to_labels()

# 2. Load Perch model
pipeline.load_perch_model()

# 3. Prepare datasets
X_train, X_val, y_train, y_val = pipeline.prepare_datasets(train_df, soundscape_df)

# 4. Bootstrap training data
bootstrap_samples = pipeline.bootstrap_training_data(X_train, y_train, n_bootstrap=5)

# 5. Train ensemble models
models = pipeline.train_ensemble_models(bootstrap_samples)

# 6. Evaluate
val_predictions = pipeline.predict_ensemble(models, X_val)
results = pipeline.evaluate_performance(y_val, val_predictions)
```

## Pipeline Components

### 1. Audio Processing

- **Individual Audio**: Direct matching from `train.csv`
- **Soundscape Audio**: Segmented using the denoising pipeline
- **Denoising**: Bandpass filtering (2kHz-10kHz) with energy-based segmentation

### 2. Feature Extraction

- **Primary**: Google Perch model embeddings (32kHz audio)
- **Fallback**: MFCC features (13 coefficients)
- **Output**: Fixed-dimensional embedding vectors

### 3. Model Architecture

- **Ensemble Method**: Multiple logistic regression models
- **Training Strategy**: Bootstrap sampling with replacement
- **Prediction**: Average logits across all models
- **Multi-label**: Binary classification per species class

### 4. Evaluation Metrics

- **Overall Accuracy**: Multi-label classification accuracy
- **Per-class Accuracy**: Individual class performance
- **Threshold**: 0.5 default for binary conversion

## Configuration

### Denoising Parameters (in denoising.py)

- `threshold`: Energy threshold (0.0-1.0, default: 0.1)
- `min_duration`: Minimum segment duration (seconds, default: 0.5)
- `pad`: Padding around segments (seconds, default: 0.2)
- `frequency_range`: 2kHz-10kHz bandpass filter

### Model Parameters

- `n_bootstrap`: Number of bootstrap samples (default: 5)
- `validation_split`: 20% for validation
- `random_state`: 42 for reproducibility

## Output

The pipeline generates:

- `processed_data/extracted_segments/`: Denoised audio segments
- Performance metrics (accuracy, per-class results)
- Trained ensemble models
- Embedding arrays for all processed audio

## Notes

- The Perch model requires TensorFlow and TensorFlow Hub
- MFCC fallback used if Perch model unavailable
- Multi-label classification supports overlapping species calls
- Bootstrap sampling helps with class imbalance

## Performance

Expected performance metrics:
- Overall accuracy depends on class balance
- Per-class accuracy varies by species prevalence
- Ensemble method improves robustness over single models

## Troubleshooting

1. **Missing Dependencies**: Install all requirements.txt packages
2. **Audio Loading Errors**: Check audio file paths and formats
3. **Memory Issues**: Process data in smaller batches
4. **Perch Model Loading**: Falls back to MFCC if Perch unavailable
