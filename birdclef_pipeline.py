import pandas as pd
import numpy as np
import os
import glob
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import denoising functions
import sys
sys.path.append('/home/kenll/KaggleSOundCompetition')
from denoising import process_file

# For Perch embeddings (will need to install)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    print("TensorFlow/Hub not installed. Will install later.")

# For ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample

class BirdCLEFPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_audio_dir = self.data_dir / "train_audio"
        self.train_soundscapes_dir = self.data_dir / "train_soundscapes"
        self.output_dir = Path("processed_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.soundscape_labels_df = pd.read_csv(self.data_dir / "train_soundscapes_labels.csv")
        self.taxonomy_df = pd.read_csv(self.data_dir / "taxonomy.csv")
        
        # Initialize Perch model (will be loaded later)
        self.perch_model = None
        self.embedding_dim = None
        
        # Create label mapping from string codes to inat_taxon_id integers
        self.label_mapping = self._create_label_mapping()
        
    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from string codes to inat_taxon_id integers"""
        mapping = {}
        for _, row in self.taxonomy_df.iterrows():
            # Map primary_label (string or int) to inat_taxon_id (int)
            mapping[str(row['primary_label'])] = int(row['inat_taxon_id'])
        print(f"Created label mapping for {len(mapping)} species")
        return mapping
        
    def match_audio_to_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Match audio files to their labels in both datasets"""
        
        # Process individual training audio (train.csv)
        print("Processing individual training audio...")
        train_matched = []
        
        for _, row in self.train_df.iterrows():
            audio_path = self.train_audio_dir / row['filename']
            if audio_path.exists():
                train_matched.append({
                    'audio_path': str(audio_path),
                    'primary_label': str(row['primary_label']),  # Keep original string code
                    'primary_label_int': self.label_mapping.get(str(row['primary_label']), -1),  # Convert to int
                    'class_name': row['class_name'],
                    'scientific_name': row['scientific_name'],
                    'common_name': row['common_name'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'type': 'individual'  # Mark as individual recording
                })
        
        train_matched_df = pd.DataFrame(train_matched)
        print(f"Matched {len(train_matched_df)} individual training audio files")
        
        # Process soundscape audio (train_soundscapes_labels.csv)
        print("Processing soundscape audio...")
        soundscape_matched = []
        
        # Group by audio file to process all segments for each file
        grouped = self.soundscape_labels_df.groupby('filename')
        
        for filename, group in grouped:
            audio_path = self.train_soundscapes_dir / filename
            if audio_path.exists():
                # Process each time segment for this file
                for _, row in group.iterrows():
                    # Parse time segments
                    start_time = self._parse_time(row['start'])
                    end_time = self._parse_time(row['end'])
                    
                    # Parse multiple labels (separated by ;)
                    if pd.isna(row['primary_label']) or row['primary_label'] == '':
                        primary_labels = []
                        primary_labels_int = []
                    else:
                        primary_labels = [str(label.strip()) for label in str(row['primary_label']).split(';') if label.strip()]
                        # Convert to integers using mapping
                        primary_labels_int = [self.label_mapping.get(str(label.strip()), -1) for label in str(row['primary_label']).split(';') if label.strip()]
                    
                    # Get species information for each label
                    species_info = []
                    for i, label in enumerate(primary_labels):
                        # Try to find in taxonomy - handle both string and int labels
                        tax_info = self.taxonomy_df[self.taxonomy_df['primary_label'].astype(str) == str(label)]
                        if not tax_info.empty:
                            species_info.append({
                                'primary_label': label,
                                'primary_label_int': primary_labels_int[i] if i < len(primary_labels_int) else -1,
                                'class_name': tax_info.iloc[0]['class_name'],
                                'scientific_name': tax_info.iloc[0]['scientific_name'],
                                'common_name': tax_info.iloc[0]['common_name']
                            })
                        else:
                            print(f"Warning: Label {label} not found in taxonomy")
                    
                    soundscape_matched.append({
                        'audio_path': str(audio_path),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'primary_labels': primary_labels,  # List of string labels
                        'primary_labels_int': primary_labels_int,  # List of integer labels
                        'species_info': species_info,  # List of species dictionaries
                        'type': 'soundscape_segment'  # Mark as soundscape segment
                    })
        
        soundscape_matched_df = pd.DataFrame(soundscape_matched)
        print(f"Matched {len(soundscape_matched_df)} soundscape segments")
        
        return train_matched_df, soundscape_matched_df
    
    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '00:00:05' to seconds"""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    def extract_audio_segments(self, soundscape_df: pd.DataFrame) -> List[str]:
        """Extract relevant segments from soundscape audio using denoising"""
        print("Extracting audio segments from soundscapes...")
        
        extracted_files = []
        segments_dir = self.output_dir / "extracted_segments"
        segments_dir.mkdir(exist_ok=True)
        
        # Group by audio file to avoid processing the same file multiple times
        grouped = soundscape_df.groupby('audio_path')
        
        for audio_path, group in grouped:
            print(f"Processing {audio_path}")
            
            # Extract segments using denoising pipeline
            file_segments = process_file(
                audio_path, 
                str(segments_dir),
                threshold=0.1,
                min_duration=0.5,
                pad=0.2
            )
            
            extracted_files.extend(file_segments)
        
        print(f"Extracted {len(extracted_files)} audio segments")
        return extracted_files
    
    def load_perch_model(self):
        """Load Google's Perch model for audio embeddings"""
        print("Loading Perch model...")
        
        # For now, we'll use a placeholder. In practice, you'd load the actual Perch model
        # This would require downloading the model from TensorFlow Hub
        try:
            # Example URL (actual URL may differ)
            model_url = "https://tfhub.dev/google/bird-vocalization-model/1"
            self.perch_model = hub.load(model_url)
            print("Perch model loaded successfully")
        except Exception as e:
            print(f"Could not load Perch model: {e}")
            print("Will use alternative embedding method")
            
            # Fallback to MFCC embeddings
            self.perch_model = None
    
    def extract_embeddings(self, audio_files: List[str]) -> np.ndarray:
        """Extract embeddings from audio files"""
        print(f"Extracting embeddings from {len(audio_files)} audio files...")
        
        embeddings = []
        
        for i, audio_path in enumerate(audio_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(audio_files)} files")
            
            try:
                # Load audio
                y, sr = librosa.load(audio_path, sr=32000)  # Perch expects 32kHz
                
                if self.perch_model is not None:
                    # Use Perch model
                    embedding = self._get_perch_embedding(y, sr)
                else:
                    # Fallback to MFCC
                    embedding = self._get_mfcc_embedding(y, sr)
                
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Add zero embedding as fallback
                if self.perch_model is not None:
                    embeddings.append(np.zeros(self.embedding_dim))
                else:
                    embeddings.append(np.zeros(13))  # MFCC dimension
        
        return np.array(embeddings)
    
    def _get_perch_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Get embedding using Perch model"""
        # This is a placeholder - actual implementation depends on Perch model API
        # For now, return MFCC as placeholder
        return self._get_mfcc_embedding(audio, sr)
    
    def _get_mfcc_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Get MFCC embedding as fallback"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    
    def prepare_datasets(self, train_df: pd.DataFrame, soundscape_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation datasets"""
        print("Preparing datasets...")
        
        # Get all unique class names from taxonomy
        all_classes = sorted(self.taxonomy_df['class_name'].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        print(f"Found {len(all_classes)} classes: {all_classes}")
        
        # Process individual training audio
        train_audio_files = train_df['audio_path'].tolist()
        train_labels = []
        
        for _, row in train_df.iterrows():
            # Create multi-hot encoding for classes
            class_vector = np.zeros(len(all_classes))
            class_idx = class_to_idx[row['class_name']]
            class_vector[class_idx] = 1
            train_labels.append(class_vector)
        
        # Process soundscape segments
        print("Extracting soundscape segments...")
        extracted_segments = self.extract_audio_segments(soundscape_df)
        
        # Create labels for extracted segments
        soundscape_labels = []
        for i, segment_file in enumerate(extracted_segments):
            # For simplicity, use the labels from the first few segments
            # In practice, you'd want to match segments to their corresponding time labels
            if i < len(soundscape_df):
                row = soundscape_df.iloc[i]
                class_vector = np.zeros(len(all_classes))
                
                # Multi-label encoding for soundscape segments
                for species_info in row['species_info']:
                    class_name = species_info['class_name']
                    if class_name in class_to_idx:
                        class_idx = class_to_idx[class_name]
                        class_vector[class_idx] = 1
                
                soundscape_labels.append(class_vector)
            else:
                # Fallback: random class (this is a simplification)
                class_vector = np.zeros(len(all_classes))
                class_vector[0] = 1
                soundscape_labels.append(class_vector)
        
        # Combine all data
        all_audio_files = train_audio_files + extracted_segments
        all_labels = train_labels + soundscape_labels
        
        print(f"Total audio files: {len(all_audio_files)}")
        print(f"Individual files: {len(train_audio_files)}, Extracted segments: {len(extracted_segments)}")
        
        # Extract embeddings
        embeddings = self.extract_embeddings(all_audio_files)
        
        # Convert labels to numpy array
        labels = np.array(all_labels)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def bootstrap_training_data(self, X_train: np.ndarray, y_train: np.ndarray, n_bootstrap: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap samples for training"""
        print(f"Generating {n_bootstrap} bootstrap samples...")
        
        bootstrap_samples = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)
            bootstrap_samples.append((X_boot, y_boot))
        
        return bootstrap_samples
    
    def train_ensemble_models(self, bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]]) -> List[LogisticRegression]:
        """Train multiple logistic regression models on bootstrap samples"""
        print("Training ensemble models...")
        
        models = []
        n_classes = bootstrap_samples[0][1].shape[1]
        
        for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
            print(f"Training model {i+1}/{len(bootstrap_samples)}")
            
            # For multi-class classification, we'll train one model per class
            class_models = []
            
            for class_idx in range(n_classes):
                # Binary classification for this class
                y_binary = y_boot[:, class_idx]
                
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
                model.fit(X_boot, y_binary)
                class_models.append(model)
            
            models.append(class_models)
        
        return models
    
    def predict_ensemble(self, models: List[LogisticRegression], X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble averaging"""
        print("Making ensemble predictions...")
        
        n_classes = len(models[0])
        n_samples = X.shape[0]
        
        # Collect predictions from all models
        all_predictions = np.zeros((len(models), n_samples, n_classes))
        
        for model_idx, class_models in enumerate(models):
            for class_idx, model in enumerate(class_models):
                # Get probabilities for this class
                probs = model.predict_proba(X)[:, 1]  # Probability of positive class
                all_predictions[model_idx, :, class_idx] = probs
        
        # Average predictions across models
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
    
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
        """Evaluate model performance"""
        print("Evaluating performance...")
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # Calculate per-class metrics
        n_classes = y_true.shape[1]
        class_accuracies = []
        
        for i in range(n_classes):
            if np.sum(y_true[:, i]) > 0:  # Only if class exists in true labels
                class_acc = accuracy_score(y_true[:, i], y_pred_binary[:, i])
                class_accuracies.append(class_acc)
        
        results = {
            'overall_accuracy': accuracy,
            'mean_class_accuracy': np.mean(class_accuracies),
            'class_accuracies': class_accuracies
        }
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Mean Class Accuracy: {np.mean(class_accuracies):.4f}")
        
        return results
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("Starting BirdCLEF 2026 Pipeline...")
        
        # Step 1: Match audio to labels
        train_df, soundscape_df = self.match_audio_to_labels()
        
        # Step 2: Load Perch model
        self.load_perch_model()
        
        # Step 3: Prepare datasets
        X_train, X_val, y_train, y_val = self.prepare_datasets(train_df, soundscape_df)
        
        # Step 4: Bootstrap training data
        bootstrap_samples = self.bootstrap_training_data(X_train, y_train, n_bootstrap=5)
        
        # Step 5: Train ensemble models
        models = self.train_ensemble_models(bootstrap_samples)
        
        # Step 6: Evaluate on validation set
        val_predictions = self.predict_ensemble(models, X_val)
        results = self.evaluate_performance(y_val, val_predictions)
        
        print("Pipeline completed successfully!")
        return results, models

if __name__ == "__main__":
    # Run the pipeline
    pipeline = BirdCLEFPipeline("/home/kenll/KaggleSOundCompetition/birdclef-2026")
    results, models = pipeline.run_complete_pipeline()
