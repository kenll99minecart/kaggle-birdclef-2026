#!/usr/bin/env python3
"""
Script 2: Convert audio to embeddings and train models
- Load preprocessed audio data
- Extract embeddings using Perch/MFCC
- Train ensemble models with bootstrapping
- Evaluate performance
"""

import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import pickle
import json

warnings.filterwarnings('ignore')

# For ML models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample

# For Perch embeddings
try:
    import tensorflow as tf
    from perch_hoplite.zoo import model_configs
    PERCH_AVAILABLE = True
except ImportError:
    print("Perch Hoplite not installed. Will use MFCC fallback.")
    PERCH_AVAILABLE = False

class EmbeddingTrainer:
    def __init__(self, processed_data_dir: str = "processed_data"):
        self.processed_dir = Path(processed_data_dir)
        self.output_dir = Path("training_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load preprocessed data
        self.audio_files = self._load_audio_files()
        self.labels = self._load_labels()
        self.metadata = self._load_metadata()
        
        # Initialize Perch model (will be loaded later)
        self.perch_model = None
        self.embedding_dim = None
        
    def _load_audio_files(self) -> List[str]:
        """Load audio file paths"""
        with open(self.processed_dir / "audio_files.pkl", "rb") as f:
            return pickle.load(f)
    
    def _load_labels(self) -> List[Dict]:
        """Load labels"""
        with open(self.processed_dir / "labels.pkl", "rb") as f:
            return pickle.load(f)
    
    def _load_metadata(self) -> Dict:
        """Load metadata"""
        with open(self.processed_dir / "metadata.json", "r") as f:
            return json.load(f)
    
    def load_perch_model(self):
        """Load Google's Perch model for audio embeddings"""
        print("Loading Perch model...")
        
        if not PERCH_AVAILABLE:
            print("Perch Hoplite not available. Will use MFCC fallback.")
            self.perch_model = None
            self.embedding_dim = 13  # MFCC dimension
            return
        
        try:
            # Load Perch model using Hoplite
            self.perch_model = model_configs.load_model_by_name('perch_v2')
            
            # Test with dummy audio to get embedding dimension
            dummy_audio = np.zeros(5 * 32000, dtype=np.float32)
            test_output = self.perch_model.embed(dummy_audio)
            if hasattr(test_output.embeddings, 'numpy'):
                self.embedding_dim = test_output.embeddings.numpy().shape[-1]
            else:
                self.embedding_dim = len(test_output.embeddings)
            
            print(f"Perch model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Could not load Perch model: {e}")
            print("Will use MFCC fallback")
            self.perch_model = None
            self.embedding_dim = 13  # MFCC dimension
    
    def extract_embeddings(self, audio_files: List[str], max_files: Optional[int] = None) -> np.ndarray:
        # default_output_embeddings = Path("training_results/embeddings.npy")
        # print(default_output_embeddings.exists())
        # if default_output_embeddings.exists():
        #     embeddings = np.load(default_output_embeddings)
        #     if max_files:
        #         return embeddings[:max_files]
        #     else:
        #         return embeddings
        """Extract embeddings from audio files"""
        if max_files:
            audio_files = audio_files[:max_files]
            
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
        try:
            # Ensure audio is 32 kHz mono float32
            if sr != 32000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Convert to float32
            audio = audio.astype(np.float32)
            
            # Perch expects 5 seconds of audio, pad or truncate if needed
            target_length = 5 * 32000  # 5 seconds at 32 kHz
            
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            elif len(audio) > target_length:
                # Truncate to 5 seconds
                audio = audio[:target_length]
            
            # Get embeddings using Perch model
            outputs = self.perch_model.embed(audio)
            embedding = outputs.embeddings
            
            # Convert to numpy array if needed
            if hasattr(embedding, 'numpy'):
                embedding = embedding.numpy()
            
            # Flatten to 1D array (remove extra dimensions)
            embedding = embedding.flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Error getting Perch embedding: {e}")
            # Fallback to MFCC
            return self._get_mfcc_embedding(audio, sr)
    
    def _get_mfcc_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Get MFCC embedding as fallback"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    
    def prepare_labels_for_training(self, labels: List[Dict] = None) -> Tuple[np.ndarray, List[int]]:
        """Prepare labels for multi-class training using primary_label"""
        if labels is None:
            labels = self.labels
            
        print("Preparing labels for training...")
        
        # Get all unique primary_label values (species-level)
        all_species = set()
        for label_dict in labels:
            if label_dict['type'] == 'individual':
                all_species.add(label_dict['primary_label'])
            elif label_dict['type'] == 'soundscape_segment':
                for species_info in label_dict['species_info']:
                    all_species.add(species_info['primary_label'])
        
        # Remove -1 (unknown labels) and sort
        all_species = sorted([s for s in all_species if s != -1])
        species_to_idx = {species: idx for idx, species in enumerate(all_species)}
        
        print(f"Found {len(all_species)} unique species (primary_label)")
        
        # Create multi-hot encoded labels
        encoded_labels = []
        
        for label_dict in labels:
            class_vector = np.zeros(len(all_species))
            
            if label_dict['type'] == 'individual':
                # Single species for individual recordings
                species = label_dict['primary_label']
                if species != -1 and species in species_to_idx:
                    species_idx = species_to_idx[species]
                    class_vector[species_idx] = 1
            
            elif label_dict['type'] == 'soundscape_segment':
                # Multiple species for soundscape segments
                for species_info in label_dict['species_info']:
                    species = species_info['primary_label']
                    if species != -1 and species in species_to_idx:
                        species_idx = species_to_idx[species]
                        class_vector[species_idx] = 1
            
            encoded_labels.append(class_vector)
        
        return np.array(encoded_labels), all_species
    
    def split_datasets(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train/validation datasets"""
        print("Splitting datasets...")
        
        # Check if stratification is possible
        from collections import Counter
        label_counts = Counter(tuple(row) for row in labels)
        min_samples = min(label_counts.values())
        
        if min_samples < 2:
            print("Warning: Some classes have too few samples for stratification. Using random split.")
            # Use random split without stratification
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42
            )
        else:
            # Use stratified split
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
    
    def train_ensemble_models(self, bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]], species_list: List[str]) -> Dict[str, List[LogisticRegression]]:
        """Train multiple logistic regression models on bootstrap samples"""
        print("Training ensemble models...")
        
        # Initialize dictionary for models
        models_dict = {species: [] for species in species_list}
        
        for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
            print(f"Training model {i+1}/{len(bootstrap_samples)}")
            
            # Train one model per class
            for class_idx, species in enumerate(species_list):
                # Binary classification for this class
                y_binary = y_boot[:, class_idx]
                
                # Skip if no positive samples
                if np.sum(y_binary) == 0:
                    continue
                
                model = LogisticRegression(
                    random_state=42,
                    max_iter=100,
                    class_weight='balanced'
                )
                model.fit(X_boot, y_binary)
                models_dict[species].append(model)
        
        return models_dict
    
    def predict_ensemble(self, models_dict: Dict[str, List[LogisticRegression]], X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble averaging"""
        print("Making ensemble predictions...")
        
        n_samples = X.shape[0]
        n_species = len(models_dict)
        
        # Initialize predictions
        all_predictions = np.zeros((n_samples, n_species))
        
        # Get species order
        species_list = list(models_dict.keys())
        
        for species_idx, species in enumerate(species_list):
            models = models_dict[species]
            
            if not models:  # No models for this species
                continue
            
            # Average predictions from all models for this species
            species_predictions = np.zeros(n_samples)
            
            for model in models:
                pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
                species_predictions += pred_proba
            
            # Average across all models for this species
            all_predictions[:, species_idx] = species_predictions / len(models)
        
        return all_predictions
    
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
            'class_accuracies': class_accuracies,
            'n_classes': n_classes
        }
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Mean Class Accuracy: {np.mean(class_accuracies):.4f}")
        
        return results
    
    def save_results(self, models_dict: Dict[str, List[LogisticRegression]], results: Dict, embeddings: np.ndarray, species_list: List[str]):
        """Save training results"""
        print("Saving results...")
        
        # Save models as dictionary
        with open(self.output_dir / "ensemble_models.pkl", "wb") as f:
            pickle.dump(models_dict, f)
        
        # Save results with species mapping
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = results.copy()
            results_copy['class_accuracies'] = [float(x) for x in results_copy['class_accuracies']]
            results_copy['species_list'] = species_list
            results_copy['n_species'] = len(species_list)
            # Add model counts per species
            results_copy['models_per_species'] = {species: len(models) for species, models in models_dict.items()}
            json.dump(results_copy, f, indent=2)
        
        # Save embeddings (optional - can be large)
        np.save(self.output_dir / "embeddings.npy", embeddings)
        
        print(f"Results saved to {self.output_dir}")
        print(f"Number of species: {len(species_list)}")
        print(f"Models per species: {[(species, len(models)) for species, models in list(models_dict.items())[:5]]}...")
    
    def run_training(self, max_files: Optional[int] = None):
        """Run complete training pipeline"""
        print("Starting embedding extraction and training...")
        
        # Step 1: Load Perch model
        self.load_perch_model()
        
        # Step 2: Extract embeddings
        embeddings = self.extract_embeddings(self.audio_files, max_files=max_files)
        
        # Step 3: Prepare labels (match the limited audio files)
        if max_files:
            labels = self.labels[:max_files]  # Limit labels to match audio files
        else:
            labels = self.labels
        
        labels_array, species_list = self.prepare_labels_for_training(labels)
        
        # Step 4: Split datasets
        X_train, X_val, y_train, y_val = self.split_datasets(embeddings, labels_array)
        
        # Step 5: Bootstrap training data
        bootstrap_samples = self.bootstrap_training_data(X_train, y_train, n_bootstrap=5)
        
        # Step 6: Train ensemble models
        models = self.train_ensemble_models(bootstrap_samples, species_list)
        
        # Step 7: Evaluate on validation set
        val_predictions = self.predict_ensemble(models, X_val)
        results = self.evaluate_performance(y_val, val_predictions)
        
        # Step 8: Save results
        self.save_results(models, results, embeddings, species_list)
        
        print("Training completed successfully!")
        return results, models

if __name__ == "__main__":
    # Run training with limited files for testing
    trainer = EmbeddingTrainer()
    
    # For testing, limit to first 1000 files
    # Remove this parameter for full dataset
    results, models = trainer.run_training()
    
