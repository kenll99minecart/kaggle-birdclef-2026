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

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# For ML models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import resample

# For PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For Perch embeddings
try:
    import tensorflow as tf
    from perch_hoplite.zoo import model_configs
    PERCH_AVAILABLE = True
except ImportError:
    print("Perch Hoplite not installed. Will use MFCC fallback.")
    PERCH_AVAILABLE = False

class MLPDataset(Dataset):
    """Custom Dataset for MLP training"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPModel(nn.Module):
    """2-layer Multi-layer Perceptron for multi-label classification"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
    
    def train_mlp_model(self, X_train: np.ndarray, y_train: np.ndarray, input_dim: int, output_dim: int, X_val: np.ndarray = None, y_val: np.ndarray = None, hidden_dim: int = 256, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 4) -> MLPModel:
        """Train a single MLP model with early stopping"""
        print(f"Training MLP with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        
        # Create dataset and dataloader
        dataset = MLPDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataset if provided
        val_dataset = None
        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_dataset = MLPDataset(X_val, y_val)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MLPModel(input_dim, hidden_dim, output_dim).to(device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()  # Cross entropy for multi-label classification
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_embeddings, batch_labels in dataloader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate validation loss if validation data is provided
            val_loss = None
            if val_dataloader is not None:
                model.eval()
                val_total_loss = 0.0
                with torch.no_grad():
                    for val_batch_embeddings, val_batch_labels in val_dataloader:
                        val_batch_embeddings = val_batch_embeddings.to(device)
                        val_batch_labels = val_batch_labels.to(device)
                        
                        val_outputs = model(val_batch_embeddings)
                        val_loss_batch = criterion(val_outputs, val_batch_labels)
                        val_total_loss += val_loss_batch.item()
                
                val_loss = val_total_loss / len(val_dataloader)
                model.train()  # Return to training mode
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_without_improvement += 1
                
                # Print training and validation loss
                if (epoch + 1) % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (patience={patience})")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                # No validation data, just print training loss
                if (epoch + 1) % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Load best model if validation was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def train_ensemble_models(self, bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]], X_val: np.ndarray, y_val: np.ndarray, input_dim: int, output_dim: int, hidden_dim: int = 256, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 4) -> List[MLPModel]:
        """Train multiple MLP models on bootstrap samples with early stopping"""
        print("Training ensemble MLP models...")
        
        models = []
        
        for i, (X_boot, y_boot) in enumerate(bootstrap_samples):
            print(f"Training model {i+1}/{len(bootstrap_samples)}")
            
            model = self.train_mlp_model(
                X_boot, y_boot, input_dim, output_dim, X_val, y_val, hidden_dim, epochs, batch_size, learning_rate, patience
            )
            models.append(model)
        
        return models
    
    def predict_ensemble(self, models: List[MLPModel], X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble averaging"""
        print("Making ensemble predictions...")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Initialize predictions
        all_predictions = []
        
        # Get predictions from each model
        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_predictions.append(probs.cpu().numpy())
        
        # Average across all models
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
    
    def save_results(self, models: List[MLPModel], results: Dict, embeddings: np.ndarray, species_list: List[str]):
        """Save training results"""
        print("Saving results...")
        
        # Save models
        model_dir = self.output_dir / "mlp_models"
        model_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(models):
            torch.save(model.state_dict(), model_dir / f"mlp_model_{i}.pth")
        
        # Save model architecture info
        model_info = {
            'input_dim': models[0].fc1.in_features,
            'hidden_dim': models[0].fc1.out_features,
            'output_dim': models[0].fc2.out_features,
            'n_models': len(models)
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Save results with species mapping
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = results.copy()
            results_copy['class_accuracies'] = [float(x) for x in results_copy['class_accuracies']]
            results_copy['species_list'] = species_list
            results_copy['n_species'] = len(species_list)
            results_copy['model_info'] = model_info
            json.dump(results_copy, f, indent=2)
        
        # Save embeddings (optional - can be large)
        np.save(self.output_dir / "embeddings.npy", embeddings)
        
        print(f"Results saved to {self.output_dir}")
        print(f"Number of species: {len(species_list)}")
        print(f"Number of models: {len(models)}")
    
    def run_training(self, max_files: Optional[int] = None, hidden_dim: int = 256, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 4):
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
        
        # Step 6: Train ensemble MLP models
        input_dim = embeddings.shape[1]
        output_dim = len(species_list)
        models = self.train_ensemble_models(
            bootstrap_samples, X_val, y_val, input_dim, output_dim, hidden_dim, epochs, batch_size, learning_rate, patience
        )
        
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
    
