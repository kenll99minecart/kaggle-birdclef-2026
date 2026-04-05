#!/usr/bin/env python3
"""
BirdCLEF 2026 Inference Script
Processes new audio data and generates predictions in submission format
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.signal import butter, lfilter
import warnings
warnings.filterwarnings('ignore')

# Try to import Perch
try:
    import perch_hoplite
    from perch_hoplite.zoo import model_configs
    PERCH_AVAILABLE = True
except ImportError:
    print("Perch not available, will use MFCC fallback")
    PERCH_AVAILABLE = False


# Denoising functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_bird_segments(audio, sr, threshold=0.1, min_duration=0.5, pad=0.2):
    """
    Extract bird sound segments from audio using denoising
    """
    # Bandpass filter (2kHz - 10kHz)
    lowcut = 2000.0
    highcut = 10000.0
    highcut = min(highcut, sr/2 - 100)
    
    y_filtered = bandpass_filter(audio, lowcut, highcut, sr)
    
    # Calculate energy (RMS)
    frame_length = int(sr * 0.1)
    hop_length = int(sr * 0.05)
    rms = librosa.feature.rms(y=y_filtered, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize RMS
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    else:
        return []  # Silent file
    
    # Detect active frames
    active = rms > threshold
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    intervals = []
    start_time = None
    
    for i in range(len(active)):
        if active[i] and start_time is None:
            start_time = times[i]
        elif not active[i] and start_time is not None:
            end_time = times[i]
            if end_time - start_time >= min_duration:
                intervals.append((start_time, end_time))
            start_time = None
    
    if start_time is not None:
        end_time = times[-1]
        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))
            
    if not intervals:
        return []  # No bird sounds detected
        
    # Merge segments within 0.5s
    merged = []
    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start - curr_end < 0.5:
            curr_end = next_end
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    
    # Create segments with padding
    segments = []
    for start, end in merged:
        start_pad = max(0, start - pad)
        end_pad = min(len(audio)/sr, end + pad)
        
        start_idx = int(start_pad * sr)
        end_idx = int(end_pad * sr)
        
        segment = audio[start_idx:end_idx]
        segments.append((segment, start_pad, end_pad))
    
    return segments


class BirdCLEFInference:
    def __init__(self, models_path: str = "training_results", processed_dir: str = "processed_data"):
        self.models_path = Path(models_path)
        self.processed_dir = Path(processed_dir)
        self.perch_model = None
        self.embedding_dim = None
        self.models_dict = None
        self.species_list = None
        self.taxonomy_df = None
        
    def load_models_and_metadata(self):
        """Load trained models and metadata"""
        print("Loading models and metadata...")
        
        # Load models dictionary
        with open(self.models_path / "ensemble_models.pkl", "rb") as f:
            self.models_dict = pickle.load(f)
        
        # Load evaluation results to get species list
        with open(self.models_path / "evaluation_results.json", "r") as f:
            results = json.load(f)
        self.species_list = results["species_list"]
        
        # Load taxonomy for reference
        self.taxonomy_df = pd.read_csv("birdclef-2026/taxonomy.csv")
        
        # Load label mapping
        with open(self.processed_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        self.label_mapping = metadata["label_mapping"]
        
        print(f"Loaded {len(self.models_dict)} species models")
        print(f"Species list: {len(self.species_list)} species")
        
    def load_perch_model(self):
        """Load Perch model for embedding extraction"""
        print("Loading Perch model...")
        
        # Check what embedding dimension was used in training
        try:
            embeddings = np.load(self.models_path / "embeddings.npy")
            self.embedding_dim = embeddings.shape[1]
            print(f"Using embedding dimension from training: {self.embedding_dim}")
        except:
            self.embedding_dim = 1536  # Default Perch dimension
            print(f"Using default embedding dimension: {self.embedding_dim}")
        
        if not PERCH_AVAILABLE:
            print("Perch not available, will use MFCC fallback")
            # Pad MFCC to match expected dimension
            self.use_mfcc_fallback = True
            return
        
        try:
            import kagglehub
            model_path = kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2")
            self.perch_model = model_configs.load_model_by_name(
                "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
            )
            
            # Test with dummy audio to get embedding dimension
            dummy_audio = np.random.randn(16000)  # 1 second of audio
            dummy_embedding = self.perch_model.embed(dummy_audio)
            actual_embedding_dim = dummy_embedding.shape[-1]
            
            print(f"Perch model loaded successfully. Actual embedding dimension: {actual_embedding_dim}")
            self.use_mfcc_fallback = False
            
        except Exception as e:
            print(f"Failed to load Perch model: {e}")
            print("Will use MFCC fallback")
            self.use_mfcc_fallback = True
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int = 32000) -> np.ndarray:
        """Extract MFCC features as fallback"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    
    def _get_perch_embedding(self, audio_path: str) -> np.ndarray:
        """Extract embedding using Perch model with denoising"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=32000)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Apply denoising and extract bird segments
            segments = extract_bird_segments(audio, sr, threshold=0.1, min_duration=0.5, pad=0.2)
            
            if not segments:
                # No bird sounds detected, return zero embedding
                return np.zeros(self.embedding_dim)
            
            # Use the first (most prominent) segment for embedding
            # Alternatively, could average embeddings from all segments
            best_segment, _, _ = segments[0]
            
            # Pad or truncate segment to 5 seconds (160000 samples at 32kHz)
            target_length = 160000
            if len(best_segment) < target_length:
                best_segment = np.pad(best_segment, (0, target_length - len(best_segment)))
            else:
                best_segment = best_segment[:target_length]
            
            # Get embedding
            if hasattr(self, 'use_mfcc_fallback') and self.use_mfcc_fallback:
                # Fallback to MFCC with padding
                mfcc = self.extract_mfcc_features(best_segment, sr)
                # Pad MFCC to match expected dimension
                if mfcc.shape[0] < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - mfcc.shape[0])
                    embedding = np.concatenate([mfcc, padding])
                else:
                    embedding = mfcc[:self.embedding_dim]
                return embedding
            elif self.perch_model is not None:
                # Use Perch model
                embedding = self.perch_model.embed(best_segment)
                # Flatten to 1D
                embedding = embedding.flatten()
                return embedding
            else:
                # MFCC fallback
                mfcc = self.extract_mfcc_features(best_segment, sr)
                # Pad MFCC to match expected dimension
                if mfcc.shape[0] < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - mfcc.shape[0])
                    embedding = np.concatenate([mfcc, padding])
                else:
                    embedding = mfcc[:self.embedding_dim]
                return embedding
                
        except Exception as e:
            print(f"Error extracting embedding from {audio_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.embedding_dim)
    
    def extract_embeddings(self, audio_files: List[str]) -> np.ndarray:
        """Extract embeddings for all audio files"""
        print(f"Extracting embeddings for {len(audio_files)} audio files...")
        
        embeddings = []
        for i, audio_path in enumerate(audio_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(audio_files)} files")
            
            embedding = self._get_perch_embedding(audio_path)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        print(f"Embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def predict_species(self, embeddings: np.ndarray) -> np.ndarray:
        """Make predictions for all species"""
        print("Making species predictions...")
        
        n_samples = embeddings.shape[0]
        n_species = len(self.species_list)
        
        # Initialize predictions
        all_predictions = np.zeros((n_samples, n_species))
        
        # Get species order
        species_list = list(self.species_list)
        
        for species_idx, species in enumerate(species_list):
            models = self.models_dict.get(species, [])
            
            if not models:  # No models for this species
                continue
            
            # Average predictions from all models for this species
            species_predictions = np.zeros(n_samples)
            
            for model in models:
                pred_proba = model.predict_proba(embeddings)[:, 1]  # Probability of positive class
                species_predictions += pred_proba
            
            # Average across all models for this species
            all_predictions[:, species_idx] = species_predictions / len(models)
        
        return all_predictions
    
    def process_test_soundscape(self, test_audio_dir: str, segment_duration: float = 5.0) -> Tuple[List[str], List[str]]:
        """Process test soundscape files using denoising to extract bird segments"""
        print(f"Processing test soundscape files from {test_audio_dir}...")
        
        test_dir = Path(test_audio_dir)
        audio_files = list(test_dir.glob("*.ogg"))
        
        segment_files = []
        segment_ids = []
        
        for audio_file in audio_files:
            print(f"Processing {audio_file.name}...")
            
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=32000)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            # Apply denoising to extract bird segments
            segments = extract_bird_segments(audio, sr, threshold=0.1, min_duration=0.5, pad=0.2)
            
            if not segments:
                print(f"  No bird sounds detected in {audio_file.name}")
                # Create a fallback segment at the beginning
                segment_files.append(str(audio_file))
                segment_ids.append(f"{audio_file.stem}_00")
                continue
            
            # Create segments for each detected bird sound
            for i, (segment, start_time, end_time) in enumerate(segments):
                # Save segment to temporary file
                segment_filename = f"{audio_file.stem}_seg_{i:03d}.wav"
                segment_path = f"temp_segments/{segment_filename}"
                
                # Create temp directory if needed
                os.makedirs("temp_segments", exist_ok=True)
                
                # Save segment
                sf.write(segment_path, segment, sr)
                
                segment_files.append(segment_path)
                segment_ids.append(f"{audio_file.stem}_{int(start_time):02d}")
                
                print(f"  Segment {i}: {start_time:.2f}s - {end_time:.2f}s")
        
        print(f"Created {len(segment_files)} segments from {len(audio_files)} files")
        return segment_files, segment_ids
    
    def create_submission(self, predictions: np.ndarray, segment_ids: List[str], output_path: str = "submission.csv"):
        """Create submission file in required format"""
        print("Creating submission file...")
        
        # Get all species from sample submission to match format
        sample_submission = pd.read_csv("birdclef-2026/sample_submission.csv")
        all_species = sample_submission.columns[1:].tolist()  # Skip row_id column
        
        # Create DataFrame
        submission_data = []
        
        for i, segment_id in enumerate(segment_ids):
            row = {"row_id": segment_id}
            
            # Map predictions to all species in submission format
            for species in all_species:
                if species in self.species_list:
                    species_idx = self.species_list.index(species)
                    row[species] = float(predictions[i, species_idx])
                else:
                    # Species not in our trained models - use small probability
                    row[species] = 0.004273504273504274  # Same as sample submission
            
            submission_data.append(row)
        
        # Create DataFrame and save
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        
        print(f"Submission saved to {output_path}")
        print(f"Shape: {submission_df.shape}")
        print(f"Sample predictions for first row:")
        print(submission_df.iloc[0, 1:10].to_dict())
        
        return submission_df
    
    def run_inference(self, test_audio_dir: str, output_path: str = "submission.csv"):
        """Run complete inference pipeline"""
        print("Starting BirdCLEF inference...")
        
        # Step 1: Load models and metadata
        self.load_models_and_metadata()
        
        # Step 2: Load Perch model
        self.load_perch_model()
        
        # Step 3: Process test soundscape files with denoising
        segment_files, segment_ids = self.process_test_soundscape(test_audio_dir)
        
        # Step 4: Extract embeddings
        embeddings = self.extract_embeddings(segment_files)
        
        # Step 5: Make predictions
        predictions = self.predict_species(embeddings)
        
        # Step 6: Create submission file
        submission_df = self.create_submission(predictions, segment_ids, output_path)
        
        # Step 7: Clean up temporary segments
        import shutil
        if os.path.exists("temp_segments"):
            shutil.rmtree("temp_segments")
            print("Cleaned up temporary segments")
        
        print("Inference completed successfully!")
        return submission_df


def main():
    """Main inference function"""
    # Initialize inference
    inference = BirdCLEFInference()
    
    # Run inference on test data
    # Update this path to your test audio directory
    test_audio_dir = "birdclef-2026/test_soundscapes"
    
    if not os.path.exists(test_audio_dir):
        print(f"Test audio directory not found: {test_audio_dir}")
        print("Please update the test_audio_dir path in the main() function")
        return
    
    # Run inference
    submission_df = inference.run_inference(test_audio_dir, "submission.csv")
    
    print(f"\nSubmission file created: submission.csv")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Species predicted: {len(submission_df.columns) - 1}")


if __name__ == "__main__":
    main()
