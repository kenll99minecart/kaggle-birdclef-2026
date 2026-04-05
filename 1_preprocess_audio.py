#!/usr/bin/env python3
"""
Script 1: Preprocess all audio data
- Match audio files to labels
- Extract audio segments from soundscapes
- Save processed data and labels for later use
"""

import pandas as pd
import numpy as np
import os
import glob
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import pickle
import json

warnings.filterwarnings('ignore')

# Import denoising functions
import sys
sys.path.append('/home/kenll/KaggleSOundCompetition')
from denoising import process_file

class AudioPreprocessor:
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
        
        # Create label mapping
        self.label_mapping = self._create_label_mapping()
        
    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from string codes to inat_taxon_id integers"""
        mapping = {}
        for _, row in self.taxonomy_df.iterrows():
            mapping[str(row['primary_label'])] = int(row['inat_taxon_id'])
        print(f"Created label mapping for {len(mapping)} species")
        return mapping
        
    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '00:00:05' to seconds"""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
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
    
    def prepare_final_dataset(self, train_df: pd.DataFrame, soundscape_df: pd.DataFrame):
        """Prepare final dataset with all audio files and labels"""
        print("Preparing final dataset...")
        
        # Process individual training audio
        train_audio_files = train_df['audio_path'].tolist()
        train_labels = []
        
        for _, row in train_df.iterrows():
            train_labels.append({
                'primary_label': row['primary_label'],
                'primary_label_int': row['primary_label_int'],
                'class_name': row['class_name'],
                'type': 'individual'
            })
        
        # Process soundscape segments - use the matched data directly
        print("Processing soundscape segments...")
        soundscape_audio_files = []
        soundscape_labels = []
        
        for _, row in soundscape_df.iterrows():
            audio_path = row['audio_path']
            soundscape_audio_files.append(audio_path)
            soundscape_labels.append({
                'primary_labels': row['primary_labels'],
                'primary_labels_int': row['primary_labels_int'],
                'species_info': row['species_info'],
                'type': 'soundscape_segment'
            })
        
        print(f"Individual files: {len(train_audio_files)}")
        print(f"Soundscape segments: {len(soundscape_audio_files)}")
        
        # Combine all data
        all_audio_files = train_audio_files + soundscape_audio_files
        all_labels = train_labels + soundscape_labels
        
        print(f"Total audio files: {len(all_audio_files)}")
        print(f"Individual files: {len(train_audio_files)}, Soundscape segments: {len(soundscape_audio_files)}")
        
        return all_audio_files, all_labels
    
    def save_processed_data(self, audio_files: List[str], labels: List[Dict]):
        """Save processed data for later use"""
        print("Saving processed data...")
        
        # Save audio file paths
        with open(self.output_dir / "audio_files.pkl", "wb") as f:
            pickle.dump(audio_files, f)
        
        # Save labels
        with open(self.output_dir / "labels.pkl", "wb") as f:
            pickle.dump(labels, f)
        
        # Save metadata
        metadata = {
            'total_files': len(audio_files),
            'individual_files': sum(1 for label in labels if label['type'] == 'individual'),
            'segment_files': sum(1 for label in labels if label['type'] == 'soundscape_segment'),
            'label_mapping': self.label_mapping,
            'classes': sorted(self.taxonomy_df['class_name'].unique())
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(audio_files)} audio files and labels to {self.output_dir}")
        print(f"Metadata: {metadata}")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("Starting audio preprocessing...")
        
        # Step 1: Match audio to labels
        train_df, soundscape_df = self.match_audio_to_labels()
        
        # Step 2: Prepare final dataset
        audio_files, labels = self.prepare_final_dataset(train_df, soundscape_df)
        
        # Step 3: Save processed data
        self.save_processed_data(audio_files, labels)
        
        print("Preprocessing completed successfully!")
        return audio_files, labels

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = AudioPreprocessor("/home/kenll/KaggleSOundCompetition/birdclef-2026")
    audio_files, labels = preprocessor.run_preprocessing()
