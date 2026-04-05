#!/usr/bin/env python3
"""
Test inference script using existing training data
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from inference import BirdCLEFInference


def create_test_data_from_training():
    """Create a small test dataset from existing training soundscape files"""
    print("Creating test data from training soundscape files...")
    
    # Create test directory
    test_dir = Path("test_soundscapes")
    test_dir.mkdir(exist_ok=True)
    
    # Copy a few soundscape files for testing
    soundscape_dir = Path("birdclef-2026/train_soundscapes")
    soundscape_files = list(soundscape_dir.glob("*.ogg"))
    
    # Copy first 5 files for testing
    test_files = soundscape_files[:5]
    
    for file in test_files:
        shutil.copy2(file, test_dir / file.name)
        print(f"Copied {file.name}")
    
    print(f"Created test directory with {len(test_files)} files")
    return str(test_dir)


def run_test_inference():
    """Run inference on test data"""
    print("Running test inference...")
    
    # Create test data
    test_audio_dir = create_test_data_from_training()
    
    # Initialize inference
    inference = BirdCLEFInference()
    
    # Run inference
    submission_df = inference.run_inference(test_audio_dir, "test_submission.csv")
    
    print(f"\nTest submission created: test_submission.csv")
    print(f"Shape: {submission_df.shape}")
    print(f"First few rows:")
    print(submission_df.head())
    
    # Clean up test directory
    import shutil
    shutil.rmtree("test_soundscapes")
    print("Cleaned up test directory")


if __name__ == "__main__":
    run_test_inference()
