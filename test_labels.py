import pandas as pd
import numpy as np
from pathlib import Path

def test_label_parsing():
    """Test the label parsing logic"""
    
    # Load the data
    data_dir = Path("/home/kenll/KaggleSOundCompetition/birdclef-2026")
    train_df = pd.read_csv(data_dir / "train.csv")
    soundscape_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy_df = pd.read_csv(data_dir / "taxonomy.csv")
    
    print("=== TRAIN.CSV SAMPLE ===")
    print(train_df[['primary_label', 'class_name', 'scientific_name']].head(10))
    print(f"\nUnique primary_label types in train.csv: {train_df['primary_label'].apply(type).value_counts()}")
    
    print("\n=== SOUNDSCAPE LABELS SAMPLE ===")
    print(soundscape_labels_df.head(10))
    print(f"\nUnique primary_label types in soundscapes: {soundscape_labels_df['primary_label'].apply(type).value_counts()}")
    
    print("\n=== TAXONOMY SAMPLE ===")
    print(taxonomy_df.head(10))
    print(f"\nUnique primary_label types in taxonomy: {taxonomy_df['primary_label'].apply(type).value_counts()}")
    
    # Test parsing a soundscape label
    sample_label = soundscape_labels_df.iloc[0]['primary_label']
    print(f"\n=== TESTING SOUNDSCAPE LABEL PARSING ===")
    print(f"Sample soundscape label: {sample_label}")
    parsed_labels = [str(label.strip()) for label in str(sample_label).split(';') if label.strip()]
    print(f"Parsed labels: {parsed_labels}")
    
    # Test lookup in taxonomy
    for label in parsed_labels:
        tax_info = taxonomy_df[taxonomy_df['primary_label'].astype(str) == str(label)]
        if not tax_info.empty:
            print(f"Label {label}: {tax_info.iloc[0]['class_name']} - {tax_info.iloc[0]['common_name']}")
        else:
            print(f"Label {label}: NOT FOUND in taxonomy")
    
    # Test train.csv labels
    print(f"\n=== TESTING TRAIN.CSV LABEL PARSING ===")
    sample_train_labels = train_df['primary_label'].head(5).tolist()
    for label in sample_train_labels:
        tax_info = taxonomy_df[taxonomy_df['primary_label'].astype(str) == str(label)]
        if not tax_info.empty:
            print(f"Label {label}: {tax_info.iloc[0]['class_name']} - {tax_info.iloc[0]['common_name']}")
        else:
            print(f"Label {label}: NOT FOUND in taxonomy")

if __name__ == "__main__":
    test_label_parsing()
