"""
Preprocessing script for dev Arabic dataset
Applies basic preprocessing and saves as arb_clean.csv
"""

import pandas as pd
import sys
sys.path.append('..')
from ArbPreBasic import ArabicBasicPreprocessor

def main():
    # Initialize preprocessor
    print("Initializing Arabic Basic Preprocessor...")
    preprocessor = ArabicBasicPreprocessor()
    
    # Load dev data
    print("\nLoading dev Arabic dataset...")
    dev_df = pd.read_csv('../dev/arb.csv')
    print(f"Loaded {len(dev_df)} samples")
    print(f"Columns: {dev_df.columns.tolist()}")
    
    # Display sample before preprocessing
    print("\n" + "="*80)
    print("Sample BEFORE preprocessing:")
    print("="*80)
    for i in range(min(3, len(dev_df))):
        print(f"\nSample {i+1}:")
        print(f"ID: {dev_df['id'].iloc[i]}")
        print(f"Text: {dev_df['text'].iloc[i][:150]}...")
    
    # Apply preprocessing
    print("\n" + "="*80)
    print("Applying basic preprocessing...")
    print("="*80)
    dev_df['text_clean'] = dev_df['text'].apply(preprocessor.preprocess)
    
    # Display sample after preprocessing
    print("\nSample AFTER preprocessing:")
    print("="*80)
    for i in range(min(3, len(dev_df))):
        print(f"\nSample {i+1}:")
        print(f"Original: {dev_df['text'].iloc[i][:100]}...")
        print(f"Cleaned:  {dev_df['text_clean'].iloc[i][:100]}...")
    
    # Save cleaned data
    output_file = 'arb_clean.csv'
    print(f"\n" + "="*80)
    print(f"Saving cleaned data to {output_file}...")
    print("="*80)
    
    # Save with id and text_clean columns
    dev_df[['id', 'text_clean']].to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully saved {len(dev_df)} cleaned samples to {output_file}")
    
    # Verification
    print("\n" + "="*80)
    print("Verification:")
    print("="*80)
    verify_df = pd.read_csv(output_file)
    print(f"File: {output_file}")
    print(f"Shape: {verify_df.shape}")
    print(f"Columns: {verify_df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(verify_df.head())
    
    # Check for any null values
    null_counts = verify_df.isnull().sum()
    if null_counts.sum() == 0:
        print("\n✓ No missing values detected")
    else:
        print(f"\n⚠ Warning: Missing values detected:\n{null_counts}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Input:  ../dev/arb.csv ({len(dev_df)} samples)")
    print(f"Output: {output_file} ({len(verify_df)} samples)")
    print("\nReady for model training and prediction!")

if __name__ == "__main__":
    main()
