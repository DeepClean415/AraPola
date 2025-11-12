import pandas as pd

# Read the two CSV files
arb_df = pd.read_csv('arb.csv')
eng2arbN_df = pd.read_csv('eng2arbN.csv')

# For arb.csv, keep id, text, and polarization columns
arb_merged = arb_df[['id', 'text', 'polarization']]

# For eng2arbN.csv, use translated_text as the text column
eng2arbN_merged = eng2arbN_df[['id', 'translated_text', 'polarization']].copy()
eng2arbN_merged.rename(columns={'translated_text': 'text'}, inplace=True)

# Concatenate the two dataframes
merged_df = pd.concat([arb_merged, eng2arbN_merged], ignore_index=True)

# Save the merged dataset
merged_df.to_csv('arb_merged.csv', index=False)

print(f"Merged dataset created successfully!")
print(f"Total rows: {len(merged_df)}")
print(f"Rows from arb.csv: {len(arb_merged)}")
print(f"Rows from eng2arbN.csv: {len(eng2arbN_merged)}")
print(f"\nFirst few rows of merged dataset:")
print(merged_df.head())
print(f"\nDataset info:")
print(merged_df.info())
