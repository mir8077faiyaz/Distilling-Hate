from datasets import load_dataset

# Load the dataset
ds = load_dataset("aplycaebous/BanTH")

# Basic info
print(ds)

# Show splits
print("\nSplits:")
for split in ds:
    print(split, len(ds[split]))

# Show column names for each split
print("\nColumns:")
for split in ds:
    print(split, ds[split].column_names)

# Show one example from each split
print("\nSample examples:")
for split in ds:
    print(f"\n--- {split.upper()} EXAMPLE ---")
    print(ds[split][0])
