import pandas as pd
from pathlib import Path

# Load your training dataset
csv_path = Path(r"C:\Syeda_Nazish\BCA_TY_CP\fake_news_dataset_ac.csv")
train_data = pd.read_csv(csv_path)

# Clean column names
train_data.columns = [
    col.strip().lower()
    for col in train_data.columns
]

print("\n COLUMNS")
print(train_data.columns.tolist())

print("\n HEADLINE")

for i in range(3):
    text = str(train_data["headline"].iloc[i])

    print(f"\nRow {i}")
    print("Word count:", len(text.split()))
    print(text[:500])


print("\n TITLE")

for i in range(3):
    text = str(train_data["title"].iloc[i])

    print(f"\nRow {i}")
    print("Word count:", len(text.split()))
    print(text[:500])


print("\n OTHER COLUMNS")

print("\nSOURCE:")
print(train_data["source"].head(5).to_string(index=False))

print("\nAUTHOR:")
print(train_data["author"].head(5).to_string(index=False))

print("\nCATEGORY:")
print(train_data["category"].head(5).to_string(index=False))

print("\nLABEL:")
print(train_data["label"].head(10).to_string(index=False))