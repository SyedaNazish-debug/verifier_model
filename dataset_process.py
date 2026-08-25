import pandas as pd

file_path = r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\Fake.csv"
fake = pd.read_csv(r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\Fake.csv")
true = pd.read_csv(r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\True.csv")

print("fake news dataset:",fake.shape)
print("True news dataset:",true.shape)

fake_clean = fake.drop_duplicates().copy()
true_clean = true.drop_duplicates().copy()

fake_clean["label"]=0
true_clean["label"]=1

fake_clean = fake_clean[fake_clean["text"].fillna("").str.strip()!=""].copy()
true_clean = true_clean[true_clean["text"].fillna("").str.strip()!=""].copy()

print("clean dataset FAKE:",fake_clean.shape)
print("clean dataset TRUE:",true_clean.shape)

data = pd.concat(
    [fake_clean,true_clean],
    ignore_index=True
)

print("combined data:", data.shape)
print(data["label"].value_counts())

output_path = r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\combined_news.csv"
data.to_csv(output_path, index=False)

print("cleaned and whole dataset saved successfully.")