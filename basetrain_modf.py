import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

file_path = r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\combined_news.csv"
data =pd.read_csv(file_path)

x = data["text"]
y = data["label"]

groups = data.groupby("text")

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx , test_idx =next (gss.split(x,y,groups=groups))

x_train = x.iloc[train_idx]
x_test = x.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


X = combined["text"]
y = combined["label"]

unique_text = combined["text"].drop_duplicates()

train_test_split(X, y, unique_text)

print("training data samples:",len(x_train))
print("testing data samples:",len(x_test))

print("\n Training label distribution:",y_train.value_counts())
print("\n Testing label distribution:",y_test.value_counts())

train_texts =set(x_train)
test_texts = set(x_test)

overlap = train_texts.intersection(test_texts)
print("Number of overlapping texts between train and test sets:", len(overlap))

combined_clean = combined.drop_duplicates(
    subset=["text"],
    keep="first"
)
combined_clean = combined.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

print("Before removing duplicate text:", len(combined))
print("After removing duplicate text:", len(combined_clean))

print("\nLabel distribution after removing duplicate text:")
print(combined_clean["label"].value_counts())