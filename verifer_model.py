# Fake News Detection using Machine Learning
from os import path
from pathlib import Path
from unittest import result
from matplotlib.pylab import average
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from streamlit import metric


# Load dataset
# Replace 'news.csv' with your dataset filename
csv_path = Path(r"C:\Syeda_Nazish\BCA_TY_CP\fake_news_dataset_ac.csv") # Update this path to your dataset
train_data = pd.read_csv(csv_path)

csv_path = Path(r"C:\Syeda_Nazish\BCA_TY_CP\test_t.csv") # Update this path to your dataset
test_data =pd.read_csv(csv_path)

train_data.columns  = [col.strip().lower() for col in train_data.columns]

X_train, X_test, y_train, y_test = train_test_split(
    train_data["title"],
    train_data["label"],
    test_size=0.2,
    random_state=42,
    stratify=train_data["label"]
)
train_data = train_data.dropna(subset=["title", "label"])


label_mapping = {
    "true": "real",
    "mostly-true": "real",
    "false": "fake",
    "pants-fire": "fake"
}
test_data = test_data[
    test_data["label"].isin(label_mapping.keys())
].copy()

test_data.columns  = [col.strip().lower() for col in test_data.columns]

test_data["label"] = test_data["label"].map(label_mapping)
test_data = test_data.dropna(subset=["title", "label"])


print("\n========== FAKE NEWS EXAMPLES ==========")
print(test_data[test_data["label"] == "fake"][["title"]].head(3).to_string())

print("\n========== REAL NEWS EXAMPLES ==========")
print(test_data[test_data["label"] == "real"][["title"]].head(3).to_string()) 

# =========================
# PRIMARY TRAIN / TEST SPLIT
# =========================

X = train_data["title"]
y = train_data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== PRIMARY DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Primary test records:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nPrimary test labels:")
print(y_test.value_counts())

# Convert text into numerical features

train_data["label"] = (
    train_data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

test_data["label"] = (
    test_data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

train_predictions = model.predict(X_train_tfidf)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(
    f"Training Accuracy: {train_accuracy * 100:.2f}%"
)

# Predict on test data  Evaluate model

predictions = model.predict(X_test_tfidf)

print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts())

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy:{accuracy*100:.2f}%")

print("\nClassification  report :")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Function to predict custom news
def detect_news(fake_news):
     vector = tfidf.transform([fake_news])

     probabilities = model.predict_proba(vector)[0]
     prediction = model.predict(vector)[0]

     confidence = max(probabilities) * 100

     return prediction , confidence

# Example

print(test_data["label"].value_counts())
print(test_data["label"].unique())

print("Total rows:", len(test_data))
print("Empty titles:", test_data["title"].isna().sum())
print("Duplicate rows:", test_data.duplicated().sum())


print("\nDataset Information:")
print("Total records:", len(test_data))

print("\nLabel counts:")
print(test_data["label"].value_counts())

print("\nUnique labels:")
print(test_data["label"].unique())

print("\nMissing values:")
print(test_data[["title", "label"]].isnull().sum())

print("\nDuplicate records:", test_data.duplicated().sum())


##########################################

print("\n========== TRAINING DATASET ==========")

print("Total rows:", len(train_data))

print("\nColumns:")
print(train_data.columns.tolist())

print("\nLabel distribution:")
print(train_data["label"].value_counts())

print("\nUnique labels:")
print(train_data["label"].unique())

print("\nSample records:")
print(
    train_data[["title", "label"]]
    .head(10)
    .to_string()
)

print("\nMissing values:")
print(
    train_data[["title", "label"]]
    .isnull()
    .sum()
)

print("\nDuplicate records:")
print(train_data.duplicated().sum())





###########################

print("\n========== ARTICLE TEXT LENGTH ==========")

train_lengths = train_data["title"].astype(str).str.split().str.len()
test_lengths = test_data["title"].astype(str).str.split().str.len()

print("Training:")
print(train_lengths.describe())

print("\nTesting:")
print(test_lengths.describe())