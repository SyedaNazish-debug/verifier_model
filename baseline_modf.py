import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
file_path = r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\combined_news.csv"
data =pd.read_csv(file_path)

print("combined dataset shape:",data.shape)
print(data.head())
print("\n labelled distribution :",data["label"].value_counts())

print("total duplicate text:",data["text"].duplicated().sum())
print("total unique text:",data["text"].nunique())
print("Total records in the dataset:",len(data))

########

text_label_check = (data.groupby("text")["label"].nunique()
)
conflicting_texts = text_label_check [
          text_label_check > 1
]
print("total different appeaing text with different labels:",
      len(conflicting_texts)
)

#####################

combined_clean = data.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

print("Before removal of similar values:",len(data))
print("After removal of similar values:",len(combined_clean))

X = combined_clean["text"]
Y = combined_clean["label"]

print("\n X sample values:",len(X))
print("\n Y sample values:",len(Y))

groups = data.groupby("text")

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

X = combined_clean["text"]
Y = combined_clean["label"]

unique_text = combined_clean["text"].drop_duplicates()

train_test_split(X, Y, unique_text)

print("training data samples:",len(X_train))
print("testing data samples:",len(X_test))

print("\n Training label distribution:",Y_train.value_counts())
print("\n Testing label distribution:",Y_test.value_counts())

train_texts =set(X_train)
test_texts = set(X_test)

overlap = train_texts.intersection(test_texts)
print("Number of overlapping texts between train and test sets:", len(overlap))

combined_clean = combined_clean.drop_duplicates(
    subset=["text"],
    keep="first"
)
combined_clean = combined_clean.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

print("Before removing duplicate text:", len(combined_clean))
print("After removing duplicate text:", len(combined_clean))

print("\nLabel distribution after removing duplicate text:")
print(combined_clean["label"].value_counts())




### TF-IDF VECTORIZATION MODEL 
tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    ngram_range=(1,2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


print("TF-IDF training shape:",X_train_tfidf.shape)
print("TF-IDF testing shape:",X_test_tfidf.shape)

### train logistic regression model

model=LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, Y_train)


### now to make prediction

Y_pred =model.predict(X_test_tfidf)

### Evaluation of the model

accuracy = accuracy_score(Y_test,Y_pred)

print("\n MODELS ACCURACY :",accuracy)
print("\n classification report :", classification_report(Y_test,Y_pred))
print("\n confusion matrix:", confusion_matrix(Y_test,Y_pred))

### MODEL ACCURACY IS BEING TESTED AND REFINE

title_train = data.loc[X_train.index, "title"]
title_test = data.loc[X_test.index, "title"]

title_tfidf = TfidfVectorizer(
stop_words = "english",
max_features =50000,
ngram_range=(1,2)
)

X_title_train = title_tfidf.fit_transform(title_train)
X_title_test = title_tfidf.transform(title_test)

title_model =  LogisticRegression(
    max_iter=1000,
    random_state=42
)

title_model.fit(X_title_train, Y_train)
title_pred = title_model.predict(X_title_test)

print("\n ..Title only model evaluation..")
print("\n Model's title only accuracy:", accuracy_score(Y_test, title_pred))
print("\n Classification report:", classification_report(Y_test, title_pred))
print("\n confusion matrix of title only model:", confusion_matrix(Y_test, title_pred))



### feature extraction

feature_name = tfidf.get_feature_names_out()
coefficient = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": feature_name,
    "coefficient": coefficient
})

print("\n Top feature for labels 1st:", 
      feature_importance
      .sort_values("coefficient",ascending = False)
      .head(30)
    )

print("\n Top features for label 0:",
      feature_importance
      .sort_values("coefficient", ascending = True)
      .head(30)
    )




###
