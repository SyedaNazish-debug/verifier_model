import re
import pandas as pd  
import baseline_modf as bm
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import( 
    accuracy_score,
    classification_report,
    confusion_matrix
)

def refined_clean_text(text):
    text = str(text)

    text = re.sub(r'https?://\s+|www\.\s+', ' ', text)

    text = re.sub(r'\breuters\b','',text,flags=re.IGNORECASE)
    text = re.sub(r'\bgetty images\b','',text,flags=re.IGNORECASE)
    text  =re.sub(r'\bfeatured image\b','', text, flags=re.IGNORECASE)

    text = re.sub(r'\btwitter\.com\b','',text, flags=re.IGNORECASE)
    text = re.sub(r'\bpic\.twitter\b','',text,flags=re.IGNORECASE)

    text = re.sub(r'\[.*?\]\(.*?\)','',text)

    text = re.sub(r'\s+', ' ',text)

    return text.strip()

# Refine training data
refined_train = bm.X_train.apply(refined_clean_text)

# Refine testing data
refined_test = bm.X_test.apply(refined_clean_text)

features_to_remove = [
    "reuters",
    "washington reuters",
    "told reuters",
    "getty",
    "getty images",
    "featured image",
    "image",
    "images",
    "https",
    "twitter com",
    "pic",
    "pic twitter"
]

#########3

X_train_refined = bm.X_train.apply(refined_clean_text)
X_test_refined_tfidf = bm.X_test.apply(refined_clean_text)

print("\n original training  sample:", len(bm.X_train))
print("\n Refined training sample data:",len(X_train_refined))

print("\n original ex:",bm.X_train.iloc[0][:500])
print("\n Refined ex:",X_train_refined.iloc[0][:500])

refined_tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1,2)
)

X_train_refined = refined_tfidf.fit_transform(X_train_refined)
X_test_refined_tfidf = refined_tfidf.transform(X_test_refined_tfidf)


print("\n Defined TF-IDF training data shape:",X_train_refined.shape)
print("\n Refined TF-IDF testing data shape:", X_test_refined_tfidf.shape)

refined_model = LogisticRegression(max_iter=1000,random_state=42)

refined_model.fit(X_train_refined, bm.Y_train)

refined_pred = refined_model.predict(X_test_refined_tfidf)

print("\n Refined model results")
print("\n Refined model accuracy:",
      accuracy_score(bm.Y_test,refined_pred))

print("\n Classification report:",
      classification_report(
          bm.Y_test,
          refined_pred
      )
   )

print("\n Confusion matrix:",
      confusion_matrix(
          bm.Y_test,
          refined_pred
      )
    )


feature_names_refined = refined_tfidf.get_feature_names_out()
coefficients_refined = refined_model.coef_[0]

refined_feature_imp = pd.DataFrame({
    "feature": feature_names_refined,
    "coefficient":coefficients_refined
})


print("\n Top 30 feature in label 1st:",
      refined_feature_imp
      .sort_values(
          "coefficient",
          ascending=False
      )
      .head(30)
      .to_string(index=False)
    )

print("\n Top 30 feature in label 0th:",
      refined_feature_imp
      .sort_values(
          "coefficient",
          ascending=True
      )
      .head(30)
      .to_string(index=False)
    )


###
# Test samples for refined_clean_text()

test_samples = [
    
    "Breaking news: https://example.com/article Government announces new policy.",
    
    "Visit our website at www.example.com for more information.",
    
    "Reuters reported that the economy is growing rapidly.",
    
    "Getty Images shows the aftermath of the event.",
    
    "Featured Image: A large crowd gathered outside the building.",
    
    "Follow updates at twitter.com/news for more details.",
    
    "Check this image pic.twitter.com/ABC123",
    
    "Read the full story [here](https://example.com/news/article).",
    
    "Reuters: Visit https://news.com/article and see Getty Images.",
    
    "   This   text    contains    multiple     spaces.   "
]


for i, sample in enumerate(test_samples, start=1):

    cleaned = refined_clean_text(sample)

    print(f"\n{'='*60}")
    print(f"TEST {i}")

    print("\nOriginal:")
    print(sample)

    print("\nCleaned:")
    print(cleaned)