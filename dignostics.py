import pandas as pd

fake=pd.read_csv(r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\Fake.csv")
true=pd.read_csv(r"C:\Syeda_Nazish\BCA_TY_CP\News _dataset\True.csv")

print("Fake news dataset shape:", fake.shape)
print(fake.columns.tolist())

print("\nTrue news dataset columns:")
print("True news dataset shape:", true.shape)
print(true.columns.tolist())

print(fake.head())
print(true.head())

print("\n checking the possible null values in the data:",fake.isnull().sum())
print("\n checking the possible null values in the data:",true.isnull().sum())

print("\n checking the duplicates values possibilities in fake data:",fake.duplicated().sum())

print("\n Actual fake duplicate values in the fake data:")
print(fake[fake.duplicated()].sort_values(by="text").head(20))
print(fake["text"].duplicated().sum())

print("\n checking the duplicates values possibilities in true data:",true.duplicated().sum())

print("\n Actual true duplicate values in the true data:")
print(true[true.duplicated()].sort_values(by="text").head(20))
print(true["text"].duplicated().sum())

true_text_duplicated = true[true["text"].duplicated(keep=False)]
print(true_text_duplicated.sort_values(by="text").head(20))

print("\n checking the length of the text in fake data:",fake["text"].str.len().describe())
print("\n fake article with the mini length <=50:",(fake["text"].str.len() <=50).sum())

print(fake.loc[fake["text"].str.len() <=50, ["title", "text"]].head(20))

print("\n Most frequent short fake articles:")
print(fake.loc[fake["text"].str.len() <=50, "text"].value_counts().head(20))

print("\n checking the length of the text in true data:",true["text"].str.len().describe())
print("\n true article with the mini length <=50:",(true["text"].str.len() <=50).sum())

print(true.loc[true["text"].str.len() <=50, ["title", "text"]].head(20))

Empty_fake = fake[fake["text"].str.strip()==""]
print("Empty fake text articles:", len(Empty_fake))
print("\n empty fake text in articles subject wise:",Empty_fake["subject"].value_counts())

Empty_true = true[true["text"].str.strip()==""]
print("Empty true text articles:", len(Empty_true))
print("\n empty true text in articles subject wise:",Empty_true["subject"].value_counts())


#now let's remove the duplicates in the both datasets and
#  it's columns and also remove the empty rows in the both


print("\n cleaning the dataset for further processing before training and the development of "
"\n any classification , detection or predection model for  verifier lens model:")

fake_clean = fake.drop_duplicates().copy()
true_clean = true.drop_duplicates().copy()

print("fake before:",len(fake))
print("fake after count of exact duplicates:",len(fake_clean))
print("fake removed:",len(fake)-len(fake_clean))

fake_non_empty =fake[fake["text"].str.strip()!=""].copy()
print("fake Non empty rows:",len(fake_non_empty))
print("fake  unique Non empty rows:",fake_non_empty["text"].nunique())


print("true before:",len(true))
print("true after count of exact duplicates:",len(true_clean))
print("true removed:",len(true)-len(true_clean))

true_non_empty =true[true["text"].str.strip()!=""].copy()
print("true Non empty rows:",len(true_non_empty))
print("true  unique Non empty rows:",true_non_empty["text"].nunique())

short_fake = fake_clean[fake_clean["text"].str.len() <=50]
short_true = true_clean[true_clean["text"].str.len() <=50]

print("\n shortfake news articles :",len(short_fake))
print("\n shorttrue news articles :",len(short_true))

print("\n short fake articles title length:",short_fake["title"]
      .str.len().describe())
print("\n short true articles title length:",short_true["title"]
      .str.len().describe())


short_fake = short_fake.copy()
short_fake["title_len"] = short_fake["title"].str.len()
short_true = short_true.copy()
short_true["title_len"] = short_true["title"].str.len()

print(short_fake.sort_values(
    by="text",
    key=lambda x: x.str.len()
    )
    .head(20)
)

print("short fake news articles :",len(short_fake))
print(short_fake[["title","text"]])


print(short_true.sort_values(
    by="text"
    ,key=lambda x: x.str.len()
    )
    .head(20)
)

print("short true news articles :",len(short_true))
print(short_true[["title","text"]])

# finally finding the whole  duplicates sum in the both datasets being used for the devoplement of project verifier lense model

print("\n  Duplicate articles text in total:")
print("fake:",fake_clean["text"].duplicated().sum())
print("true:",true_clean["text"].duplicated().sum())

#crosschecking the overlapping articles in the data

fake_text = set(fake_clean["text"])
true_text = set(true_clean["text"])

common_text =fake_text.intersection(true_text)
print("\n similar appreance in the both datasets:",len(common_text))

for text in common_text:
    print("\n --- COMMON TEXT ---")
    print(text)

    print("\n--- FAKE RECORD ---")
    print(fake_clean[fake_clean["text"] == text][
        ["title", "text", "subject", "date"]
    ])

    print("\n--- TRUE RECORD ----------")
    print(true_clean[true_clean["text"] == text][
        ["title", "text", "subject", "date"]
    ])