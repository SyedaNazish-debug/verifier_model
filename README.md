[README.md](https://github.com/user-attachments/files/34516293/README.md)
# Verifier Lense

**A machine-learning-based news text classification application built with Python, Flask, TF-IDF, and Logistic Regression.**

Verifier Lense accepts news text from a user, converts it into numerical features using a previously trained TF-IDF vectorizer, and uses a saved Logistic Regression model to predict whether the text resembles the `FAKE NEWS` or `REAL NEWS` class learned from its training dataset.

> **Important:** Verifier Lense is a text-classification system, not an independent fact-checking service. Its prediction and confidence score do not prove whether a claim is factually true or false. Always verify important information using credible, independent sources.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [How the System Works](#how-the-system-works)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Evaluation](#model-evaluation)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Development Notes and Known Follow-up Work](#development-notes-and-known-follow-up-work)
- [Future Improvements](#future-improvements)
- [Project Team](#project-team)
- [Repository](#repository)

---

## Project Overview

The rapid circulation of information through websites and social platforms makes it difficult for readers to assess the reliability of every news item they encounter. Verifier Lense explores how Natural Language Processing (NLP) and supervised machine learning can support the initial classification of news text.

The project follows an end-to-end machine learning workflow:

1. Prepare and inspect a labelled news dataset.
2. Combine the news title and article text for model input.
3. Clean the text and remove duplicate records.
4. Convert text into numerical features using TF-IDF.
5. Train and evaluate a Logistic Regression classifier.
6. Save the trained classifier and vectorizer.
7. Integrate the saved components into a Flask web application.
8. Accept new text and display a predicted class with a probability-based confidence score.

The model is trained separately from the web application. During normal prediction, Flask loads the saved model and vectorizer rather than retraining them for every request.

## Objectives

- Develop a practical application of NLP and supervised machine learning.
- Prepare labelled news text for binary classification.
- Apply TF-IDF feature extraction to represent textual data numerically.
- Train a Logistic Regression classifier to distinguish the two dataset labels.
- Evaluate the model using standard classification metrics.
- Integrate the saved machine learning components into a Flask application.
- Provide a straightforward interface for submitting news text and viewing a prediction.

## How the System Works

The application follows this prediction flow:

```text
User enters news text
        |
        v
Flask receives the form submission
        |
        v
Saved TF-IDF vectorizer transforms the text
        |
        v
Saved Logistic Regression model predicts a class
        |
        v
Model probability is converted to a percentage
        |
        v
Flask sends the result to the HTML template
        |
        v
User sees the predicted label and confidence score
```

### Main components

| Component | Responsibility |
|---|---|
| HTML template | Collects the news text and displays the prediction |
| CSS | Styles the web interface |
| Flask | Handles page requests and prediction submissions |
| TF-IDF vectorizer | Converts text into the feature representation expected by the model |
| Logistic Regression | Predicts the class from the transformed text |
| Joblib model files | Store the trained model and vectorizer for reuse |

## Machine Learning Workflow

The model-development notebook is available in [`VERIFIER_L_EX.ipynb`](VERIFIER_L_EX.ipynb).

### 1. Dataset preparation

The notebook loads a CSV dataset from Google Drive in Google Colab. The observed initial dataset contains **44,058 records** with the following columns:

- `title`
- `text`
- `subject`
- `date`
- `label`

The title and article body are combined to create the text input used for classification. The `label` column supplies the target class.

### 2. Duplicate handling and text cleaning

The notebook combines the title and article body, removes duplicate combined text, and applies a `clean_text()` function.

The documented cleaning operations include:

- Removing selected URL patterns.
- Removing selected source and image-reference phrases, including Reuters, Getty Images, Featured Image, and selected Twitter references.
- Removing Markdown-style links.
- Normalizing repeated whitespace.

The notebook reports that the initial 44,058 records were reduced to 38,658 after the first duplicate-removal step. It then checks cleaned text for empty records, duplicate text, and identical text associated with conflicting labels. The notebook output reports zero empty cleaned texts and zero conflicting-label groups in that check.

### 3. Train-test split

The notebook/report documents a stratified 80:20 split with `random_state=42`.

- Training records: 30,922
- Testing records: 7,731

The split is intended to preserve the class distribution between training and testing data.

### 4. TF-IDF feature extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features. The report describes the vectorizer configuration as:

- English stop-word removal
- Unigram and bigram features
- Maximum feature limit of 50,000

The vectorizer is fitted on the training text. The same fitted vectorizer is then used to transform test text and new application input. A new vectorizer must not be fitted independently for each prediction because the model expects the feature space learned during training.

### 5. Logistic Regression

The transformed TF-IDF features are used to train a supervised Logistic Regression classifier. The saved classifier is reused by the Flask application.

The label mapping documented in the project report and current `app.py` is:

| Model label | Application result |
|---|---|
| `0` | `FAKE NEWS` |
| `1` | `REAL NEWS` |

### 6. Saving and integration

The trained classifier and TF-IDF vectorizer are serialized using Joblib and loaded by `app.py`.

The current Flask application expects these files:

- `model/VERIFIER LENSE MODEL.pkl`
- `model/TF-IDF_VECTORIZIER.pkl`

Keep the filenames and paths consistent with the loading statements in `app.py`.

## Model Evaluation

The project report documents the final model's evaluation on a separate test set of **7,731 samples**.

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.98 | 0.98 |
| Recall | 0.97 | 0.99 |
| F1-score | 0.98 | 0.98 |
| Support | 3,492 | 4,239 |

**Reported test accuracy: approximately 97.93%.**

The report's confusion-matrix counts are:

| Actual class | Predicted class 0 | Predicted class 1 |
|---|---:|---:|
| Class 0 | 3,385 | 107 |
| Class 1 | 53 | 4,186 |

These are results reported for the project's held-out dataset, not a guarantee of performance on current events, unseen sources, other languages, or arbitrary real-world claims. Evaluation quality also depends on dataset composition, possible source/style patterns, preprocessing consistency, and the independence of training and testing examples.

## Technology Stack

| Technology | Use |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| HTML | Web page structure |
| CSS | Interface styling |
| scikit-learn | TF-IDF vectorization and Logistic Regression |
| Joblib | Saving and loading trained ML artifacts |
| Pandas | Dataset loading and preparation in the notebook |
| Google Colab | Model development environment used in the notebook |
| Git and GitHub | Version control and source-code hosting |

The current `requirements.txt` lists Flask, Joblib, and scikit-learn. Pandas is used in the notebook and should be installed separately if you plan to rerun the notebook locally.

## Project Structure

The following is the logical structure based on the files referenced by the current application and repository materials. The exact presence of every optional asset or model artifact should be checked in the repository.

```text
verifier_model/
│
├── app.py
├── requirements.txt
├── README.md
├── VERIFIER_L_EX.ipynb
│
├── model/
│   ├── VERIFIER LENSE MODEL.pkl
│   └── TF-IDF_VECTORIZIER.pkl
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    ├── VLM-LOGO.png       # if included in the repository
    └── team-logo.png      # if included in the repository
```

- `app.py`: Flask routes, model loading, text transformation, prediction, and result rendering.
- `requirements.txt`: Python dependencies for the web application.
- `VERIFIER_L_EX.ipynb`: Dataset preparation, cleaning, model-development, and evaluation workflow.
- `model/`: Serialized model and vectorizer required for inference.
- `templates/index.html`: User interface and prediction-result template.
- `static/`: CSS and visual assets used by the interface.

## Features

- Text-based news classification through a web form.
- Flask backend integration with saved machine learning artifacts.
- TF-IDF transformation using the trained vectorizer.
- Logistic Regression prediction.
- Display of the predicted `FAKE NEWS` or `REAL NEWS` label.
- Display of a probability-based confidence percentage.
- Informational sections describing the application and its workflow.

The current application does **not** independently verify URLs, retrieve live news, compare claims against fact-checking databases, or maintain prediction history.

## Getting Started

### Prerequisites

- Python 3.9 or later is recommended.
- Git.
- The trained model and vectorizer files in the expected `model/` directory.

Use a Python version compatible with the scikit-learn version used to create the serialized model artifacts. Loading Joblib/pickle-based models across incompatible library versions can fail or produce unexpected behaviour.

### 1. Clone the repository

```bash
git clone https://github.com/SyedaNazish-debug/verifier_model.git
cd verifier_model
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you intend to run the notebook locally, install its additional data-processing dependency:

```bash
pip install pandas jupyter
```

### 4. Check the model files

Confirm that both files exist at the paths expected by `app.py`:

```text
model/VERIFIER LENSE MODEL.pkl
model/TF-IDF_VECTORIZIER.pkl
```

The application cannot perform predictions if either artifact is missing. Do not rename the files unless you also update the paths in `app.py`.

## Running the Application

Start the Flask development server:

```bash
python app.py
```

Open the local address shown in the terminal. With the current Flask defaults, this is usually:

```text
http://127.0.0.1:5000/
```

The current `app.py` enables Flask debug mode for local development. Debug mode should be disabled before deploying the application to a public production environment.

## Using the Application

1. Open the application in your browser.
2. Navigate to the news analysis section.
3. Paste the news text into the input field.
4. Submit the form using the **Analyze** button.
5. Review the predicted class and model confidence percentage.
6. Treat the output as an automated classification result, not a factual verdict.

## Limitations and Responsible Use

- **Text classification is not fact-checking.** The application does not independently verify claims against authoritative sources.
- **Dataset dependence:** The model learns patterns from its labelled training data. Its behaviour may change on different topics, sources, writing styles, languages, or time periods.
- **Confidence is not truth probability:** The displayed percentage is a model probability estimate for its predicted class. It is not proof that the content is true or false.
- **No live-source retrieval:** The current prediction route processes submitted text and does not fetch or validate the original article URL.
- **No user history:** The current application does not provide persistent prediction-history storage.
- **Input quality matters:** Very short, incomplete, unrelated, or poorly formatted text may not provide useful evidence for classification.
- **Model artifact compatibility:** Serialized models should be loaded only from trusted sources and with compatible library versions.

For important decisions, compare the original claim with multiple credible sources and established fact-checking organisations.

## Development Notes and Known Follow-up Work

The repository and project report describe the model-development workflow, while the current Flask source shows a relatively small inference layer. A few implementation details should be reviewed before treating the application as production-ready:

1. **Preprocessing parity:** The notebook defines a `clean_text()` function, but the currently published `app.py` sends the submitted form text directly to `vectorizer.transform()`. The training and inference preprocessing steps should be made identical. Otherwise, the web application may not reproduce the exact input preparation used during model training.
2. **Input validation:** Add server-side checks for missing, whitespace-only, and excessively long input. Browser-side `required` attributes are not a substitute for backend validation.
3. **Frontend input limit:** The current template's textarea has `maxlength="100"` while its helper text says the limit is 5,000 characters. These values should be reconciled.
4. **Template cleanup:** Review the textarea markup for the extra closing tag and remove obsolete commented-out markup.
5. **Debug configuration:** Keep debug mode for local development only. Use a production-ready WSGI server and secure deployment configuration for public hosting.
6. **Testing:** Add tests for the home route, valid prediction submissions, empty input, model-loading failures, label mapping, and confidence-score rendering.
7. **Reproducibility:** Record the dataset source/version, Python and library versions, model parameters, and the exact preprocessing/vectorizer configuration used to produce the committed model artifacts.
8. **Evaluation transparency:** Keep the reported test metrics linked to the exact model artifact and evaluation run. Avoid presenting the test accuracy as a guarantee of real-world reliability.

These are follow-up engineering tasks, not claims that the features have already been implemented.

## Future Improvements

Potential extensions, subject to implementation and testing, include:

- Consistent shared preprocessing for both training and inference.
- More robust input validation and user-facing error handling.
- A reproducible training script and documented environment.
- Broader evaluation on newer and more diverse datasets.
- External source retrieval and evidence-based fact-checking.
- Prediction history with appropriate privacy controls.
- Model monitoring and periodic evaluation for changing news patterns.
- Accessibility, responsive layout, and interface usability improvements.

## Project Team

| Name | Project role |
|---|---|
| Syeda Nazish | Developer |
| Afifa Naaz | Designer |
| Nabiji Zunera | Researcher |

## Repository

**GitHub:** [SyedaNazish-debug/verifier_model](https://github.com/SyedaNazish-debug/verifier_model)

**Model development notebook:** [VERIFIER_L_EX.ipynb](https://github.com/SyedaNazish-debug/verifier_model/blob/main/VERIFIER_L_EX.ipynb)

---

*Verifier Lense is an academic project demonstrating the integration of text preprocessing, feature extraction, supervised machine learning, and a Flask-based web application.*
