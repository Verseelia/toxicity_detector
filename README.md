# Comment Toxicity Detector

A multi-label NLP application that detects different types of toxicity in user-provided comments using a fine-tuned **Toxic-BERT** model and provides predictions through a **Streamlit web application**.

## Project Overview

Online platforms receive a large number of user comments every day. Some comments may contain toxic, offensive, threatening, or hateful language.

The goal of this project is to automatically identify different categories of toxicity in a given comment.

Unlike a simple binary classifier that predicts only whether a comment is toxic or non-toxic, this project performs **multi-label classification**. This means that a single comment can belong to multiple toxicity categories at the same time.

For example, a comment could be classified as both:

* Toxic
* Insult
* Obscene

## Toxicity Categories

The model predicts six different toxicity categories:

| Label           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `toxic`         | General toxic or abusive language                    |
| `severe_toxic`  | Highly severe toxic language                         |
| `obscene`       | Obscene or vulgar language                           |
| `threat`        | Threatening language                                 |
| `insult`        | Insulting or abusive language directed at someone    |
| `identity_hate` | Hate or abusive language targeting an identity group |

## Project Workflow

```text
User enters a comment
        ↓
Streamlit Application
        ↓
Text Tokenization
        ↓
Toxic-BERT Model
        ↓
Probability for each toxicity category
        ↓
Threshold-based classification
        ↓
Predicted toxicity labels
```

## Dataset

The project uses the **Jigsaw Toxic Comment Classification** dataset.

The training dataset contains approximately:

* **159,571 comments**
* **6 toxicity labels**

The six target columns are:

```text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

Before model training, the dataset was checked for:

* Missing values
* Duplicate records
* Data structure
* Label distribution

## Model

### Toxic-BERT

The primary model used in this project is:

```text
unitary/toxic-bert
```

Toxic-BERT is a BERT-based language model specifically trained for toxicity detection.

BERT is useful for this task because it considers the context of words within a sentence rather than treating each word independently.

For example:

```text
"You are stupid"
```

The model does not simply look for the word `"stupid"`. It considers the surrounding context when determining the toxicity categories.

### Tokenization

The input text is converted into tokens before being passed to the model.

The tokenizer prepares the text in the format expected by BERT.

The project uses:

```text
max_length = 128
```

This limits the maximum number of tokens processed for a comment.

## Multi-Label Classification

This project uses **multi-label classification** rather than multi-class classification.

### Multi-class

Only one class can be selected.

```text
Comment → Toxic
```

### Multi-label

Multiple labels can be selected.

```text
Comment → Toxic + Insult + Obscene
```

This is more appropriate for toxicity detection because different types of toxic behavior can occur in the same comment.

## Threshold Optimization

The model produces a probability for each toxicity category.

For example:

```text
toxic          → 0.91
severe_toxic   → 0.20
obscene        → 0.84
threat         → 0.10
insult         → 0.78
identity_hate  → 0.15
```

Instead of using the same threshold for every category, individual thresholds were used.

The optimized thresholds were saved in:

```text
thresholds.npy
```

This is particularly useful because some toxicity categories are much rarer than others.

For example, `threat` and `severe_toxic` have fewer positive examples than the general `toxic` category.

## Model Performance

The Toxic-BERT model achieved the following approximate F1 scores:

| Category      | F1 Score |
| ------------- | -------: |
| Toxic         |     0.91 |
| Severe Toxic  |     0.45 |
| Obscene       |     0.88 |
| Threat        |     0.67 |
| Insult        |     0.85 |
| Identity Hate |     0.74 |
| **Micro F1**  | **0.86** |

The performance varies between categories because the dataset is highly imbalanced. Some toxicity categories have significantly fewer examples than others.

## Streamlit Application

A Streamlit application was developed to allow users to interact with the trained model.

The application accepts:

```text
Comment Text
```

and returns the predicted toxicity categories.

### Example

Input:

```text
You are such an idiot.
```

Possible output:

```text
Toxic
Insult
```

The application is designed to make the trained NLP model accessible without requiring the user to write Python code.

## Project Structure

```text
comment_toxicity/
│
├── app.py
├── toxicbert_model/
│   └── trained model files
│
├── thresholds.npy
│
├── requirements.txt
│
├── README.md
│
└── ...
```

> The exact files may vary depending on the final version of the project.

## Technologies Used

### Programming Language

* Python

### Machine Learning / NLP

* PyTorch
* Hugging Face Transformers
* Toxic-BERT
* NumPy

### Application

* Streamlit

### Development Environment

* Python Virtual Environment
* Jupyter Notebook / Python
* VS Code

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd comment_toxicity
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the browser.

The default local address is usually:

```text
http://localhost:8501
```

## How the Application Works

### Step 1 — User Input

The user enters a comment in the Streamlit interface.

### Step 2 — Tokenization

The comment is converted into tokens using the Toxic-BERT tokenizer.

### Step 3 — Model Prediction

The tokenized input is passed to the trained Toxic-BERT model.

The model generates a probability for each of the six toxicity categories.

### Step 4 — Thresholding

The probabilities are compared against the corresponding optimized thresholds.

### Step 5 — Final Prediction

The application displays the toxicity categories predicted for the comment.

## Why Toxic-BERT?

A general machine-learning model based only on manually engineered features may not capture the contextual meaning of language effectively.

Toxic-BERT is useful because:

* It is based on the BERT architecture.
* It understands contextual relationships between words.
* It was specifically trained for toxicity-related language.
* It can capture patterns that are difficult to represent using simple keyword-based approaches.

## Why F1 Score?

The dataset contains significant class imbalance.

For example, some categories such as `threat` and `severe_toxic` contain far fewer positive examples than `toxic`.

Accuracy alone can therefore be misleading.

F1 score combines:

```text
Precision + Recall
```

and provides a more useful measure for evaluating classification performance on imbalanced classes.

## Challenges

### 1. Class Imbalance

Some toxicity categories have significantly fewer positive examples.

This makes minority-category prediction more difficult.

### 2. Overlapping Categories

A single comment can belong to multiple categories.

For example:

```text
Toxic + Insult + Obscene
```

Therefore, the problem cannot be treated as a simple single-class classification problem.

### 3. Threshold Selection

A single threshold may not provide the best performance for every toxicity category.

Individual thresholds were therefore used for the different labels.

### 4. Deployment

The trained NLP model needs to be loaded efficiently when the Streamlit application starts.

The application uses Streamlit resource caching to avoid unnecessarily loading the model repeatedly.

## Earlier Model Approach

An earlier version of the project also explored a **Bidirectional LSTM** approach.

The LSTM model used:

* Tokenization
* Embedding layer
* Bidirectional LSTM
* Dropout
* Dense layers
* Sigmoid output

The final Toxic-BERT approach provided stronger overall performance and was used as the primary model for the application.

## Future Improvements

Possible future improvements include:

* Improving minority-class performance
* Further threshold optimization
* Experimenting with additional transformer models
* Adding confidence scores to the Streamlit interface
* Improving the user interface
* Adding batch prediction for multiple comments
* Monitoring model performance on new data
* Deploying the application to a cloud platform

## Key Learning Outcomes

Through this project, I worked with:

* Natural Language Processing
* Text preprocessing
* Transformer-based models
* BERT
* Multi-label classification
* Imbalanced datasets
* Precision, Recall and F1 score
* Threshold optimization
* Model inference
* Streamlit application development
* Model deployment workflow

## Conclusion

The **Comment Toxicity Detector** demonstrates how transformer-based NLP models can be used to automatically identify different types of toxic language.

The project combines a **Toxic-BERT model** with a **Streamlit application**, allowing users to enter a comment and receive predictions for multiple toxicity categories.

The final model achieved an approximate **micro F1 score of 0.86**, demonstrating strong overall performance while also highlighting the challenges of detecting less frequent toxicity categories.
