# Advanced Fake News Detector

This project implements a sophisticated, hybrid system for detecting fake news. It combines a multi-model Machine Learning ensemble with a real-time, evidence-based verification engine that queries external APIs.

## Description

The core of this project is a two-pronged approach to classification:

1.  **ML-Based Analysis:** An ensemble of classic machine learning models and deep learning transformer models analyze the linguistic characteristics of a news headline.
2.  **Evidence-Based Verification:** The system extracts key entities (people, organizations, places) from the headline using spaCy and cross-references them with Wikipedia and a real-time news API.

A final weighted logic combines the signals from both approaches to deliver a more nuanced and accurate prediction, classifying headlines as REAL, FAKE, or NEUTRAL/UNCERTAIN.

## Team Members

* **Aryan Gupta** - [GitHub](https://github.com/Escanor-solos) | [LinkedIn](https://www.linkedin.com/in/aryan-gupta-b41214345/)

## Features
- **Hybrid Ensemble:** The core of the detector is a sophisticated ensemble combining five distinct models to analyze the text from different perspectives:
    - **Classic ML Models (using TF-IDF features):**
        - Logistic Regression
        - Multinomial Naive Bayes
        - LightGBM
    - **Transformer Models (Deep Learning):**
        - BERT (`bert-base-uncased`)
        - DistilBert (`distilbert-base-uncased`)
- **Real-Time Fact-Checking:** Utilizes Wikipedia and the News API to find real-world evidence supporting or refuting a claim.
- **Advanced Keyword Extraction:** Uses spaCy for Named Entity Recognition (NER) to identify the most relevant query terms.
- **Weighted Decision Logic:** A smart prediction engine that prioritizes real-world evidence when available but falls back to the powerful ML ensemble for linguistic analysis.
- **Interactive Prediction:** After training, the script launches an interactive loop to test any headline.

## Tech Stack
- **Core Libraries:** Python, Pandas, Scikit-learn
- **Deep Learning:** PyTorch, Hugging Face Transformers (BERT, DistilBert)
- **NLP:** spaCy
- **APIs:** News API, Wikipedia

## How to Run

1.  **Clone Repository:** Clone this repository to your local machine.

2.  **Set up Environment:** Create a Python virtual environment and install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file listing all packages like `pandas`, `torch`, `transformers`, `scikit-learn`, `spacy`, etc.)*

3.  **Download Datasets:**
    * Download the `Fake.csv` and `True.csv` datasets.
    * Place both `.csv` files in the root of your project folder.

4.  **Set API Key:**
    * Open the Python script and replace the placeholder for `NEWS_API_KEY` with your own key from [newsapi.org](https://newsapi.org/).

5.  **Train and Run:**
    * Run the script. The first time, it will train all the models and save the artifacts to your disk.
    * After the initial training, running the script again will load the saved models and start the interactive prediction loop.
    ```bash
    python FakeNews.py
    ```
