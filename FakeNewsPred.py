import pandas as pd
import numpy as np
import warnings
import os
import joblib
import requests
import wikipedia
import spacy
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
NEWS_API_KEY = "0a43321d66aa4b26ac777ea1110e5ac8" # IMPORTANT: Replace with your own key if you have one, this works just fine, avoid misuse and keep requests limited to 100 per 24 hrs
BERT_MODEL_NAME = 'bert-base-uncased'
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
TEST_SPLIT_SIZE = 0.2

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================
print("Loading local Fake.csv and True.csv files...")
if not (os.path.exists('Fake.csv') and os.path.exists('True.csv')):
    raise FileNotFoundError("ERROR: Fake.csv and/or True.csv not found. Please upload them.")

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake['label'] = 1
df_true['label'] = 0

df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['text'] = df['text'].astype(str)
df['full_text'] = df['title'] + " " + df['text'].fillna("")
print(f"Dataset loaded successfully. Total samples: {len(df)}")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['full_text'], df['label'], test_size=TEST_SPLIT_SIZE, random_state=42, stratify=df['label']
)

print("\nCreating TF-IDF features for classic models...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)
print("TF-IDF features created.")


# ==============================================================================
# 2. CLASSIC ML ENSEMBLE TRAINING
# ==============================================================================
print("\nDefining and training the classic ML model ensemble...")
model_lr = LogisticRegression(solver='liblinear', random_state=42)
model_nb = MultinomialNB()
model_lgbm = lgb.LGBMClassifier(random_state=42)

ensemble_classic = VotingClassifier(
    estimators=[('lr', model_lr), ('nb', model_nb), ('lgbm', model_lgbm)],
    voting='soft'
)
ensemble_classic.fit(X_train_tfidf, train_labels)
accuracy_classic = ensemble_classic.score(X_test_tfidf, test_labels)
print(f"\nClassic Ensemble Test Accuracy: {accuracy_classic:.4f}")


# ==============================================================================
# 3. PYTORCH MODEL DEFINITION
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PyTorchTransformerModel(nn.Module):
    def __init__(self, model_name, base_model_class):
        super(PyTorchTransformerModel, self).__init__()
        self.transformer = base_model_class.from_pretrained(model_name)
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0]
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits


# ==============================================================================
# 4. PYTORCH TRAINING AND EVALUATION LOOP
# ==============================================================================
def train_pytorch_model(model, train_loader, val_loader, epochs):
    """A complete training and validation loop for a PyTorch model."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), labels.float())
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                total_val_loss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                correct_predictions += (predictions.squeeze() == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        print(f"  Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
    return model


# ==============================================================================
# 5. DATA PREPARATION AND TRAINING OF TRANSFORMER MODELS
# ==============================================================================
# --- Prepare BERT Data ---
print("\nPreparing data for PyTorch BERT model...")
bert_tokenizer_pt = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
train_encodings_bert = bert_tokenizer_pt(train_texts.tolist(), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
test_encodings_bert = bert_tokenizer_pt(test_texts.tolist(), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
train_dataset_bert = TensorDataset(train_encodings_bert['input_ids'], train_encodings_bert['attention_mask'], torch.tensor(train_labels.values))
test_dataset_bert = TensorDataset(test_encodings_bert['input_ids'], test_encodings_bert['attention_mask'], torch.tensor(test_labels.values))
train_loader_bert = DataLoader(train_dataset_bert, batch_size=BATCH_SIZE, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=BATCH_SIZE)

# --- Train BERT Model ---
print("\nCreating and training PyTorch BERT model...")
model_bert_pt = PyTorchTransformerModel(BERT_MODEL_NAME, BertModel)
model_bert_pt = train_pytorch_model(model_bert_pt, train_loader_bert, test_loader_bert, EPOCHS)

# --- Prepare DistilBERT Data ---
print("\nPreparing data for PyTorch DistilBERT model...")
distilbert_tokenizer_pt = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
train_encodings_distil = distilbert_tokenizer_pt(train_texts.tolist(), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
test_encodings_distil = distilbert_tokenizer_pt(test_texts.tolist(), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
train_dataset_distil = TensorDataset(train_encodings_distil['input_ids'], train_encodings_distil['attention_mask'], torch.tensor(train_labels.values))
test_dataset_distil = TensorDataset(test_encodings_distil['input_ids'], test_encodings_distil['attention_mask'], torch.tensor(test_labels.values))
train_loader_distil = DataLoader(train_dataset_distil, batch_size=BATCH_SIZE, shuffle=True)
test_loader_distil = DataLoader(test_dataset_distil, batch_size=BATCH_SIZE)

# --- Train DistilBERT Model ---
print("\nCreating and training PyTorch DistilBERT model...")
model_distilbert_pt = PyTorchTransformerModel(DISTILBERT_MODEL_NAME, DistilBertModel)
model_distilbert_pt = train_pytorch_model(model_distilbert_pt, train_loader_distil, test_loader_distil, EPOCHS)


# ==============================================================================
# 6. SAVE MODEL ARTIFACTS
# ==============================================================================
print("\nSaving all model components...")
joblib.dump(ensemble_classic, 'classic_ensemble.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

torch.save(model_bert_pt.state_dict(), 'bert_model.pth')
bert_tokenizer_pt.save_pretrained('bert_tokenizer_pt')

torch.save(model_distilbert_pt.state_dict(), 'distilbert_model.pth')
distilbert_tokenizer_pt.save_pretrained('distilbert_tokenizer_pt')

print("All components saved successfully!")


# ==============================================================================
# 7. REAL-TIME PREDICTION PIPELINE
# ==============================================================================
nlp = spacy.load("en_core_web_sm")

def extract_entities_spacy(headline):
    doc = nlp(headline)
    entities, noun_chunks = [], []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
            entities.append(ent.text)
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        words = text.split()
        if words[0] in ["the", "a", "an", "this", "that"]: text = " ".join(words[1:])
        if len(text) > 3: noun_chunks.append(chunk.text)
    combined_keywords = sorted(list(set(entities + noun_chunks)), key=len, reverse=True)
    return combined_keywords if combined_keywords else [headline]

def get_wikipedia_evidence(keywords):
    if not keywords: return {"page_found": False}
    try:
        wikipedia.page(keywords[0], auto_suggest=False, redirect=True)
        return {"page_found": True}
    except Exception: return {"page_found": False}

def get_news_api_evidence(keywords, api_key):
    if not keywords or not api_key or api_key == "PASTE_YOUR_NEWS_API_KEY_HERE": return {"articles_found": 0, "trusted_sources_count": 0}
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    query = f'\"{keywords[0]}\"'
    url = f"https://newsapi.org/v2/everything?q={query}&from={seven_days_ago}&sortBy=relevancy&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'ok' and data['totalResults'] > 0:
            trusted_sources = ["Reuters", "Associated Press", "BBC News", "CNN", "The New York Times", "The Guardian", "The Times of India", "The Hindu"]
            count = 0
            for article in data['articles'][:5]:
                if any(source in article['source']['name'] for source in trusted_sources): count += 1
            return {"articles_found": data['totalResults'], "trusted_sources_count": count}
    except requests.exceptions.RequestException as e:
        print(f"-> News API Warning: {e}")
    return {"articles_found": 0, "trusted_sources_count": 0}

def predict_pytorch(text, model, tokenizer):
    model.to(device); model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        output = model(**inputs)
        probability = torch.sigmoid(output).item()
    return probability

def predict_news_with_evidence(title, api_key, classic_model, tfidf_vec, bert_m, bert_t, distil_m, distil_t):
    print("\n--- Starting Analysis ---")
    keywords = extract_entities_spacy(title)
    print(f"1. Extracted Keywords (using spaCy): {keywords}")
    print("2. Querying APIs for real-world evidence...")
    wiki_evidence = get_wikipedia_evidence(keywords)
    news_evidence = get_news_api_evidence(keywords, api_key)
    print(f"   - Wikipedia: Page found for '{keywords[0] if keywords else 'N/A'}': {wiki_evidence['page_found']}.")
    print(f"   - Real-Time News: Found {news_evidence['articles_found']} recent articles, with {news_evidence['trusted_sources_count']} from top sources.")

    # Gate 1: Strong REAL signal (hard rule).
    if news_evidence['trusted_sources_count'] >= 2:
        return "REAL (Verified by multiple recent news sources)", "Evidence-Based"

    # Gate 2: Strong UNVERIFIED signal (triggers weighted vote).
    elif news_evidence['articles_found'] == 0 and not wiki_evidence['page_found']:
        print("\n3. Strong signal that subject is unverified. Performing weighted vote...")
        api_prob_fake = 0.55
        model_prob_fake = (
            classic_model.predict_proba(tfidf_vec.transform([title]))[0][1] +
            predict_pytorch(title, bert_m, bert_t) +
            predict_pytorch(title, distil_m, distil_t)
        ) / 3.0
        final_prob_fake = (0.50 * api_prob_fake) + (0.50 * model_prob_fake)
        print(f"   - API Evidence Score (prob FAKE):   {api_prob_fake:.3f} (Weight: 50%)")
        print(f"   - ML Ensemble Score (prob FAKE):    {model_prob_fake:.3f} (Weight: 50%)")
        decision_method = "Weighted API + ML"

    # Inconclusive Case: Fallback to 100% ML ensemble vote.
    else:
        print("\n3. API evidence is inconclusive. Falling back to 5-model ML ensemble...")
        pred_classic_prob = classic_model.predict_proba(tfidf_vec.transform([title]))[0][1]
        pred_bert_prob = predict_pytorch(title, bert_m, bert_t)
        pred_distil_prob = predict_pytorch(title, distil_m, distil_t)
        final_prob_fake = (pred_classic_prob + pred_bert_prob + pred_distil_prob) / 3.0
        print(f"   - Classic Ensemble (prob FAKE): {pred_classic_prob:.3f}")
        print(f"   - BERT Model (prob FAKE):       {pred_bert_prob:.3f}")
        print(f"   - DistilBERT Model (prob FAKE): {pred_distil_prob:.3f}")
        decision_method = "ML-Based"

    if final_prob_fake > 0.65: label = "FAKE"
    elif final_prob_fake < 0.35: label = "REAL"
    else: label = "NEUTRAL / UNCERTAIN"

    print(f"   -> Final Weighted/Averaged (prob FAKE): {final_prob_fake:.3f}")
    return f"{label} (Classified by {decision_method})", decision_method


# ==============================================================================
# 8. INTERACTIVE PREDICTION LOOP
# ==============================================================================
def run_interactive_session():
    print("\nLoading all model components for prediction...")
    ensemble_classic = joblib.load('classic_ensemble.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

    model_bert_pt = PyTorchTransformerModel(BERT_MODEL_NAME, BertModel)
    model_bert_pt.load_state_dict(torch.load('bert_model.pth'))
    bert_tokenizer_pt = BertTokenizer.from_pretrained('bert_tokenizer_pt')

    model_distilbert_pt = PyTorchTransformerModel(DISTILBERT_MODEL_NAME, DistilBertModel)
    model_distilbert_pt.load_state_dict(torch.load('distilbert_model.pth'))
    distilbert_tokenizer_pt = DistilBertTokenizer.from_pretrained('distilbert_tokenizer_pt')

    print("All components loaded successfully!")

    if NEWS_API_KEY == "PASTE_YOUR_NEWS_API_KEY_HERE" or NEWS_API_KEY == "":
        print("\n\nWARNING: NEWS_API_KEY is not set. The real-time news check will not work.")

    while True:
        try:
            news_title = input("\nEnter a news headline to check (or 'exit' to quit): ")
            if news_title.lower() == 'exit': break
            if not news_title.strip(): continue

            final_label, reason = predict_news_with_evidence(
                news_title,
                NEWS_API_KEY,
                ensemble_classic,
                tfidf_vectorizer,
                model_bert_pt,
                bert_tokenizer_pt,
                model_distilbert_pt,
                distilbert_tokenizer_pt
            )
            print(f"\n-> Final Prediction: {final_label}")
            print(f"   (Decision Method: {reason})")
        except KeyboardInterrupt:
            print("\nExiting session.")
            break

if __name__ == "__main__":
    # This block can be used to either train or run interactively.
    # For now, we assume if models exist, we run interactively.
    if os.path.exists('bert_model.pth'):
        run_interactive_session()
    else:
        print("Models not found. Please run the script once to train and save the models.")
        # The main training logic from cell 1-6 would be here.
        # For simplicity, this script is separated into a training phase (implicit)
        # and a prediction phase (explicitly called).