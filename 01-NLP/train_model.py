# train_model.py
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# 1. Load data
# ------------------------------
print("Loading data...")
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)

# 2. Filter categories
categories = ['TECHNOLOGY', 'ENTERTAINMENT', 'POLITICS', 'BUSINESS']
df = df[df['category'].isin(categories)]
print(f"Kept {len(df)} rows with categories: {df['category'].unique()}")

# 3. Load spaCy model (disable unnecessary components for speed)
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

# 4. Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# 5. Apply preprocessing
print("Preprocessing headlines...")
df['processed_headline'] = df['headline'].apply(preprocess_text)

# 6. Vectorization (unigrams + bigrams, max 5000 features)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['processed_headline'])
y = df['category']

# 7. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train Logistic Regression
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 10. Save model and vectorizer
print("Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Done. Files saved: model.pkl, vectorizer.pkl")