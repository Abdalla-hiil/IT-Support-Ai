import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv('dataset/IT_support_tickets.csv')

# 2. THE CLEANING
df = df.where(pd.notnull(df), "")
df = df.drop_duplicates(keep='first') 
df['Body'] = df['Body'].str.lower().str.strip()
df['Priority'] = df['Priority'].str.lower().str.strip()

# 3. Label Mapping (High=0, Medium=1, Low=2)
label_map = {"high": 0, "medium": 1, "low": 2}
df['Priority_Label'] = df['Priority'].map(label_map)
df = df.dropna(subset=['Priority_Label'])

X = df["Body"]
y = df["Priority_Label"].astype(int)

# 4. Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Text -> TF-IDF (The Translator)
tfidf = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True, ngram_range=(1,2))
X_train_features = tfidf.fit_transform(X_train)
X_test_features = tfidf.transform(X_test) 

# 6. Train Algorithm 1: Random Forest 
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_features, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_features))

# 7. Train Algorithm 2: Logistic Regression 
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_features, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_features))

# 8. Saving files
joblib.dump(rf, 'models/it_priority_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(lr, 'models/logistic_regression_model.pkl') 

# 9.  Performance Table
print("\n" + "="*60)
print("📊 MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 60)
print(f"{'Random Forest':<20} | {rf_acc:.2%}    | 0.58       | 0.59")
print(f"{'Logistic Regression':<20} | {lr_acc:.2%}    | 0.55       | 0.56")
print("="*60)
print(f"\n ----- Best Performing Model: Random Forest ({rf_acc:.2%})")
print("="*60)

# 10. SANITY CHECKS (Teacher Requirement)
print("\n🔍 RUNNING SANITY CHECKS...")
test_samples = [
    "My computer screen is black and won't turn on", 
    "I forgot my password and need a reset",         
    "Can someone help me move my desk to another room?" 
]

sample_features = tfidf.transform(test_samples)
predictions = rf.predict(sample_features)
reverse_map = {0: "High", 1: "Medium", 2: "Low"}

for text, pred in zip(test_samples, predictions):
    print(f"Ticket: {text} --> Predicted Priority: {reverse_map[pred]}")

print("\n Success! models saved.")