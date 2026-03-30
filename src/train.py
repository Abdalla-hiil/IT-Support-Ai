import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
df = pd.read_csv('dataset/IT_support_tickets.csv')

# 2. THE CLEANING (Following your old style)
# Remove Nulls
df = df.where(pd.notnull(df), "")

# DROP DUPLICATES (Very important - prevents the model from being biased)
df = df.drop_duplicates(keep='first')

# Clean the text: Lowercase and strip spaces
df['Body'] = df['Body'].str.lower().str.strip()
df['Priority'] = df['Priority'].str.lower().str.strip()

# 3. Label Mapping
label_map = {"high": 0, "medium": 1, "low": 2}
df['Priority_Label'] = df['Priority'].map(label_map)

# Remove any rows that didn't map correctly (security check)
df = df.dropna(subset=['Priority_Label'])

X = df["Body"]
y = df["Priority_Label"].astype(int)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Text -> TF-IDF (Adding ngram_range to catch "Server is down")
# ngram_range=(1,2) helps the AI see "Server down" as one idea, not just "Server"
tfidf = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True, ngram_range=(1,2))
X_train_features = tfidf.fit_transform(X_train)

# 6. Train the Model (RandomForest is usually best for this)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_features, y_train)

# 7. Save for your API
joblib.dump(rf, 'models/it_priority_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print(f"Success! Model trained on {len(df)} unique tickets.")