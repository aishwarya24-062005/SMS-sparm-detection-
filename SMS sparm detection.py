import pandas as pd
from google.colab import files

# Upload CSV
uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))
print(df.head())

import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Load Dataset ---
df = pd.read_csv("spam.csv")  # Make sure spam.csv is in the same folder

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_message'] = df['message'].apply(clean_text)

# --- Features and Labels ---
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_message'])
y = df['label']

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Predict User Input ---
while True:
    user_input = input("\nEnter an SMS message (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    cleaned = clean_text(user_input)
    vector = cv.transform([cleaned])
    prediction = model.predict(vector)[0]
    print(f"Prediction: {prediction}")