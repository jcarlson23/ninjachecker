import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib 

# Load your CSV file
data = pd.read_csv('password_data.csv', names=['text', 'is_password'])

# Convert boolean labels to integers
def convert_to_binary(x):
    if isinstance(x, str):
        return 1 if x.strip().lower() in ('1', 'true', 'yes', 't', 'y') else 0
    elif isinstance(x, (int, float)):
        return 1 if x == 1 else 0
    else:
        return 0

data['is_password'] = data['is_password'].apply(convert_to_binary)

# Add these debugging lines
print("Label distribution before split:")
print(data['is_password'].value_counts())
print("Sample data:")
print(data.head(10))

# Feature extraction: transform the text data into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'].values)

# Target variable
y = data['is_password'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add these debugging lines
print("\nTraining set label distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set label distribution:")
print(pd.Series(y_test).value_counts())

# Use stratified sampling to ensure both classes are represented in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check distribution again
print("\nTraining set label distribution after stratified split:")
print(pd.Series(y_train).value_counts())

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'password_classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')