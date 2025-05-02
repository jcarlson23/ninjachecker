# predictor.py
import joblib

# Load the saved model and vectorizer
model = joblib.load('password_classifier.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_password(input_text):
    # Transform the input text using the same vectorizer
    input_vector = vectorizer.transform([input_text])
    
    # Make a prediction
    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]  # Probability of class 1
    
    return prediction, probability

def main():
    print("\n=== Password Classifier ===")
    print("Enter text to check if it's a password (or 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        prediction, probability = predict_password(user_input)
        
        if prediction == 1:
            print(f"RESULT: Likely a password (probability: {probability:.2f})")
        else:
            print(f"RESULT: Not a password (probability: {probability:.2f})")

if __name__ == "__main__":
    main()