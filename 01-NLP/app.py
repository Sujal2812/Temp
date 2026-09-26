# app.py
import streamlit as st
import pickle
import spacy

# Load model and vectorizer (cached for performance)
import os

@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

    return model, vectorizer, nlp

# Preprocessing function (same as training)
def preprocess_text(text, nlp):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

# Prediction function
def predict_category(headline, model, vectorizer, nlp):
    processed = preprocess_text(headline, nlp)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    return pred

# Streamlit UI
st.title("NewsBot - Headline Classifier")
st.write("Enter a news headline to predict its category.")

headline = st.text_input("Headline:")
if st.button("Predict"):
    if headline:
        model, vectorizer, nlp = load_artifacts()
        category = predict_category(headline, model, vectorizer, nlp)
        st.success(f"Predicted Category: **{category}**")
    else:
        st.warning("Please enter a headline.")

# Optional: Terminal chatbot if you run as a script (not needed for Streamlit)
# You can also run this file directly to get a terminal chatbot.
if __name__ == "__main__":
    import sys
    # If no arguments, run Streamlit app
    if len(sys.argv) == 1:
        # This part is actually handled by streamlit run command.
        # But if you run `python app.py` directly, we can fallback to terminal.
        print("To use the Streamlit app, run: streamlit run app.py")
        print("Or you can use the terminal chatbot by running: python app.py chat")
    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # Terminal chatbot
        model, vectorizer, nlp = load_artifacts()
        print("NewsBot: Hello! Enter a news headline to get its category. Type 'quit' or 'exit' to stop.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("NewsBot: Goodbye!")
                break
            category = predict_category(user_input, model, vectorizer, nlp)
            print(f"NewsBot: The category is {category}")