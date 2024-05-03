import streamlit as st

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

st.title("Sentiment Analysis App")
user_input = st.text_input("Enter text to analyze:")

if user_input:
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Make a prediction
    outputs = model(**inputs)
    preds = outputs.logits.argmax(-1).item()

    # Display the result
    if preds == 0:
        st.write("The sentiment is negative")
    else:
        st.write("The sentiment is positive")