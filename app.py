import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Accuracy from training
ACCURACY = 0.96   # replace with your actual value

st.title("📧 Spam Mail Detection System")
st.write("Check whether an email is Spam or Not Spam")

st.subheader("📊 Model Performance")
st.write(f"**Accuracy Score:** {ACCURACY * 100:.2f}%")

st.divider()

message = st.text_area("✉️ Enter Email Text", height=150)

if st.button("Check Email"):
    if message.strip() == "":
        st.warning("Please enter an email message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM**")
