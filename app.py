import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Set page config
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")

# Title and description
st.title("📧 Spam Mail Detector")
st.markdown("Enter an email message below to check if it's spam or not.")

# Load and train model (cached to avoid retraining)
@st.cache_resource
def load_model():
    # You'll need to upload your mail_data.csv file
    # For now, I'll show how to load it
    try:
        df = pd.read_csv("mail_data.csv")
        data = df.where((pd.notnull(df)), '')
        
        # Label encoding
        data.loc[data['Category'] == 'spam', 'Category'] = 0
        data.loc[data['Category'] == 'ham', 'Category'] = 1
        
        # Prepare features and labels
        X = data["Message"]
        y = data["Category"].astype('int')
        
        # Split data
        X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature extraction
        feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train_features, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, feature_extraction, accuracy
    except FileNotFoundError:
        st.error("Please upload the 'mail_data.csv' file to the same directory as this app.")
        return None, None, None

# Load model
model, feature_extraction, accuracy = load_model()

if model is not None and feature_extraction is not None:
    # Display model accuracy
    st.sidebar.header("Model Information")
    st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    st.sidebar.info("Model: Logistic Regression\nFeatures: TF-IDF Vectorization")
    
    # Input section
    st.subheader("Enter Email Message")
    user_input = st.text_area("", height=150, placeholder="Paste your email message here...")
    
    # Predict button
    if st.button("Check Email", type="primary"):
        if user_input.strip():
            # Transform input
            input_features = feature_extraction.transform([user_input])
            
            # Make prediction
            prediction = model.predict(input_features)
            
            # Display result
            st.divider()
            if prediction[0] == 1:
                st.success("✅ This is a **HAM** (legitimate) email!")
                st.balloons()
            else:
                st.error("⚠️ This is a **SPAM** email!")
            
        else:
            st.warning("Please enter an email message to check.")
    
    # Example emails section
    with st.expander("📝 Try Example Emails"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Example: Spam"):
                st.session_state.example = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
        
        with col2:
            if st.button("Example: Ham"):
                st.session_state.example = "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if that time still works for you."
        
        if 'example' in st.session_state:
            st.text_area("Example message:", value=st.session_state.example, height=100, key="example_text")

else:
    st.warning("⚠️ Model could not be loaded. Please ensure 'mail_data.csv' is in the same directory.")
    st.info("Upload your mail_data.csv file to get started!")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by Scikit-learn")