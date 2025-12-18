```
# 📧 Spam Mail Detection System

## 📌 Project Overview
This project is a **Machine Learning–based Spam Mail Detection System** developed using **Python and Scikit-learn**.  
The system classifies an email or text message as **Spam** or **Not Spam (Ham)** based on its content.

An interactive **Streamlit web application** is also included to allow real-time spam detection using a trained model.

---

## 🎯 Problem Statement
With the rapid growth of digital communication, spam emails have become a major problem.  
Spam emails:
- Waste time
- Spread scams and malware
- Reduce productivity

This project aims to automatically detect spam messages using **Natural Language Processing (NLP)** and **Machine Learning**.

---

## 📂 Dataset Information
- **Dataset Type:** SMS/Email text data
- **Target Variable:** `label`
  - `1 → Spam`
  - `0 → Not Spam`
- **Data Format:** Text-based dataset

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:
- Converted text to lowercase
- Removed stopwords
- Converted text into numerical form using **TF-IDF Vectorization**
- Split data into training and testing sets

---

## 🔤 Text Vectorization
The text data was converted into numerical features using:

- **TF-IDF Vectorizer**
- Removes common English stopwords
- Assigns importance based on word frequency

The trained vectorizer was saved as:
```

vectorizer.pkl

````

---

## 🤖 Model Selection
- **Algorithm Used:** Logistic Regression
- **Reason for Selection:**
  - Efficient for text classification
  - Works well with TF-IDF features
  - Fast and lightweight
  - Suitable for binary classification

---

## 📊 Model Evaluation
The model was evaluated using a test dataset.

| Metric | Value |
|------|------|
| **Accuracy Score** | 96% |

The high accuracy indicates that the model effectively distinguishes spam from non-spam messages.

---

## 💾 Model Saving
The trained components were saved using `joblib`:

```
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
````

Saved files:

* `model.pkl` → trained Logistic Regression model
* `vectorizer.pkl` → TF-IDF vectorizer

---

## 🌐 Streamlit Web Application

A Streamlit-based web application allows users to:

* Enter email or message text
* View model accuracy
* Instantly check if a message is spam or not

### 🖥 Application Features

* Simple and clean UI
* Real-time prediction
* Accuracy score display
* Error handling for empty input

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```
pip install streamlit scikit-learn numpy pandas joblib
```

### 2️⃣ Project Structure

```
Spam-Mail-Detection/
│── app.py
│── model.pkl
│── vectorizer.pkl
│── README.md
```

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

---

## 🧠 Prediction Workflow

1. User enters email or message text
2. Text is transformed using the saved TF-IDF vectorizer
3. Transformed data is passed to the trained model
4. Model predicts Spam or Not Spam
5. Result is displayed on the UI

---

## ⚠️ Limitations

* Accuracy depends on training data quality
* Cannot detect spam patterns not present in training data
* Very short messages may reduce accuracy

---

## 🚀 Future Enhancements

* Add spam probability score
* Highlight suspicious words
* Support multiple languages
* Deploy app online using Streamlit Cloud
* Use advanced models like Naive Bayes or SVM

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* NLP (TF-IDF)

---

## 👨‍🎓 Academic Use

This project is suitable for:

* Machine Learning assignments
* NLP projects
* Semester projects
* Model deployment demonstrations
* Viva and project defense

---

## 📌 Author

**Alaqmar Yousuf**
Machine Learning & Software Engineering Student

---
