# ❤️ HeartGuard AI – Heart Disease Risk Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Purpose-Educational-green)

HeartGuard AI is a machine learning application that predicts **heart disease risk** using patient-related health and clinical features. It uses a **Neural Network built with TensorFlow/Keras** and provides an interactive **Streamlit web application** that generates a **heart disease risk probability**.

> ⚠️ **Disclaimer:** This is an educational project. Predictions are not a medical diagnosis or professional medical advice. See the [Medical Disclaimer](#️-medical-disclaimer) section.

---

## 📑 Table of Contents

- [Project Objective](#-project-objective)
- [Key Features](#-key-features)
- [Technologies Used](#️-technologies-used)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Information](#-model-information)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Important Files](#-important-files)
- [Installation](#-installation)
- [How to Run the Application](#-how-to-run-the-application)
- [Expected Application Output](#-expected-application-output)
- [Use Cases](#-use-cases)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Medical Disclaimer](#️-medical-disclaimer)

---

## 🎯 Project Objective

Heart disease is one of the major health concerns worldwide, and early risk assessment can help identify people who may need further medical evaluation.

The objective of HeartGuard AI is to demonstrate how **Machine Learning**, **Neural Networks**, and **Streamlit** can be combined to build an interactive health-risk prediction application, from data preprocessing and model training to a simple user-facing interface.

---

## ✨ Key Features

- 🧠 Neural Network-based heart disease risk prediction
- 📊 Outputs a heart disease **risk probability**
- 🖥️ Interactive **Streamlit** web interface
- 📏 Feature scaling using a saved scaler for consistent preprocessing
- 💾 Pre-trained Keras model included in the repository
- 📓 Jupyter Notebook documenting analysis and model development
- 🐍 Fully Python-based implementation

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning / Deep Learning | TensorFlow, Keras, Scikit-learn |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook |

---

## 🔄 Machine Learning Workflow

```text
Patient Data
     ↓
Data Preprocessing
     ↓
Feature Scaling
     ↓
Neural Network Model
     ↓
Heart Disease Risk Probability
```

1. **Data collection** – Health and clinical features are loaded from the dataset.
2. **Data preprocessing** – The data is cleaned and prepared for modeling.
3. **Feature scaling** – Features are scaled using a scaler saved as `scalerHeart.pkl`, so the same transformation is applied during prediction.
4. **Model training** – A Neural Network is trained using TensorFlow/Keras.
5. **Prediction** – The Streamlit app takes user input, scales it, and passes it to the trained model.
6. **Output** – The app displays a heart disease risk probability.

---

## 🧠 Model Information

| Property | Details |
|----------|---------|
| Model Type | Neural Network |
| Framework | TensorFlow / Keras |
| Number of Model Features | 28 |
| Output | Heart disease risk probability |
| Saved Model File | `Heart_diseases.keras` |
| Saved Scaler File | `scalerHeart.pkl` |

> 📌 Model performance metrics (such as accuracy, precision, recall, or ROC-AUC) are not reported in this README. Adding a proper evaluation is listed under [Future Improvements](#-future-improvements).

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| File Name | `heart_disease_risk_2026.csv` |
| Size | Approximately 9,000 rows |
| Content | Patient-related health and clinical features |
| Features Used by the Model | 28 |

The dataset is used to train and evaluate the Neural Network that predicts heart disease risk.

---

## 📂 Project Structure

```text
HeartGuard-AI-Heart-Disease-Prediction/
│
├── app.py                          # Main Streamlit application
├── app(1).py                       # Additional / alternate application file
├── Heart_Attack_Project.ipynb      # Project analysis and model development
├── Heart_diseases.keras            # Trained Keras Neural Network model
├── scalerHeart.pkl                 # Saved feature scaler
├── heart_disease_risk_2026.csv     # Dataset
└── README.md                       # Project documentation
```

---

## 📄 Important Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application. Run this file to launch HeartGuard AI. |
| `app(1).py` | Additional / alternate version of the application file. |
| `Heart_Attack_Project.ipynb` | Jupyter Notebook containing the project analysis and development. |
| `Heart_diseases.keras` | Trained TensorFlow/Keras Neural Network model. |
| `scalerHeart.pkl` | Saved feature scaler used to preprocess input data before prediction. |
| `heart_disease_risk_2026.csv` | Dataset used to train and evaluate the model. |
| `README.md` | Project documentation. |

---

## ⚙️ Installation

### Requirements

- Python 3.x
- pip (Python package manager)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Vaishnavi5544/HeartGuard-AI-Heart-Disease-Prediction.git
cd HeartGuard-AI-Heart-Disease-Prediction
```

**2. (Optional) Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install the required libraries**

```bash
pip install streamlit pandas numpy tensorflow scikit-learn
```

---

## ▶️ How to Run the Application

1. Make sure you are inside the project folder and that `app.py`, `Heart_diseases.keras`, and `scalerHeart.pkl` are in the same directory.
2. Install the libraries as shown above.
3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Streamlit will start a local server and open the app in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

5. Enter the required health-related information in the app.
6. View the predicted **heart disease risk probability**.

---

## 🖥️ Expected Application Output

When the app is running, you can expect:

- A **HeartGuard AI** web page opened in your browser
- Input fields for the patient-related health and clinical features
- A prediction step that processes the entered values using the saved scaler and trained model
- A displayed **heart disease risk probability**, for example:

```text
Predicted Heart Disease Risk Probability: XX%
```

> The value above is only a format example. The actual output depends on the values entered. It is a machine learning estimate and **not** a medical diagnosis.

---

## 💡 Use Cases

- 🎓 **Learning project** – Understand an end-to-end ML workflow from data to a deployed-style app
- 🧪 **Neural Network practice** – Example of tabular-data classification with TensorFlow/Keras
- 🖥️ **Streamlit demonstration** – Shows how to turn a trained model into an interactive web app
- 📁 **Portfolio project** – Showcases Python, machine learning, and application-building skills
- 🔬 **Base for experimentation** – Can be extended with more models, metrics, and explainability tools

---

## 📌 Future Improvements

- [ ] Improve model performance through hyperparameter tuning
- [ ] Add other machine learning models for comparison
- [ ] Add model performance metrics (accuracy, precision, recall, F1-score)
- [ ] Add a confusion matrix and ROC-AUC visualization
- [ ] Improve the UI/UX of the Streamlit application
- [ ] Add explainable AI techniques such as SHAP
- [ ] Deploy the application online
- [ ] Add automated model monitoring
- [ ] Add a `requirements.txt` file for easier setup

---

## 👩‍💻 Author

**Vaishnavi Jadhav**

🔗 GitHub: [https://github.com/Vaishnavi5544](https://github.com/Vaishnavi5544)

If you found this project useful, consider giving it a ⭐ on GitHub!

---

## ⚠️ Medical Disclaimer

**HeartGuard AI is an educational machine learning project.**

The predictions generated by this application should **not** be considered a medical diagnosis, treatment recommendation, or professional medical advice. The model is intended for learning and demonstration purposes only and has not been clinically validated.

Always consult a qualified healthcare professional for any medical concern or decision.
