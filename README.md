# 🌾 Crop Yield Prediction using Machine Learning  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=plotly) ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-3792cb?logo=seaborn)  

---

##  Overview  
This project applies **Machine Learning regression models** to predict **crop yield** based on agricultural and environmental factors.  
The goal is to **train, compare, and evaluate** multiple algorithms to determine the best-performing model.  

---

##  Models Used  
- 📈 **Linear Regression**  
- 👥 **K-Nearest Neighbors (KNN) Regressor**  
- 🧮 **Lasso Regression**  
- ➗ **Ridge Regression**  
- 🌳 **Decision Tree Regressor**  

---

##  Objective  
- Build an ML pipeline for **predicting crop yields**.  
- Compare multiple regression algorithms.  
- Select the most **accurate and efficient** model.  

---
## How to work on the Pickle File :
```python
import pickle

# Save the trained model
pickle.dump(model, open('model.pkl', 'wb'))

# Load the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))
```
## Significance of the Pickle File

The **model.pkl** file stores the trained Random Forest model in binary format.

**1) Model Reusability**: You don’t need to retrain the model every time you run predictions.

**2) Deployment Ready**: Can be easily integrated into web apps (Flask, Streamlit, etc.) for live predictions.

**3) Efficiency**: Reduces computational load and saves time.

**4) Consistency**: Ensures the same trained model is used across all environments.

 **dtr.pkl** and **preprocessor.pkl** act as the bridge between model training and deployment, enabling real-time rainfall predictions.

---

##  Results  
- Performance was evaluated using:  
  - ✅ R² Score  
  - ✅ Mean Absolute Error (MAE)  
- The comparison shows the **best regression model** for yield prediction.  

---

