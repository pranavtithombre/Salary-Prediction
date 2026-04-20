# 💼 Salary Prediction Model

This project builds a machine learning model to predict salaries based on various input features such as experience, job-related attributes, and other factors. It compares multiple regression algorithms and selects the best-performing model.

---

## 🚀 Project Overview

The goal of this project is to solve a regression problem where the target variable is **Salary**. Multiple models are trained and evaluated to determine the most accurate one.

The following algorithms are used:

* Linear Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVR)
* Decision Tree Regressor
* Random Forest Regressor

---

## 📊 Workflow

1. **Data Loading**

   * Dataset loaded using Pandas

2. **Data Preprocessing**

   * Handling missing values:

     * Numerical → mean
     * Categorical → mode
   * Encoding categorical features using Label Encoding

3. **Feature Selection**

   * Input features (X)
   * Target variable (y = Salary)

4. **Train-Test Split**

   * 80% training, 20% testing

5. **Feature Scaling**

   * Applied using StandardScaler (for KNN and SVM)

6. **Model Training & Evaluation**

   * Metrics used:

     * Mean Squared Error (MSE)
     * Root Mean Squared Error (RMSE)
     * R² Score

7. **Model Comparison**

   * All models compared using R² score visualization

8. **Model Saving**

   * Best model saved as:

     ```
     best_model.pkl
     ```

---

## 📈 Results

* Models are evaluated and compared visually using a bar chart
* The model with the highest **R² score** is selected as the best model
* The best model is saved using pickle for future predictions

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle

---

## 📁 Project Structure

```
├── salary_prediction_model_building.py
├── Salary_Dataset_.csv
├── best_model.pkl
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/salary-prediction.git
```

2. Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Run the script:

```
python salary_prediction_model_building.py
```

---

## 💡 Key Highlights

* Multiple model comparison in one pipeline
* Automatic handling of missing data
* Feature scaling for better performance
* Model selection based on evaluation metrics
* Model persistence using pickle

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Use advanced models like XGBoost
* Deploy using Flask or Streamlit
* Add real-time prediction interface

---

## 📬 Conclusion

This project demonstrates how machine learning can be applied to predict salaries and compare different regression models effectively. It is a great beginner-to-intermediate level project for understanding the full ML pipeline.

---
