# ✈️ Airline Flight Delay Analysis & Prediction

## 📌 Project Overview

**Airline Flight Delay Analysis & Prediction** is a machine learning project developed for the Statistical Inference Lab. It analyzes historical airline flight data, applies statistical techniques, compares regression algorithms, and predicts estimated arrival delays based on flight details.

The project also includes a **Streamlit web application** that allows users to enter flight information, view delay predictions, and explore model evaluation metrics through an interactive interface.

## 🎯 Objectives

- Analyze historical airline flight delay data.
- Perform data cleaning and exploratory data analysis (EDA).
- Apply statistical inference techniques to understand flight delays.
- Compare regression models using evaluation metrics.
- Predict estimated arrival delays using a trained machine learning model.
- Develop an interactive web interface using Streamlit.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib, Seaborn, and Streamlit charts
- **Dataset Source:** Bureau of Transportation Statistics (BTS)

## 📊 Machine Learning Models

The project compares the following regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regression
5. Random Forest Regression

The models are evaluated using:

| Metric | Purpose |
|---|---|
| MAE | Measures the average absolute prediction error. |
| MSE | Measures the average squared prediction error. |
| RMSE | Measures prediction error in the target's units. |
| R² Score | Measures how much variation in the target is explained by the model. |

### Best Model

Based on the model comparison results, **Linear Regression** achieved the lowest RMSE of approximately **15.4997**.

Its R² score was approximately **0.9550**, indicating that the model explained about 95.5% of the variation in the target on the evaluated dataset. This is not the same as 95.5% prediction accuracy.

## 🌐 Streamlit Web Application

The application provides:

- A professional dark-themed user interface.
- Flight input fields for airline, departure delay, flight distance, and date.
- Estimated arrival delay prediction using a saved model.
- An estimated flight status based on the predicted delay.
- A complete regression model performance table.
- Key performance indicators for RMSE, MAE, and R².
- Model comparison results for evaluating algorithm performance.

**Note:** Predictions are generated from the saved model and are not live airline flight-status updates.

## 📁 Project Structure

```text
Airline-Flight-Delay-Analysis-Prediction/
│
├── app.py
├── best_flight_delay_regression_model.pkl
├── regression_model_comparison.csv
├── cleaned_airline_flight_delays.csv
├── multiple_regression_predictions.csv
├── requirements.txt
└── README.md
```

The cleaned dataset and prediction-results CSV files are optional for the basic prediction interface. Include them only if they are part of your project.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install dependencies

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📦 Requirements

Create a file named `requirements.txt` containing:

```text
streamlit
pandas
numpy
joblib
scikit-learn==1.6.1
```

Use the same compatible scikit-learn version used to train and save the model.

## 📈 Results

The regression models were compared using MAE, MSE, RMSE, and R² score. Based on the saved comparison results, Linear Regression performed best in terms of RMSE.

The Streamlit application makes it easier to explore model performance and generate estimated arrival delays from user-provided flight information.

## 🔮 Future Enhancements

- Integrate live flight-status APIs.
- Include weather and airport congestion data.
- Improve predictions using additional flight features.
- Add interactive visualizations for historical delay trends.
- Deploy the application online for public access.

## ⚠️ Limitations

- Prediction quality depends on the training dataset and model.
- The application does not currently retrieve live airline information.
- Predictions are estimates and may differ from actual arrival delays.
- The saved model must be compatible with the application's input features and preprocessing.

## 👩‍💻 Project Information

**Project Title:** Airline Flight Delay Analysis & Prediction  
**Subject:** Statistical Inference Lab  
**Application:** Streamlit-based Machine Learning Interface

---

⭐ If you find this project useful, consider giving the repository a star.
