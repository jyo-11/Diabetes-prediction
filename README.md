# Diabetes Prediction using Machine Learning

This project implements a machine learning model to predict the likelihood of diabetes based on various health metrics. It utilizes the Support Vector Machine (SVM) algorithm for classification.

## Introduction
Diabetes is a chronic health condition that affects millions worldwide. Early detection and diagnosis are crucial for effective management. This project aims to develop a predictive model that can assist in identifying individuals at risk of diabetes.

## Dataset
The dataset used in this project is the "Diabetes Dataset," stored in a CSV file named `diabetes.csv`. It contains various features related to health measurements:

### Features (Independent Variables):
    *  Pregnancies
    *   Glucose
    *   BloodPressure
    *   SkinThickness
    *   Insulin
    *   BMI
    *   DiabetesPedigreeFunction
    *   Age
### Target Variable (Dependent Variable):
    *   Outcome (0 for non-diabetic, 1 for diabetic)

The dataset used in this project is available in this repository and here is the link to it: [diabetes.csv](diabetes.csv)
  

## Methodology
The project follows these steps:

1.  **Data Loading and Exploration:** The dataset is loaded using pandas, and initial exploration is performed.
2.  **Data Preprocessing:**
    *   Features (X) and labels (Y) are separated.
    *   Data standardization is applied to the features using `StandardScaler`.
3.  **Train-Test Split:** The data is split into training (80%) and testing (20%) sets using `train_test_split` with stratified sampling.
4.  **Model Training:** An SVM classifier with a linear kernel is trained on the training data.
5.  **Model Evaluation:** The model's performance is evaluated using accuracy scores on both training and testing sets.
6.  **Prediction:** A function is implemented to make predictions on new input data.

## Requirements
*   Python 3
*   NumPy
*   Pandas
*   Scikit-learn

## Installation
pip install numpy pandas scikit-learn

## Usage
1.  Clone the repository: `git clone [repository URL]` (If you are using Git)
2.  Navigate to the project directory: `cd diabetes-prediction`
3.  Place the 'diabetes.csv' file in the project directory.
4.  Run the script: `python diabetes_prediction.py` (or the name of your Python script).

## Model Evaluation
The script outputs the accuracy scores for both the training and testing datasets.  Look for output similar to:

Accuracy score of the training data: 0.7833333333333333
Accuracy score of the testing data: 0.7272727272727273

## Prediction
The script includes an example of making predictions on new data. Modify the `input_data` variable in the script with the features of a new individual.  The output will indicate the prediction (diabetic or non-diabetic).  For example:

### example input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  

The **output** will be:
The person is diabetic

## Code Explanation

Key parts of the code include:

  * **Data Standardization:**  `StandardScaler` is used to standardize the features, ensuring that all features have a similar scale. This is important for SVM performance.
  * **SVM Model:** An SVM classifier with a linear kernel is used.  The `kernel='linear'` argument specifies the linear kernel.
  * **Train/Test Split:** The `train_test_split` function from scikit-learn is used to divide the dataset into training and testing sets.  The `stratify` parameter is used to ensure that the class distribution in the training and testing sets is similar to the original dataset.
  * **Prediction:** The trained SVM model's `predict()` method is used to make predictions on new data.


## License
This project is licensed under the [MIT License](https://www.google.com/url?sa=E&source=gmail&q=https://opensource.org/licenses/MIT) (or specify your chosen license).


