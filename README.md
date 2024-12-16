# Traffic Stop Dataset Analysis and Prediction

This project aims to analyze traffic stop data and build a machine learning model to predict whether a driver will be arrested based on various features such as gender, age, race, violation type, and other factors. The project involves data preprocessing, exploratory data analysis (EDA), and building a predictive model using a Random Forest classifier.

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Steps to Run the Project](#steps-to-run-the-project)
5. [Model Explanation](#model-explanation)
6. [Evaluation Metrics](#evaluation-metrics)
7. [License](#license)

---

## Overview

The goal of this project is to predict if a driver will be arrested during a traffic stop. The dataset contains various attributes like:
- Driver’s gender, age, race
- Type of violation
- Whether a search was conducted
- Duration of the stop

The steps involved in this project are:
1. Data Preprocessing: Clean the dataset by handling missing values, converting categorical variables to numerical values using one-hot encoding, and performing necessary transformations.
2. Exploratory Data Analysis (EDA): Visualize and analyze the dataset to gain insights into the relationship between features and arrest outcomes.
3. Model Building: Use a Random Forest classifier to predict whether a driver will be arrested based on the features.
4. Evaluation: Evaluate the model performance using metrics like accuracy, precision, recall, and F1-score.

---

## Dependencies

Make sure you have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Dataset

The dataset contains traffic stop information with the following columns:
- `stop_date`: Date of the stop
- `stop_time`: Time of the stop
- `county_name`: Name of the county
- `driver_gender`: Gender of the driver
- `driver_age_raw`: Age of the driver (raw)
- `driver_age`: Age of the driver (processed)
- `driver_race`: Race of the driver
- `violation_raw`: Raw violation type
- `violation`: Processed violation type
- `search_conducted`: Whether a search was conducted during the stop
- `search_type`: Type of search conducted
- `stop_outcome`: Outcome of the stop
- `is_arrested`: Whether the driver was arrested
- `stop_duration`: Duration of the stop
- `drugs_related_stop`: Whether the stop was related to drugs

---

## Steps to Run the Project

1. Clone the Repository:
   ```bash
   git clone https://github.com/your-username/traffic-stop-prediction.git
   cd traffic-stop-prediction
   ```

2. Install Dependencies:
   Make sure you have all the necessary dependencies installed as mentioned in the "Dependencies" section.

3. Load the Dataset:
   The dataset should be a CSV file, e.g., `traffic_stop_data.csv`. Load it using pandas:
   ```python
   import pandas as pd
   df = pd.read_csv('traffic_stop_data.csv')
   ```

4. Preprocess the Data:
   Clean the dataset by handling missing values and one-hot encoding categorical variables.
   ```python
   df = df.drop(columns=['county_name'])  # Drop columns with too many missing values
   df['driver_gender'] = df['driver_gender'].fillna('Unknown')  # Fill missing gender data
   df = pd.get_dummies(df, columns=['driver_gender', 'driver_race', 'violation', 'search_type', 'stop_outcome'], drop_first=True)
   ```

5. Split the Data:
   Split the data into features (`X`) and target (`y`), then further split into training and testing sets.
   ```python
   from sklearn.model_selection import train_test_split
   X = df.drop(columns=['is_arrested', 'stop_date', 'stop_time'])  # Features
   y = df['is_arrested']  # Target variable
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

6. Train the Model:
   Train a Random Forest classifier to predict whether a driver will be arrested.
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```

7. Evaluate the Model:
   Evaluate the model's performance on the test data.
   ```python
   from sklearn.metrics import accuracy_score, classification_report
   y_pred = model.predict(X_test)
   print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
   print(classification_report(y_test, y_pred))
   ```

---

## Model Explanation

We are using a Random Forest Classifier for this project. A Random Forest is an ensemble learning method that creates multiple decision trees and aggregates their predictions. It is a powerful model for classification tasks and works well with tabular data like this one.

### Model Steps:
1. The Random Forest classifier constructs multiple decision trees by randomly selecting features and samples from the dataset.
2. Each tree makes a prediction, and the final prediction is based on the majority vote of all the trees.

---  

## Evaluation Metrics

After training the model, we evaluate its performance using the following metrics:
- Accuracy: The percentage of correctly predicted instances out of the total number of instances.
- Precision: The ratio of correctly predicted positive observations to the total predicted positives.
- Recall: The ratio of correctly predicted positive observations to all observations in the actual class.
- F1-Score: The weighted average of precision and recall.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
