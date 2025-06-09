import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('../data/Global_Pollution_Analysis.csv')

# Handle Missing Values
# Use SimpleImputer to fill missing values with the mean of the respective column
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Data Transformation
# Normalize pollution indices (e.g., air, water, and soil)
scaler = MinMaxScaler()
df_imputed[['Air_Pollution_Index', 'Water_Pollution_Index', 'Soil_Pollution_Index']] = scaler.fit_transform(df_imputed[['Air_Pollution_Index', 'Water_Pollution_Index', 'Soil_Pollution_Index']])

# Encode categorical columns
encoder = LabelEncoder()
df_imputed['Country'] = encoder.fit_transform(df_imputed['Country'])
df_imputed['Year'] = encoder.fit_transform(df_imputed['Year'])

df_imputed.head()
df_imputed.to_csv('../data/cleaned_pollution_data.csv', index=False)
# Global Pollution Analysis Project

## Description
This project analyzes global pollution data and builds predictive models to suggest strategies for pollution reduction and converting pollutants into energy.

## Phases
1. Data Collection and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Predictive Modeling

## Structure
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for data analysis and modeling.
- `src/`: Python scripts for functions and utilities.
import matplotlib.pyplot as plt
import seaborn as sns

# Descriptive Statistics
df_imputed.describe()

# Correlation Matrix
correlation = df_imputed.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Plotting Trends Over Time (Yearly Trends)
df_imputed.groupby('Year')['CO2_Emissions'].mean().plot(kind='line', title='CO2 Emissions Trend Over Time')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.show()

# Box Plot of Air Pollution Index by Country
plt.figure(figsize=(10,6))
sns.boxplot(x='Country', y='Air_Pollution_Index', data=df_imputed)
plt.xticks(rotation=90)
plt.title('Air Pollution Index Distribution Across Countries')
plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Prepare features and target variable
X = df_imputed[['Air_Pollution_Index', 'CO2_Emissions', 'Industrial_Waste_in_tons']]
y = df_imputed['Energy_Recovery_GWh']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create categories for pollution severity
df_imputed['Pollution_Severity'] = pd.cut(df_imputed['Air_Pollution_Index'], bins=3, labels=['Low', 'Medium', 'High'])

# Prepare features and target variable
X = df_imputed[['Air_Pollution_Index', 'CO2_Emissions']]
y = df_imputed['Pollution_Severity']

# Encode categorical target variable
y = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict on test data
y_pred = log_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
git add .
git commit -m "Initial commit with preprocessing, EDA, and modeling"
git push origin main
Global-Pollution-Analysis/
│
├── data/
│   └── Global_Pollution_Analysis.csv
│   └── cleaned_pollution_data.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── linear_regression_model.ipynb
│   └── logistic_regression_model.ipynb
│
├── src/
│   └── (Optional code for utilities or functions)
│
├── README.md
└── .gitignore

