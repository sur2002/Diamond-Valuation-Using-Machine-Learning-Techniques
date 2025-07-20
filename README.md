💎 Predicting Diamond Prices Using Machine Learning


This project focuses on building a machine learning model to accurately predict the price of diamonds based on various physical and quality-related attributes.
Using regression techniques, the project aims to model real-world pricing patterns in the diamond market.

**Dataset**
Source : kaggle
Size : 193,573 rows × 11 columns
Features: 
  -Numerical: carat, depth, table, x, y, z
  -Categorical: cut, color, clarity
  -Target: price (in USD)
  
**Objective**
To build and evaluate machine learning models that can predict diamond prices based on their features using regression analysis.

**Exploratory Data Analysis**

- Checked for missing and duplicate records (none found).
- Visualized distributions of numerical features.
- Analyzed correlations between variables.
  - **Carat** and dimensions (x, y, z) showed strong correlation with **price**.
  - **Depth** and **table** had weak correlation.

**Data Preprocessing**

- Dropped non-informative `id` column.
- Encoded categorical features (`cut`, `color`, `clarity`) using ordinal mapping.
- Performed feature scaling using `StandardScaler`.

**Models Used**

1. Linear Regression
2. Random Forest Regressor

**Key Findings**

- "Carat" is the strongest predictor of diamond price.
- Larger physical dimensions (x, y, z) contribute heavily to price.
- Quality attributes like "cut", "clarity", and "color" also affect price, but less than size.
- Random Forest is better suited to model the non-linear relationships in the data.

**Tools & Libraries**

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn (train_test_split, LinearRegression, RandomForestRegressor, metrics)
- Jupyter Notebook / IDE for development
