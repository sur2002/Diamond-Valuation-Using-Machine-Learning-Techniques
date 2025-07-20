```python
# import Libraries
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
```


```python
# Data ingestions step
df=pd.read_csv("gemstone.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.52</td>
      <td>Premium</td>
      <td>F</td>
      <td>VS2</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>7.27</td>
      <td>7.33</td>
      <td>4.55</td>
      <td>13619</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.03</td>
      <td>Very Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>8.06</td>
      <td>8.12</td>
      <td>5.05</td>
      <td>13387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.70</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.2</td>
      <td>57.0</td>
      <td>5.69</td>
      <td>5.73</td>
      <td>3.50</td>
      <td>2772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.32</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4.38</td>
      <td>4.41</td>
      <td>2.71</td>
      <td>666</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.70</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS2</td>
      <td>62.6</td>
      <td>59.0</td>
      <td>7.65</td>
      <td>7.61</td>
      <td>4.77</td>
      <td>14453</td>
    </tr>
  </tbody>
</table>
</div>



showing features like carat, cut, color, clarity, dimensions, and price. price is the target variable or output 


```python
df.isnull().sum()
```




    id         0
    carat      0
    cut        0
    color      0
    clarity    0
    depth      0
    table      0
    x          0
    y          0
    z          0
    price      0
    dtype: int64



no missing value in data


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 193573 entries, 0 to 193572
    Data columns (total 11 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   id       193573 non-null  int64  
     1   carat    193573 non-null  float64
     2   cut      193573 non-null  object 
     3   color    193573 non-null  object 
     4   clarity  193573 non-null  object 
     5   depth    193573 non-null  float64
     6   table    193573 non-null  float64
     7   x        193573 non-null  float64
     8   y        193573 non-null  float64
     9   z        193573 non-null  float64
     10  price    193573 non-null  int64  
    dtypes: float64(6), int64(2), object(3)
    memory usage: 16.2+ MB
    

The dataset contains 193,573 entries with 11 columns, all of which are complete with no missing values.
It includes a mix of numerical (int64, float64) and categorical (object) features


```python
# drop id column
df=df.drop(labels=['id'],axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52</td>
      <td>Premium</td>
      <td>F</td>
      <td>VS2</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>7.27</td>
      <td>7.33</td>
      <td>4.55</td>
      <td>13619</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.03</td>
      <td>Very Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>8.06</td>
      <td>8.12</td>
      <td>5.05</td>
      <td>13387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.70</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.2</td>
      <td>57.0</td>
      <td>5.69</td>
      <td>5.73</td>
      <td>3.50</td>
      <td>2772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.32</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4.38</td>
      <td>4.41</td>
      <td>2.71</td>
      <td>666</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.70</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS2</td>
      <td>62.6</td>
      <td>59.0</td>
      <td>7.65</td>
      <td>7.61</td>
      <td>4.77</td>
      <td>14453</td>
    </tr>
  </tbody>
</table>
</div>


The column 'id' has been successfully dropped from the dataset as it does not contribute to price prediction.

```python
# check for duplicate record
df.duplicated().sum()
```




    0



The dataset contains no duplicate records, confirming that all 193,573 entries are unique.


```python
# separating  numerical and categorical columns

numerical_columns=df.columns[df.dtypes!='object']
categorical_columns=df.columns[df.dtypes=='object']
print("Numerical columns: ",numerical_columns)
print("Categorical columns: ",categorical_columns)
```

    Numerical columns:  Index(['carat', 'depth', 'table', 'x', 'y', 'z', 'price'], dtype='object')
    Categorical columns:  Index(['cut', 'color', 'clarity'], dtype='object')
    

The dataset has been successfully separating into numerical and categorical columns based on data types.


```python
df[categorical_columns].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>193573</td>
      <td>193573</td>
      <td>193573</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Ideal</td>
      <td>G</td>
      <td>SI1</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>92454</td>
      <td>44391</td>
      <td>53272</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['cut'].value_counts()
```




    cut
    Ideal        92454
    Premium      49910
    Very Good    37566
    Good         11622
    Fair          2021
    Name: count, dtype: int64




```python
df['clarity'].value_counts()
```




    clarity
    SI1     53272
    VS2     48027
    VS1     30669
    SI2     30484
    VVS2    15762
    VVS1    10628
    IF       4219
    I1        512
    Name: count, dtype: int64




```python
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
x=0
for i in numerical_columns:
    sns.histplot(data=df,x=i,kde=True)
    print('\n')
    plt.show() 
```

    
    
    


    
![png](output_16_1.png)
    


    
    
    


    
![png](output_16_3.png)
    


    
    
    


    
![png](output_16_5.png)
    


    
    
    


    
![png](output_16_7.png)
    


    
    
    


    
![png](output_16_9.png)
    


    
    
    


    
![png](output_16_11.png)
    


    
    
    


    
![png](output_16_13.png)
    


The visualizations show that most numerical features like carat, x, y, and z are right-skewed,
indicating a higher number of smaller diamonds with a few large ones. Features like depth and table are more normally distributed,
centered around standard diamond proportions.


```python
# coreelation
sns.heatmap(df[numerical_columns].corr(),annot=True)
```




    <Axes: >




    
![png](output_18_1.png)
    

The heatmap shows that price is strongly correlated with carat (0.94) and physical dimensions x, y, z (≈0.90),
indicating that larger diamonds tend to be more expensive.
features like depth and table have very weak correlations with price

```python
# we can drop x,y,z
# df.drop(labels=['x','y','z'],axis=1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52</td>
      <td>Premium</td>
      <td>F</td>
      <td>VS2</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>7.27</td>
      <td>7.33</td>
      <td>4.55</td>
      <td>13619</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.03</td>
      <td>Very Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>8.06</td>
      <td>8.12</td>
      <td>5.05</td>
      <td>13387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.70</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.2</td>
      <td>57.0</td>
      <td>5.69</td>
      <td>5.73</td>
      <td>3.50</td>
      <td>2772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.32</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VS1</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4.38</td>
      <td>4.41</td>
      <td>2.71</td>
      <td>666</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.70</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS2</td>
      <td>62.6</td>
      <td>59.0</td>
      <td>7.65</td>
      <td>7.61</td>
      <td>4.77</td>
      <td>14453</td>
    </tr>
  </tbody>
</table>
</div>



for thr For Domain Purpose we use this https://www.americangemsociety.org/ags-diamond-grading-system/ 


```python
df['cut'].unique()
```




    array(['Premium', 'Very Good', 'Ideal', 'Good', 'Fair'], dtype=object)




```python
cut_map={"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}
```


```python
df['clarity'].unique()
```




    array(['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1'],
          dtype=object)




```python
clarity_map = {"I1":1,"SI2":2 ,"SI1":3 ,"VS2":4 , "VS1":5 , "VVS2":6 , "VVS1":7 ,"IF":8}
```


```python
df['color'].unique()
```




    array(['F', 'J', 'G', 'E', 'D', 'H', 'I'], dtype=object)




```python
color_map = {"D":1 ,"E":2 ,"F":3 , "G":4 ,"H":5 , "I":6, "J":7}
```


```python
df['cut']=df['cut'].map(cut_map)
df['clarity'] = df['clarity'].map(clarity_map)
df['color'] = df['color'].map(color_map)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>7.27</td>
      <td>7.33</td>
      <td>4.55</td>
      <td>13619</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.03</td>
      <td>3</td>
      <td>7</td>
      <td>2</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>8.06</td>
      <td>8.12</td>
      <td>5.05</td>
      <td>13387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.70</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>61.2</td>
      <td>57.0</td>
      <td>5.69</td>
      <td>5.73</td>
      <td>3.50</td>
      <td>2772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.32</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>61.6</td>
      <td>56.0</td>
      <td>4.38</td>
      <td>4.41</td>
      <td>2.71</td>
      <td>666</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.70</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>62.6</td>
      <td>59.0</td>
      <td>7.65</td>
      <td>7.61</td>
      <td>4.77</td>
      <td>14453</td>
    </tr>
  </tbody>
</table>
</div>



The categorical features cut, color, and clarity have been successfully converted into numerical form using mapping dictionaries.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```


```python
#Train-Test Split
X = df.drop('price', axis=1)
y = df['price']
```

we separated the dataset into input features (X) and the target variable (y). 
The X dataset includes all the relevant attributes that influence diamond pricing, such as carat, cut, color, clarity, and physical dimensions.
The target variable y consists solely of the price column, which we aim to predict. 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

feature scaling was applied using StandardScaler to standardize the input features. Scaling transforms the data so that each feature has a mean of 0 and a standard deviation of 1. 
By scaling both the training and testing data, we ensure that all features contribute equally to the model’s learning process.


```python
#Model Training – Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
```


```python
a Linear Regression model was trained using the scaled training data. 
Linear Regression attempts to find the best-fitting straight line that predicts the target variable (price) based on the input features.
After training, the model was used to make predictions on the scaled test data. 
```


```python
#Model Training – Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Tree-based models don't need scaling
rf_preds = rf.predict(X_test)
```


```python
import warnings
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")
def evaluate_model(name, y_true, y_pred):   #Evaluation 
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} → RMSE: {rmse:.2f}, R² Score: {r2:.4f}")
```

custom evaluation function was defined to assess model performance using two key metrics: Root Mean Squared Error (RMSE) and R² Score.
RMSE measures the average difference between the predicted and actual prices — lower values indicate better accuracy. 
R² Score indicates how well the model explains the variance in the target variable — a value closer to 1 means better performance. 
This function allows consistent comparison between different models.


```python
evaluate_model("Linear Regression", y_test, lr_preds)
evaluate_model("Random Forest", y_test, rf_preds)
```

    Linear Regression → RMSE: 1006.60, R² Score: 0.9373
    Random Forest → RMSE: 607.13, R² Score: 0.9772
    

The evaluation results clearly show that the Random Forest Regressor significantly outperformed Linear Regression.
While Linear Regression explained about 93.7% of the variance in diamond prices with an RMSE of around 1006, 
the Random Forest model explained 97.7% of the variance with a much lower RMSE of 607. 
This indicates that Random Forest is more accurate and better suited for modeling the complex, non-linear relationships in the diamond dataset.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
