# Load Dependencies


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

# Read the data


```python
df_train =  pd.read_csv('../../DATA/kaggle/titanic/train.csv')
df_test = pd.read_csv('../../DATA/kaggle/titanic/test.csv')
#df_train.head()
#df_test.head()

#df_train.describe()
#df_train.info()
#df_train.value_counts()
#df_train.shape
#df_train.info()
df_train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



## Remove some column


```python
df_train.drop(columns=['Cabin','Name','Ticket','PassengerId'],axis=1,inplace=True)
df_test.drop(columns=['Cabin','Name','Ticket','PassengerId'],axis=1,inplace=True)
```


```python
df_test.isnull().sum()
```




    Pclass       0
    Sex          0
    Age         86
    SibSp        0
    Parch        0
    Fare         1
    Embarked     0
    dtype: int64



# Fill some missing data


```python
df_train['Age'].fillna(df_train['Age'].median(),inplace=True)

df_train = df_train.dropna()

df_test['Age'].fillna(df_test['Age'].median(),inplace=True)
df_test['Fare'].fillna(df_test['Fare'].mean(),inplace=True)
df_test.isnull().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    dtype: int64



# Prepare Training Data


```python
#df_train.head()
X = df_train.drop(columns=['Survived'],axis=1)

Y = df_train['Survived']
#Y.head()
X.head()
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
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# Encode some data


```python
from sklearn.preprocessing import LabelEncoder 
le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

le_embarked = LabelEncoder()
X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
X.head()
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
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Build Model


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,Y)  
```




    LogisticRegression()




```python
from sklearn.model_selection import GridSearchCV
param = {
         'penalty':['l1','l2'],
         'C':[0.001, 0.01, 0.1, 1, 10, 20,100, 1000]
}
lr= LogisticRegression(penalty='l1')
gs=GridSearchCV(log_reg,param,cv=5,n_jobs=-1)
gs.fit(X,Y)

```

    c:\users\tro\.conda\envs\ml\lib\site-packages\sklearn\model_selection\_search.py:922: UserWarning: One or more of the test scores are non-finite: [       nan 0.68056243        nan 0.73345395        nan 0.79082714
            nan 0.78743731        nan 0.78630737        nan 0.78631372
            nan 0.78518377        nan 0.78518377]
      warnings.warn(
    




    GridSearchCV(cv=5, estimator=LogisticRegression(), n_jobs=-1,
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 20, 100, 1000],
                             'penalty': ['l1', 'l2']})




```python

X_dummy = np.array([[3,"male",22,1,0,7.25,"S"]])
X_dummy[:,1] = le_sex.transform(X_dummy[:,1])
X_dummy[:,6] = le_embarked.transform(X_dummy[:,6])
gs.predict(X_dummy)
```

    c:\users\tro\.conda\envs\ml\lib\site-packages\sklearn\utils\validation.py:63: FutureWarning: Arrays of bytes/strings is being converted to decimal numbers if dtype='numeric'. This behavior is deprecated in 0.24 and will be removed in 1.1 (renaming of 0.26). Please convert your data to numeric values explicitly instead.
      return f(*args, **kwargs)
    




    array([0], dtype=int64)



# Working on test data


```python
df_test.head()
X_test = df_test.to_numpy()
X_test[:,1] = le_sex.transform(X_test[:,1])
X_test[:,6] = le_embarked.transform(X_test[:,6])
X_test[0,:]
```




    array([3, 1, 34.5, 0, 0, 7.8292, 1], dtype=object)




```python
best_model = gs.best_estimator_
best_model
```




    LogisticRegression(C=0.1)




```python
pred_t = best_model.predict(X_test)
pred_t.shape
```




    (418,)




```python
p = pd.DataFrame(pred_t)
templ = pd.read_csv('../../DATA/kaggle/titanic/everyone_dies.csv')
templ['Survived'] = p
templ.to_csv('my_submission.csv', index = False)
```


```python

```
