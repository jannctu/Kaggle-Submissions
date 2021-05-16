# Import dependencies


```python
import numpy as np
import pandas as pd 
```


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(0)
```

# Read the dataset


```python
train_df = pd.read_csv("../../Data/kaggle/commonlit-readability/train.csv")
test_df = pd.read_csv("../../Data/kaggle/commonlit-readability/test.csv")
train_df.head()
#test_df.head()
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
      <th>url_legal</th>
      <th>license</th>
      <th>excerpt</th>
      <th>target</th>
      <th>standard_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c12129c31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>When the young people returned to the ballroom...</td>
      <td>-0.340259</td>
      <td>0.464009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85aa80a4c</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All through dinner time, Mrs. Fayre was somewh...</td>
      <td>-0.315372</td>
      <td>0.480805</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b69ac6792</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>As Roger had predicted, the snow departed as q...</td>
      <td>-0.580118</td>
      <td>0.476676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dd1000b26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>And outside before the palace a great garden w...</td>
      <td>-1.054013</td>
      <td>0.450007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37c1b32fb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Once upon a time there were Three Bears who li...</td>
      <td>0.247197</td>
      <td>0.510845</td>
    </tr>
  </tbody>
</table>
</div>



# Inspect the data


```python
train_df.nunique()
```




    id                2834
    url_legal          667
    license             15
    excerpt           2834
    target            2834
    standard_error    2834
    dtype: int64



# Preprocessing


```python
train_df = train_df.drop(columns=['id','url_legal','license'])
train_df.head()
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
      <th>excerpt</th>
      <th>target</th>
      <th>standard_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>When the young people returned to the ballroom...</td>
      <td>-0.340259</td>
      <td>0.464009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All through dinner time, Mrs. Fayre was somewh...</td>
      <td>-0.315372</td>
      <td>0.480805</td>
    </tr>
    <tr>
      <th>2</th>
      <td>As Roger had predicted, the snow departed as q...</td>
      <td>-0.580118</td>
      <td>0.476676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And outside before the palace a great garden w...</td>
      <td>-1.054013</td>
      <td>0.450007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Once upon a time there were Three Bears who li...</td>
      <td>0.247197</td>
      <td>0.510845</td>
    </tr>
  </tbody>
</table>
</div>



## Convert text into lower case


```python
excerpts = train_df["excerpt"]
target = train_df["target"]

excerpts.str.lower()
excerpts.head()
```




    0    When the young people returned to the ballroom...
    1    All through dinner time, Mrs. Fayre was somewh...
    2    As Roger had predicted, the snow departed as q...
    3    And outside before the palace a great garden w...
    4    Once upon a time there were Three Bears who li...
    Name: excerpt, dtype: object



## stemming


```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
excerpts = excerpts.apply(ps.stem)
excerpts.head()
```




    0    when the young people returned to the ballroom...
    1    all through dinner time, mrs. fayre was somewh...
    2    as roger had predicted, the snow departed as q...
    3    and outside before the palace a great garden w...
    4    once upon a time there were three bears who li...
    Name: excerpt, dtype: object



## Lemmatization


```python
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
excerpts = excerpts.apply(wnl.lemmatize)
```

## Removing stopwords


```python
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
excerpts = excerpts.apply(lambda text: remove_stopwords(text))
```

## Train-test split


```python
excerpts_train, excerpts_val, y_train, y_val = train_test_split(excerpts, target, test_size=0.2)
excerpts_train.head()
```




    405     hippocrates (c. 460 – c. 370 bc) greek doctor ...
    836     long ago, clever cat foolish dog. clever cat l...
    57      cry long, however, brave could expected prince...
    1775    produce electric current, needed lower suspend...
    2525    mun bun disobedient little boy; daddy bunker s...
    Name: excerpt, dtype: object




```python
vectorizer = CountVectorizer()
vectorizer.fit(excerpts_train)
```




    CountVectorizer()




```python
X_train = vectorizer.transform(excerpts_train)
X_val = vectorizer.transform(excerpts_val)
#print(X_train)
```

# Training

## Random Forest


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import numpy as np 

regressor = RandomForestRegressor(random_state=42,n_estimators=100,max_depth=100)
regressor.fit(X_train,y_train)
```




    RandomForestRegressor(max_depth=100, random_state=42)




```python

y_pred = regressor.predict(X_train)
error = np.sqrt(mean_squared_error(y_train,y_pred))
print("{:,.02f}".format(error))
```

    0.32
    

# Evaluate 


```python
train_pred = regressor.predict(X_train)
val_pred = regressor.predict(X_val)
rmse_train = np.sqrt(mean_squared_error(y_train,train_pred))
rmse_val = np.sqrt(mean_squared_error(y_val,val_pred))
print("RMSE Train: {}".format(rmse_train))
print("RMSE Val: {}".format(rmse_val))
```

    RMSE Train: 0.3174112784946647
    RMSE Val: 0.8325789183586854
    

# Working on Testing data


```python
test_excerpts = test_df["excerpt"]
```


```python
test_excerpts = test_excerpts.str.lower()
test_excerpts = test_excerpts.apply(ps.stem)
test_excerpts = test_excerpts.apply(wnl.lemmatize)
test_excerpts = test_excerpts.apply(lambda text: remove_stopwords(text))
```


```python
X_test = vectorizer.transform(test_excerpts)
```

# Predict


```python
test_preds = regressor.predict(X_test)
```

# Export to CSV


```python
x_sub = test_df[["id"]].copy()
x_sub["target"] = test_preds
x_sub.to_csv('submission.csv', index = False)
x_sub
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
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c0f722661</td>
      <td>-1.036358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f0953f0a5</td>
      <td>-0.365061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0df072751</td>
      <td>-0.289395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04caf4e0c</td>
      <td>-1.636836</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0e63f8bea</td>
      <td>-1.435692</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12537fe78</td>
      <td>-0.851620</td>
    </tr>
    <tr>
      <th>6</th>
      <td>965e592c0</td>
      <td>-0.999787</td>
    </tr>
  </tbody>
</table>
</div>


