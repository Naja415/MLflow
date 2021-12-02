# # Assignment 3

#!pip install mlflow

#imports 
import pandas as pd
import sys

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler 
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ### Custom transformer
#create custom transformer 
#https://wkirgsn.github.io/2018/02/15/pandas-pipelines/
class DFCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.dropna()
        X = X[['Speed','Direction','Total']]
        return X

degree = int(sys.argv[1]) if len(sys.argv) > 1 else 3
splits = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
if (degree < 0 | degree > 10):
        raise ValueError("must be between 0 and 10, inclusive")

if (splits < 2 | splits > 7):
    raise ValueError("must be between 2 and 7, inclusive")
    
#read in dataset and drop na rows
df = pd.read_json("dataset.json", orient ="split")
preproccesing = make_pipeline(DFCleaner())
df = preproccesing.fit_transform(df)
    
pipeline = ColumnTransformer(
    transformers=[
        ('imputer', SimpleImputer(strategy="median"),['Speed']),
        ('std_scaler', StandardScaler(with_mean=True, with_std=True),['Speed']),
        ("cat", OneHotEncoder(), ["Direction"])
    ]
)

#add model to pipeline
pipeline = Pipeline([
    ("pre", pipeline),
    ("pol",PolynomialFeatures(degree=degree)),
    ("K",KNeighborsRegressor()),
])
    
#Metrics
metrics = [
    ("MSE", mean_squared_error, []),
    ("MAE", mean_absolute_error , []),
    ("R2", r2_score, [])
]

X = df[['Speed',"Direction"]]
y = df["Total"]

number_of_splits = splits

#run the K-fold splits
for train, test in TimeSeriesSplit(number_of_splits).split(X, y):
    pipeline.fit(X.iloc[train], y.iloc[train])
    predictions = pipeline.predict(X.iloc[test])
    truth = y.iloc[test]
    
    # Calculate and save the metrics for this fold
    for name , func , scores in metrics:
        score = func(truth,predictions)
        scores.append(score)

#Log a summary of the metrics
for name , _ , scores in metrics:

    mean_score = sum(scores) / number_of_splits
    print(f"mean_{name}" +": "+str(mean_score.round(2)))

    max_score = max(scores)
    print(f"max_{name}" +": "+str(max_score.round(2)))

    min_score = min(scores)
    print(f"min_{name}" +": "+str(min_score.round(2)))

    print("\n")

