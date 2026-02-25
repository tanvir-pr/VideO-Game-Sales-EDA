
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("vgsales.csv")
df.head()

df.isnull().sum()
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])

df = df.drop(columns=["Rank"])

df = pd.get_dummies(df, columns=["Platform", "Genre", "Publisher"], drop_first=True)


df["Global_Sales"] = (
    df["NA_Sales"] +
    df["EU_Sales"] +
    df["JP_Sales"] +
    df["Other_Sales"]
)


Q1 = df["Global_Sales"].quantile(0.25)
Q3 = df["Global_Sales"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Global_Sales"] >= lower) & (df["Global_Sales"] <= upper)]


Q1 = df["Global_Sales"].quantile(0.25)
Q3 = df["Global_Sales"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Global_Sales"] >= lower) & (df["Global_Sales"] <= upper)]

numeric_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]



#   task 3



df = df.drop(columns=["Name"])

# Define target and features
X = df.drop("Global_Sales", axis=1)
y = df["Global_Sales"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)



pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)





#task 5

pipeline.fit(X_train, y_train)

#tassk 6

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,                
    scoring="r2"         
)

#task 7

param_grid = {
    "model__fit_intercept": [True, False],
    "model__positive": [True, False]
}


grid_search = GridSearchCV(
    estimator=pipeline,     
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)


grid_search.fit(X, y)


results = pd.DataFrame(grid_search.cv_results_)

print("Tested Hyperparameters and Mean CV Scores:")
print(results[[
    "params",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]])

print("\nBest Parameters Found:")
print(grid_search.best_params_)

print("\nBest Cross-Validation R² Score:")
print(grid_search.best_score_)



#task 8

grid_search.best_params_
grid_search.best_score_


#task 9

best_pipeline = grid_search.best_estimator_
y_test_pred = best_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Print results
print("Model Performance on Test Set")
print("-----------------------------")
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)




with open("pipeline.pkl","wb") as f:
    pickle.dump(best_pipeline,f)