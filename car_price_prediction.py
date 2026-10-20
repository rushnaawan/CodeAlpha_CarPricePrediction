import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs("outputs", exist_ok=True)

# prices are in lakhs
df = pd.read_csv("car data.csv")
print(df.shape)
print(df.head())
print(df.isnull().sum())
print("duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
print(df.describe())

# car age is more useful than the year (latest car in data is 2018)
df["car_age"] = df["Year"].max() - df["Year"]
df = df.drop(columns="Year")

# car name has 98 different values, so drop it
df = df.drop(columns="Car_Name")

# fuel type, selling type and transmission are text columns
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns="Selling_Price")
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

rows = []
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rows.append((name, round(cv, 3), round(r2, 3), round(mae, 3), round(rmse, 3)))
    print(name, "cv r2:", round(cv, 3), "test r2:", round(r2, 3), "mae:", round(mae, 3), "rmse:", round(rmse, 3))

results = pd.DataFrame(rows, columns=["model", "cv_r2", "test_r2", "mae", "rmse"])
results.to_csv("outputs/results.csv", index=False)

# choose the best model using cv r2
# test r2 is low for some models because the test set has a few very expensive cars
best_name = results.sort_values("cv_r2", ascending=False).iloc[0]["model"]
best = models[best_name]
print("best model:", best_name)

pred = best.predict(X_test)
plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("actual price (lakhs)")
plt.ylabel("predicted price (lakhs)")
plt.title(best_name)
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# only tree models have feature importance
if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()
    print(imp)
    imp.plot(kind="barh")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()
