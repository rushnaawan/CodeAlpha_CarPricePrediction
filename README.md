# Car Price Prediction

Task 3 of my CodeAlpha Data Science internship.

Regression model that predicts the selling price of a used car from features like age, kilometres driven, fuel type and transmission.

## How to run
The dataset is `car data.csv` (included in this folder, 301 rows, prices are in lakhs). The script predicts `Selling_Price`.

    pip install -r requirements.txt
    python car_price_prediction.py

Plots and results are saved in the outputs folder.

## What the script does
- Loads and checks the data (no missing values, 2 duplicate rows which get dropped)
- Turns the Year column into car age
- Drops the Car_Name column because it has 98 different values
- One-hot encodes Fuel_Type, Selling_type and Transmission
- Splits into 80% train and 20% test
- Trains Linear Regression, Random Forest and Gradient Boosting
- Compares them with R2 (cross validation and test), MAE and RMSE
- Plots actual vs predicted prices and feature importance for the best model

## Results
| Model | CV R2 | Test R2 | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.885 | 0.753 | 1.47 | 2.53 |
| Random Forest | 0.896 | 0.515 | 1.48 | 3.54 |
| Gradient Boosting | 0.897 | 0.725 | 1.18 | 2.67 |

Gradient Boosting had the best cross validation score and the lowest MAE, so the script picks it. On average its prediction is off by about 1.2 lakh.

The test R2 is lower than the CV R2 for all models. The test set has a Fortuner sold at 33 lakh, which is far from the rest of the data (most cars are under 10 lakh), and the models miss it by a lot. That is why RMSE is much bigger than MAE, and why Random Forest drops to 0.515. With only 299 rows, the test score changes a lot depending on which cars land in the test set.

Feature importance (Gradient Boosting): Present_Price is by far the most important (about 90%), then car age (about 7%). Kms driven, fuel type, transmission and owner count matter very little in this data.
