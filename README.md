# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2.Load Dataset: Import the dataset containing car prices along with relevant features.
3.Data Preprocessing: Manage missing data and select key features for the model, if required.
4.Split Data: Divide the dataset into training and testing subsets.
5.Train Model: Build a linear regression model and train it using the training data.
6.Make Predictions: Apply the model to predict outcomes for the test set.
7.Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8.Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9.Output Results: Present the predictions and evaluation metrics.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Load the dataset
df=pd.read_csv('CarPrice_Assignment.csv')
print(df.head())

#Select Features and target
X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df[['price']]

#Split data
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#Feature Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Train Model
model=LinearRegression()
model.fit(X_train_scaled, y_train)

#Predictions
y_pred=model.predict(X_test_scaled)

#Model Coefficients and metrics
#print("="*50)
print("Name: Jijo H Jebas")
print("Reg No: 212225040156")
print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:}:{coef:}")
print(f"{'Intercept':}: {model.intercept_:}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':}: {mean_squared_error(y_test, y_pred):>}")
print(f"{'RMSE':}: {np.sqrt(mean_squared_error(y_test, y_pred)):}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"{'R-squared':}:{r2_score(y_test, y_pred):}")
#print("="*50)

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()],[y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence  (Durbin-Watson)
residuals= y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistics: {dw_test:}",
      "\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

## Output:
<img width="1022" height="691" alt="image" src="https://github.com/user-attachments/assets/7a8b7711-eb4e-4b86-a26c-23e94ee273ea" />
<img width="1038" height="543" alt="image" src="https://github.com/user-attachments/assets/70e3c2e8-8317-4c3a-84c5-c6900ef8fe8f" />
<img width="1076" height="471" alt="image" src="https://github.com/user-attachments/assets/526f9fe2-f05d-48e5-88d7-0a84c2c402e2" />
<img width="1062" height="421" alt="image" src="https://github.com/user-attachments/assets/9fc1112d-3f58-4600-b7c1-68e15ca63dff" />






## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
