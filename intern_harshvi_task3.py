
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("Housing.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())



df = df.replace({'yes': 1, 'no': 0})


if 'furnishingstatus' in df.columns:
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)



print("\n--- SIMPLE LINEAR REGRESSION ---")


X = df[['area']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Simple Linear Regression (Area vs Price)")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)




print("\n--- MULTIPLE LINEAR REGRESSION ---")


X = df.drop('price', axis=1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model2 = LinearRegression()
model2.fit(X_train, y_train)


y_pred2 = model2.predict(X_test)


print("MAE:", mean_absolute_error(y_test, y_pred2))
print("MSE:", mean_squared_error(y_test, y_pred2))
print("R2 Score:", r2_score(y_test, y_pred2))


print("Intercept:", model2.intercept_)
print("Coefficients:", model2.coef_)



print("\nExample Prediction:")
print(model2.predict([X_test.iloc[0]]))