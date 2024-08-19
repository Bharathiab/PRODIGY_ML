import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('C:\\Users\\S.Bharathi\\Downloads\\kc_house_data.csv')

# Drop columns that are not needed for the model
data = data.drop(columns=['id', 'date'])

# One-hot encode categorical features (e.g., 'zipcode')
data = pd.get_dummies(data, columns=['zipcode'])

# Splitting the data into features and target variable
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Predicting the price of a new house
new_data = {
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2000,
    'sqft_lot': 5000,
    'floors': 2,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'grade': 7,
    'sqft_above': 2000,
    'sqft_basement': 0,
    'yr_built': 1990,
    'yr_renovated': 0,
    'lat': 47.5112,
    'long': -122.257,
    'sqft_living15': 2000,
    'sqft_lot15': 5000,
    # include the appropriate zipcode dummy variables
}

# Convert the new data point to DataFrame
new_data_df = pd.DataFrame([new_data])

# Align new_data_df with X_train to ensure it has the same dummy variables
new_data_df = new_data_df.reindex(columns=X_train.columns, fill_value=0)

# Scale the new data point
new_data_scaled = scaler.transform(new_data_df)

# Predict the price
predicted_price = model.predict(new_data_scaled)
print(f'Predicted Price: ${predicted_price[0]:.2f}')

# Plotting actual vs. predicted prices
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
