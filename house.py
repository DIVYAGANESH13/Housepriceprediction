import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

filepath=(r"C:\Users\Jayap\OneDrive\Documents\Desktop\house\kc_house_data.csv")
df = pd.read_csv(filepath)
print(df.columns)
print(df.info())

# Define features (X) and target variable (y)
X = df[['bedrooms',	'bathrooms',	'sqft_living'	,'sqft_lot',	'floors','waterfront',	'view',	'condition',	'grade','sqft_above',	'sqft_basement'	,'yr_built','yr_renovated',	'zipcode',	'lat',	'long'	,'sqft_living15'	,'sqft_lot15'
]]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often beneficial)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Now, you can use the model to predict the price of a new house
new_house_features = [[3, 2, 1800, 4000, 2, 0, 1, 4, 8, 1600, 200, 2000, 0, 98102, 47.6, -122.2, 1700, 3500]]

scaled_features = scaler.transform(new_house_features)
predicted_price = model.predict(scaled_features)

print(f'Predicted Price for the New House: {predicted_price[0]}')
import matplotlib.pyplot as plt

# Calculate residuals
residuals = y_test - predictions

# Plot residuals
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
