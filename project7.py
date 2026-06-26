import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("house_data.csv")

# Input Features
X = df[['Area', 'Bedrooms']]

# Target Variable
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error Calculation
mae = mean_absolute_error(y_test, predictions)

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(predictions)

print("\nMean Absolute Error:")
print(round(mae, 2))

# Predict Price for New House
area = 1500
bedrooms = 3

new_data = pd.DataFrame({
    'Area': [area],
    'Bedrooms': [bedrooms]
})

new_prediction = model.predict(new_data)

print(
    f"\nPredicted Price for a house with "
    f"{area} sq.ft and {bedrooms} bedrooms:"
)
print(f"₹ {round(new_prediction[0], 2)}")