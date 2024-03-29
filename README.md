# AI-driven-Personalized-Home-Automation
Develop an AI-powered home automation system that learns user preferences and optimizes energy consumption, comfort, and security.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Mock dataset: hour of the day, outside temperature, and preferred inside temperature
data = np.array([
    [8, 10, 20],  # 8 AM, 10°C outside, user prefers 20°C inside
    [12, 15, 22], # Noon, 15°C outside, 22°C preferred inside
    # Add more data points
])
X = data[:, :2]  # Features: hour of the day and outside temperature
y = data[:, 2]   # Target: preferred inside temperature

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model to learn user preferences
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the preferred temperature based on hour of the day and outside temperature
def predict_preferred_temperature(hour, outside_temp):
    return model.predict([[hour, outside_temp]])[0]

# Example: Predict preferred temperature at 10 AM when it's 12°C outside
predicted_temp = predict_preferred_temperature(10, 12)
print(f"Predicted preferred temperature: {predicted_temp}°C")

# Here you can add optimization algorithms for energy consumption based on the predicted preferences

