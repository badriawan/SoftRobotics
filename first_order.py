import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt

# Load the Excel file
P = 0.14
df = pd.read_csv('soft_robot_014bar.csv')  # Replace with your file path

# Assuming your Excel file has columns 'datetime' and 'response'
time_strings = df['datetime']
response = df['Theta'].values

# Convert time strings to datetime objects
# Adjust the format string to '%M:%S.%f' to correctly parse minutes and seconds
time_objects = [datetime.strptime(time_str, '%M:%S.%f') for time_str in time_strings]

# Calculate time differences and convert to seconds
base_time = time_objects[0]
x = np.array([(time - base_time).total_seconds() for time in time_objects])

# Define first-order transfer function
def first_order_response(t, K, tau, y0):
    return K * (1 - np.exp(-t/tau)) + y0

# Fit the first-order model to the data
popt, pcov = curve_fit(first_order_response, x, response, bounds=([0, 0, 0], [np.inf, np.inf, 1]))

# Parameters of the first-order model
K, tau, y0 = popt

# Predict response for a new time point
predicted_output = first_order_response(x, K, tau, y0)

# # Create a DataFrame to store the predicted response
# predicted_df = pd.DataFrame({'datetime': time_strings, '1stResponse': predicted_output}) # save data to csv

# # Save the predicted response to a CSV file
# predicted_df.to_csv('first_order004bar.csv', index=False)

# Plot original data and fitted curve
plt.plot(x, df['Theta'], label='Original Data', alpha=0.5)
plt.plot(x, first_order_response(x, *popt), label='Fitted Curve')
plt.xlabel('Timestamp (s)')
plt.ylabel('Theta (rad)')
plt.title(f"Plot 1st Order System Soft Robot {P} Bar")
plt.grid(True)
plt.legend()
plt.show()

print(f"Parameters of the FOS Soft Robot {P} Bar:")
print("Gain (K):", K)
print("Time constant (tau):", tau)
print("Initial value (y0):", y0)
