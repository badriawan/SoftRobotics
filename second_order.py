import pandas as pd
import numpy as np
import math
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

def overdamped_response(t, A, B, C, tau1, tau2):
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C

# Fit the overdamped model to the data
popt, _ = curve_fit(overdamped_response, x, response)

# Extract the parameters
A, B, C, tau1, tau2 = popt

# Use the model to predict the response at the desired time points
predicted_response = overdamped_response(x, A, B, C, tau1, tau2)
print(f'y(t) = {popt}')

# Calculate zeta and omega_n for the overdamped case
zeta = (tau1 + tau2) / (2 * math.sqrt(tau1 * tau2))
wn = 1 / math.sqrt(tau1 * tau2)

print(f"Damping Ratio: {zeta}")
print(f"Natural Frequency: {wn}")
print(f"transfer function: {wn**2} / (s^2 + {2*zeta*wn}s + {wn**2}) * {P}/s")

# # Create a DataFrame to store the predicted response
# predicted_df = pd.DataFrame({'datetime': time_strings, '2ndResponse': predicted_response}) # save data to csv

# # Save the predicted response to a CSV file
# predicted_df.to_csv('second_order014bar.csv', index=False)

# Plot the original data
plt.plot(x, df['Theta'], label='Original Data', alpha=0.5)

# Plot the smoothed data
plt.plot(x, predicted_response, label=f'Predicted Response')

# Add labels and legend
plt.title(f"Plot 2nd Order System Soft Robot {P} Bar")
plt.xlabel('Timestamp (s)')
plt.ylabel('Theta (rad)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()