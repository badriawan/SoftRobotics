import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define first-order transfer function
def first_order_response(t, K, tau, y0):
    return K * (1 - np.exp(-t/tau)) + y0

def second_order_response(t, A, B, C, tau1, tau2):
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C

pressure = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
theta_ss = np.array([0.41648307, 1.09544303, 3.12238355, 3.03638361, 3.07043271, 3.16863119])

# Generate a range of x values for plotting
x_range = np.linspace(min(pressure), max(pressure), 100)

fit_params_1, pcov = curve_fit(first_order_response, pressure, theta_ss, bounds=([0, 0, 0], [np.inf, np.inf, 1]))
fit_params_2, _ = curve_fit(second_order_response, pressure, theta_ss)

# Parameters of the first-order model
K, tau, y0 = fit_params_1
A, B, C, tau1, tau2 = fit_params_2

# Predict response for a new time point
fitted_function_1 = first_order_response(pressure, K, tau, y0)
fitted_function_2 = second_order_response(pressure, A, B, C, tau1, tau2)
print(f'Exponential Function Order 1 = {fit_params_1}')
print(f'Exponential Function Order 2 = {fit_params_2}')
print(f'Theta_1 = {fitted_function_1}')
print(f'Theta_2 = {fitted_function_2}')

print(f"Parameters of the FOS Soft Robot:")
print("Gain (K):", K)
print("Time constant (tau):", tau)
print("Initial value (y0):", y0)

# Plot data points
plt.scatter(pressure, theta_ss, label='Data')

# Plot the fitted exponential functions
plt.plot(pressure, fitted_function_1, label='First Order System')
plt.plot(pressure, fitted_function_2, label='Second Order System')
# plt.plot(x_range, y_fit_3, label='Exponential 3')
    
plt.title('Exponential Curve Fitting')
plt.xlabel('Pressure (bar)')
plt.ylabel('Theta Steady-State (rad)')
plt.legend()
plt.grid(True)
plt.show()