import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the polynomial function
def polynomial_func(x, *coefficients):
    y = 0
    for i, coef in enumerate(coefficients):
        y += coef * x**i
    return y

pressure = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
theta_ss = np.array([0.41648307, 1.09544303, 3.12238355, 3.03638361, 3.07043271, 3.16863119])

# Generate a range of x values for plotting
x_range = np.linspace(min(pressure), max(pressure), 100)

# List of polynomial orders to consider
polynomial_orders = [1, 2, 3, 4, 5]  # You can extend this list to include higher orders

# Create a single plot for all polynomial fits
plt.figure(figsize=(10, 6))  # Set the figure size

# Plot data points
plt.scatter(pressure, theta_ss, label='Data')

# Fit and plot polynomial functions for each order
for order in polynomial_orders:
    initial_guess = [1.0] * (order + 1)  # Initial guess for coefficients
    
    # Perform polynomial regression using curve_fit
    fit_params, _ = curve_fit(polynomial_func, pressure, theta_ss, p0=initial_guess)
    
    # Extract the fitted coefficients
    fitted_coefficients = fit_params
    
    # Create the fitted polynomial function
    fitted_function = lambda x: polynomial_func(x, *fitted_coefficients)
    print(f'polynomial Function Order {order}: {fitted_coefficients}')
    
    # Generate y values for the fitted curve
    y_fit = fitted_function(x_range)
    y_theta = fitted_function(pressure)
    print(f'theta_poly_{order}: {y_theta}')
    
    # Plot the fitted polynomial curve for the current order
    plt.plot(x_range, y_fit, label=f'Order {order}')
    
plt.title('Polynomial Curve Fitting')
plt.xlabel('Pressure (bar)')
plt.ylabel('Theta Steady-State (rad)')
plt.legend()
plt.grid(True)
plt.show()