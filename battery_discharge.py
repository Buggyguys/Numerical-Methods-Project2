import numpy as np
from scipy import interpolate

# Compute the numerical derivative at a specific point using different methods
# x => array of x values
# y => array of y values
# method => forward, backward, central
# target_x => the point where the derivative is complete 
# -> returns derivate value at target_x
# Find the index of the closest point to target_x
def numerical_derivative(x, y, method, target_x):
    idx = np.abs(np.array(x) - target_x).argmin()
    
    # If target_x is exactly at a data point
    if x[idx] == target_x:
        if method == 'forward' and idx < len(x) - 1:
            h = x[idx+1] - x[idx]
            return (y[idx+1] - y[idx]) / h
        elif method == 'backward' and idx > 0:
            h = x[idx] - x[idx-1]
            return (y[idx] - y[idx-1]) / h
        elif method == 'central' and idx > 0 and idx < len(x) - 1:
            h_forward = x[idx+1] - x[idx]
            h_backward = x[idx] - x[idx-1]
            # If mesh is uniform (h_forward = h_backward)
            if np.isclose(h_forward, h_backward):
                return (y[idx+1] - y[idx-1]) / (2 * h_forward)
            else:  # Non-uniform mesh
                return (y[idx+1] * h_backward - y[idx-1] * h_forward + y[idx] * (h_forward - h_backward)) / (h_forward * h_backward)
    
    # If target_x is between data points, interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    
    # Compute numerically using a small h
    h = 1e-6  # Small step for numerical derivative
    
    if method == 'forward':
        return (f(target_x + h) - f(target_x)) / h
    elif method == 'backward':
        return (f(target_x) - f(target_x - h)) / h
    elif method == 'central':
        return (f(target_x + h) - f(target_x - h)) / (2 * h)
    else:
        raise ValueError("Method must be 'forward', 'backward', or 'central'")

# Compute the definite integral using the Compositon Trapezoidal Rule 
# x => array of x values
# y => array of y values
# -> returns integral value 
def trapezoidal_rule(x, y):
    n = len(x)
    if n < 2:
        return 0
    
    # Calculate the integral using the trapezoidal rule
    integral = 0
    for i in range(n-1):
        h = x[i+1] - x[i]
        integral += h * (y[i] + y[i+1]) / 2
    
    return integral

# Compute the definite integral using the Compostion Simpson's 1/3 Rule
# x => array of x values
# y => array of y values
# -> returns integral value 
def simpson_rule(x, y):
    n = len(x)
    if n < 3:
        return trapezoidal_rule(x, y)  # Fall back to trapezoidal if not enough points
    
    # Check if the mesh is uniform
    h = x[1] - x[0]
    is_uniform = all(np.isclose(x[i+1] - x[i], h) for i in range(n-1))
    
    if is_uniform and n % 2 == 1:  # Simpson's rule requires an odd number of points
        # Standard Simpson's 1/3 formula for uniform mesh
        integral = y[0] + y[-1]  # First and last points
        
        # Add 4 times the odd-indexed points
        for i in range(1, n, 2):
            integral += 4 * y[i]
        
        # Add 2 times the even-indexed points (excluding first and last)
        for i in range(2, n-1, 2):
            integral += 2 * y[i]
        
        integral *= h / 3
        return integral
    else:
        # For non-uniform mesh or even number of points, we'll use piecewise cubic interpolation
        f = interpolate.interp1d(x, y, kind='cubic')
        
        # Create a finer uniform mesh for integration
        x_fine = np.linspace(x[0], x[-1], 1001)
        y_fine = f(x_fine)
        
        # Apply Simpson's rule on the fine mesh
        h_fine = (x_fine[-1] - x_fine[0]) / (len(x_fine) - 1)
        
        integral = y_fine[0] + y_fine[-1]  # First and last points
        
        # Add 4 times the odd-indexed points
        for i in range(1, len(x_fine), 2):
            integral += 4 * y_fine[i]
        
        # Add 2 times the even-indexed points (excluding first and last)
        for i in range(2, len(x_fine)-1, 2):
            integral += 2 * y_fine[i]
        
        integral *= h_fine / 3
        return integral

# Analyze the battery discharge curve date
# t_data => time data points
# v_data => voltage data points
# -> returns: - derivates (dictionary with derivate values)
#             - integrals (dicionary with integral values) 
#             - T_interp (interpolated time values)
#             - V_interp (interpolated voltage values)
def analyze_battery(t_data, V_data):
    # Create a cubic spline interpolation of the data
    f_interp = interpolate.interp1d(t_data, V_data, kind='cubic')
    
    # Create finer mesh for visualization
    t_interp = np.linspace(t_data[0], t_data[-1], 1000)
    V_interp = f_interp(t_interp)
    
    # Compute derivatives at t = 1800s and t = 5400s
    derivatives = {
        'forward_1800': numerical_derivative(t_data, V_data, 'forward', 1800),
        'backward_1800': numerical_derivative(t_data, V_data, 'backward', 1800),
        'central_1800': numerical_derivative(t_data, V_data, 'central', 1800),
        'forward_5400': numerical_derivative(t_data, V_data, 'forward', 5400),
        'backward_5400': numerical_derivative(t_data, V_data, 'backward', 5400),
        'central_5400': numerical_derivative(t_data, V_data, 'central', 5400)
    }
    
    # Battery current in amperes
    I = 1.3  # Given constant current
    
    # Compute the delivered energy using different integration methods
    # Energy = I * ∫V(t)dt from 0 to 7200s
    energy_trapz = I * trapezoidal_rule(t_data, V_data)
    energy_simpson = I * simpson_rule(t_data, V_data)
    
    integrals = {
        'trapz': energy_trapz,
        'simpson': energy_simpson
    }
    
    return derivatives, integrals, t_interp, V_interp 
