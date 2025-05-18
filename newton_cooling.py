import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

# Newton's law of cooling: dT/dt = -k(T - T_inf)
# t => time
# T => temperature
# k => cooling constant
# T_inf => ambient temperature
# -> returns rate of temperature change
def cooling_rate(t, T, k, T_inf):
    return -k * (T - T_inf)

# Analytical solution to Newton's law of cooling
# t => time or array of times
# T0 => initial temperature
# T_inf => ambient temperature
# k => cooling constant
# -> returns temperature at time t
def analytical_solution(t, T0, T_inf, k):
    return T_inf + (T0 - T_inf) * np.exp(-k * t)

# Calculate the time to reach a target temperature
# T0 => initial temperature
# T_inf => ambient temperature
# target => target temperature
# k => cooling constant
# -> returns time to reach target temperature
def time_to_target(T0, T_inf, target, k):
    # From T(t) = T_inf + (T0 - T_inf) * exp(-k*t)
    # Solve for t: t = -ln((T - T_inf)/(T0 - T_inf)) / k
    
    # Check if target is reachable
    if (T0 > T_inf and target < T_inf) or (T0 < T_inf and target > T_inf):
        return None  # Target not reachable
    
    if T0 == target:
        return 0  # Already at target
    
    return -np.log((target - T_inf) / (T0 - T_inf)) / k

# Implement Euler's method for solving ODEs
# f => function defining the ODE (dy/dt = f(t, y))
# t0 => initial time
# y0 => initial value
# t_end => end time
# h => step size
# args => additional arguments to pass to f
# -> returns: - t_values (array of time points)
#             - y_values (array of solution values)
def euler_method(f, t0, y0, t_end, h, args=()):
    # Calculate number of steps
    n_steps = math.ceil((t_end - t0) / h)
    
    # Initialize arrays
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # Euler's method
    for i in range(n_steps):
        t_values[i+1] = t_values[i] + h
        y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i], *args)
    
    return t_values, y_values

# Implement Runge-Kutta 2nd order method (Midpoint method)
# f => function defining the ODE (dy/dt = f(t, y))
# t0 => initial time
# y0 => initial value
# t_end => end time
# h => step size
# args => additional arguments to pass to f
# -> returns: - t_values (array of time points)
#             - y_values (array of solution values)
def rk2_method(f, t0, y0, t_end, h, args=()):
    # Calculate number of steps
    n_steps = math.ceil((t_end - t0) / h)
    
    # Initialize arrays
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # RK2 method
    for i in range(n_steps):
        t = t_values[i]
        y = y_values[i]
        
        # Stage 1
        k1 = f(t, y, *args)
        
        # Stage 2
        k2 = f(t + h/2, y + h/2 * k1, *args)
        
        # Update
        t_values[i+1] = t + h
        y_values[i+1] = y + h * k2
    
    return t_values, y_values

# Implement Runge-Kutta 4th order method
# f => function defining the ODE (dy/dt = f(t, y))
# t0 => initial time
# y0 => initial value
# t_end => end time
# h => step size
# args => additional arguments to pass to f
# -> returns: - t_values (array of time points)
#             - y_values (array of solution values)
def rk4_method(f, t0, y0, t_end, h, args=()):
    # Calculate number of steps
    n_steps = math.ceil((t_end - t0) / h)
    
    # Initialize arrays
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set initial values
    t_values[0] = t0
    y_values[0] = y0
    
    # RK4 method
    for i in range(n_steps):
        t = t_values[i]
        y = y_values[i]
        
        # Stage 1
        k1 = f(t, y, *args)
        
        # Stage 2
        k2 = f(t + h/2, y + h/2 * k1, *args)
        
        # Stage 3
        k3 = f(t + h/2, y + h/2 * k2, *args)
        
        # Stage 4
        k4 = f(t + h, y + h * k3, *args)
        
        # Update
        t_values[i+1] = t + h
        y_values[i+1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, y_values

# Find the time at which a target value is reached in numerical solution
# t_values => array of time points
# y_values => array of solution values
# target => target value
# -> returns time at which target is reached (interpolated if necessary)
def find_time_to_target_numerical(t_values, y_values, target):
    # Find where the solution crosses the target
    if y_values[0] > target:
        # Cooling down to target
        idx = np.where(y_values <= target)[0]
        if len(idx) == 0:
            return None  # Target not reached
        idx = idx[0]
    else:
        # Heating up to target
        idx = np.where(y_values >= target)[0]
        if len(idx) == 0:
            return None  # Target not reached
        idx = idx[0]
    
    # If the first point already meets the condition
    if idx == 0:
        return t_values[0]
    
    # Linear interpolation to find exact time
    t1, t2 = t_values[idx-1], t_values[idx]
    y1, y2 = y_values[idx-1], y_values[idx]
    
    # t = t1 + (t2 - t1) * (target - y1) / (y2 - y1)
    t_target = t1 + (t2 - t1) * (target - y1) / (y2 - y1)
    return t_target

# Calculate the maximum absolute error between analytical and numerical solutions
# analytical => array of analytical solution values
# numerical => array of numerical solution values
# -> returns maximum absolute error
def max_error(analytical, numerical):
    return np.max(np.abs(analytical - numerical))

# Simulate the cooling of coffee using different numerical methods
# T0 => initial temperature (°C)
# T_inf => ambient temperature (°C)
# k => cooling constant (min⁻¹)
# t_max => maximum simulation time (min)
# target_temp => target temperature to reach (°C)
# methods => list of numerical methods to use
# step_sizes => list of step sizes to use
# -> returns: - results (dictionary with simulation results)
def simulate_cooling(T0, T_inf, k, t_max, target_temp, methods, step_sizes):
    # Calculate analytical solution
    t_analytical = np.linspace(0, t_max, 1000)
    T_analytical = analytical_solution(t_analytical, T0, T_inf, k)
    
    # Calculate analytical time to reach target
    analytical_target_time = time_to_target(T0, T_inf, target_temp, k)
    
    # Initialize results dictionary
    results = {
        't_analytical': t_analytical,
        'T_analytical': T_analytical,
        'analytical_time': analytical_target_time
    }
    
    # Simulate with different methods and step sizes
    for method in methods:
        for step_size in step_sizes:
            # Choose the appropriate method
            if method == 'Euler':
                t, T = euler_method(cooling_rate, 0, T0, t_max, step_size, args=(k, T_inf))
            elif method == 'RK2':
                t, T = rk2_method(cooling_rate, 0, T0, t_max, step_size, args=(k, T_inf))
            elif method == 'RK4':
                t, T = rk4_method(cooling_rate, 0, T0, t_max, step_size, args=(k, T_inf))
            elif method == 'RK45':
                # Use scipy's adaptive solver
                sol = solve_ivp(
                    cooling_rate, [0, t_max], [T0], 
                    args=(k, T_inf),
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-9
                )
                t, T = sol.t, sol.y[0]
            else:
                continue
            
            # Store results
            results[f't_{method}_{step_size}'] = t
            results[f'T_{method}_{step_size}'] = T
            
            # Calculate time to reach target
            target_time = find_time_to_target_numerical(t, T, target_temp)
            results[f'{method}_time'] = target_time
            
            # Calculate maximum error
            # Interpolate analytical solution at numerical time points for comparison
            T_analytical_at_t = analytical_solution(t, T0, T_inf, k)
            error = max_error(T_analytical_at_t, T)
            results[f'error_{method}_{step_size}'] = error
    
    return results 