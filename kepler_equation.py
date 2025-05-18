import numpy as np

# Kepler's equation function: f(E) = E - e*sin(E) - M
# We try to find E such that f(E)=0
# E => eccentric anomaly 
# e => eccnetricity of the orbit
# M => mean anomaly
# -> returns the value of the function at E
def kepler_function(E, e, M):
    return E - e * np.sin(E) - M

# Derivate of Kepler's equations: f'(E) = 1 - e*cos(E)(for Newton's Method)
# E => eccentric anomaly 
# e => eccnetricity 
# -> value of the derivate at E
def kepler_derivative(E, e):
    return 1.0 - e * np.cos(E)

# Bisection method for the root finding
# f => function to find root of ...
# a, b => initial interval (f(a)*f(b) < 0)
# e => eccentricity
# M => mean anomaly
# tol => tolerance of stopping criteria
# max_iter => maximum number of iterations
# -> returns: - root (approximate root)
#             - iterations (number of iterations)
#             - error_history (list of errors at each iteration)
#             - E_vals ( list of E values at each iteration)
def bisection_method(f, a, b, e, M, tol=1e-10, max_iter=100):
    if f(a, e, M) * f(b, e, M) >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs")
    
    error_history = []
    E_values = []
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c, e, M)
        
        error = abs(fc)
        error_history.append(error)
        E_values.append(c)
        
        if error < tol or (b - a) / 2 < tol:
            return c, i+1, error_history, E_values
        
        if f(a, e, M) * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, error_history, E_values

# Newton's Method
# f => function to get the root
# df => dervate of the function 
# x0 => initial guess
# M => mean anomaly 
# e => eccentricity
# tol => tolerance for stopping criteria
# max_iter => maximum number of iterations
# -> returns: - root (arppox root)
#             - iterations (number of iterations)
#             - error_history (list of errors)
#             - E_values (list of E values at each iteration)
def newton_method(f, df, x0, e, M, tol=1e-10, max_iter=100):
    x = x0
    error_history = []
    E_values = []
    
    for i in range(max_iter):
        fx = f(x, e, M)
        dfx = df(x, e)
        
        error = abs(fx)
        error_history.append(error)
        E_values.append(x)
        
        if error < tol:
            return x, i+1, error_history, E_values
        
        # Avoid division by zero
        if abs(dfx) < 1e-10:
            dfx = 1e-10 if dfx >= 0 else -1e-10
            
        x_new = x - fx / dfx
        
        # Convergence check
        if abs(x_new - x) < tol:
            return x_new, i+1, error_history, E_values
            
        x = x_new
    
    return x, max_iter, error_history, E_values

# Uses Newton when it's safe and falls back to bisection when Newton might diverge
# f => function to get the root
# df => dervate of the function 
# a, b => initial interval (f(a)*f(b) < 0)
# e => eccentricity
# M => mean anomaly
# tol => tolerance for stopping criteria
# max_iter => maximum number of iterations
# -> returns: - root (arppox root)
#             - iterations (number of iterations)
#             - error_history (list of errors)
#             - E_values (list of E values at each iteration)
def hybrid_method(f, df, a, b, e, M, tol=1e-10, max_iter=100):
    if f(a, e, M) * f(b, e, M) >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs")
    
    x = (a + b) / 2  # Start with midpoint
    error_history = []
    E_values = []
    
    for i in range(max_iter):
        fx = f(x, e, M)
        
        error = abs(fx)
        error_history.append(error)
        E_values.append(x)
        
        if error < tol or (b - a) / 2 < tol:
            return x, i+1, error_history, E_values
        
        # Try Newton step
        dfx = df(x, e)
        
        # Avoid division by zero
        if abs(dfx) < 1e-10:
            dfx = 1e-10 if dfx >= 0 else -1e-10
            
        x_newton = x - fx / dfx
        
        # Check if Newton step is within the current bracket and makes good progress
        if a <= x_newton <= b and abs(x_newton - x) < 0.5 * (b - a):
            x_new = x_newton  # Accept Newton step
        else:
            # Fall back to bisection step
            c = (a + b) / 2
            x_new = c
            
            if f(a, e, M) * f(c, e, M) < 0:
                b = c
            else:
                a = c
        
        # Convergence check
        if abs(x_new - x) < tol:
            return x_new, i+1, error_history, E_values
            
        x = x_new
    
    return x, max_iter, error_history, E_values

# Solve Kepler]s equation for eccentri anomaly E
# e => eccentricity
# M => mean anomaly
# method => solution method Bisection/Newton/Hybrid
# initial_guess => intial guess for E
# -> returns: - E (radius/eccentric anomaly)
#             - iterations (number of iteration)
#             - error_history
#             - E_values
def solve_kepler(e, M, method="Newton", initial_guess=None):
    # Normalize M to be between 0 and 2π
    M = M % (2 * np.pi)
    
    # Default interval for bisection
    a, b = M, M + np.pi
    
    # Default initial guess for Newton
    if initial_guess is None:
        initial_guess = M
    
    # Choose method
    if method == "Bisection":
        E, iterations, error_history, E_values = bisection_method(
            kepler_function, a, b, e, M
        )
    elif method == "Newton":
        E, iterations, error_history, E_values = newton_method(
            kepler_function, kepler_derivative, initial_guess, e, M
        )
    elif method == "Hybrid":
        E, iterations, error_history, E_values = hybrid_method(
            kepler_function, kepler_derivative, a, b, e, M
        )
    else:
        raise ValueError("Method must be 'Bisection', 'Newton', or 'Hybrid'")
    
    return E, iterations, error_history, E_values 
