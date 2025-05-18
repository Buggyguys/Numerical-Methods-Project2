import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from matplotlib.animation import FuncAnimation
import os
import sys
from kepler_equation import solve_kepler
from battery_discharge import analyze_battery
from newton_cooling import simulate_cooling

# Create charts directory if it doesn't exist
if not os.path.exists("charts"):
    os.makedirs("charts")

st.set_page_config(page_title="Numerical Methods Project", layout="wide")

st.title("Numerical Methods Project")
st.markdown("This project demonstrates solutions to three numerical methods problems:")

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_exercise = st.sidebar.radio(
    "Select Exercise",
    [
        "A5. Kepler's Equation for Binary-Star Orbit",
        "B2. Battery Discharge Curve",
        "C3. Newton's Cooling of Coffee"
    ]
)

# Function to save current figure
def save_current_figure(fig, filename, save_type="matplotlib"):
    # Make sure charts directory exists
    if not os.path.exists("charts"):
        os.makedirs("charts")
    
    # Get full path for debugging
    full_path = os.path.abspath(f"charts/{filename}")
    
    try:
        if save_type == "matplotlib":
            fig.savefig(full_path)
            st.success(f"Chart saved as {full_path}")
        elif save_type == "plotly":
            # For Plotly figures, we need to use kaleido for static image export
            from plotly.io import write_image
            write_image(fig, full_path)
            st.success(f"Chart saved as {full_path}")
        
        # Show list of saved files for verification
        files = os.listdir("charts")
        if files:
            st.success(f"Files in charts directory: {', '.join(files)}")
        else:
            st.warning("Charts directory is empty. Save may have failed.")
    except Exception as e:
        st.error(f"Error saving chart: {str(e)}")
        st.error(f"Attempted to save to: {full_path}")

# A5. Kepler's Equation
if selected_exercise == "A5. Kepler's Equation for Binary-Star Orbit":
    st.header("A5. Kepler's Equation for Binary-Star Orbit")
    st.markdown(r"""
    The Kepler equation is:
    $$E - e \sin(E) = M$$
    
    Where:
    - $E$ is the eccentric anomaly (the angle we're solving for)
    - $e$ is the eccentricity of the orbit (0.67)
    - $M$ is the mean anomaly (1.8 radians)
    """)
    
    # Inputs with default values
    e = st.slider("Eccentricity (e)", 0.0, 0.99, 0.67, 0.01)
    M = st.slider("Mean Anomaly (M) [radians]", 0.0, 6.28, 1.8, 0.01)
    initial_guess = st.slider("Initial Guess for E", 0.0, 6.28, 1.0, 0.01)
    
    method = st.selectbox("Solution Method", ["Bisection", "Newton", "Hybrid"])
    
    # Solve Kepler's equation
    if st.button("Solve Kepler's Equation"):
        result, iterations, error_hist, E_values = solve_kepler(e, M, method, initial_guess)
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Eccentric Anomaly (E):** {result:.6f} radians")
            st.markdown(f"**Iterations:** {iterations}")
            st.markdown(f"**Final Error:** {abs(result - e*np.sin(result) - M):.2e}")
        
        # Standard Visualization - Convergence plot
        with col2:
            fig, ax = plt.subplots()
            ax.semilogy(range(len(error_hist)), error_hist, 'o-')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Error (log scale)')
            ax.set_title(f'Convergence of {method} Method')
            ax.grid(True)
            st.pyplot(fig)
            
            # Save button for standard visualization
            if st.button("Save Convergence Plot"):
                save_current_figure(fig, f"kepler_convergence_{method}.png")
        
        # Advanced Visualization - Orbit visualization
        st.subheader("Orbit Visualization")
        tab1, tab2 = st.tabs(["2D Orbit", "3D Interactive Orbit"])
        
        with tab1:
            # Create 2D orbit animation
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True)
            
            # Plot the elliptical orbit
            theta = np.linspace(0, 2*np.pi, 100)
            a = 1.0  # semi-major axis
            b = a * np.sqrt(1 - e**2)  # semi-minor axis
            x = a * np.cos(theta)
            y = b * np.sin(theta)
            ax.plot(x, y, 'b-', linewidth=1, label='Orbit')
            
            # Mark the focus (Sun)
            ax.plot(0, 0, 'yo', markersize=15, label='Star')
            
            # Mark the position of the planet at calculated E
            x_planet = a * (np.cos(result))
            y_planet = b * (np.sin(result))
            ax.plot(x_planet, y_planet, 'ro', markersize=10, label='Planet')
            
            # Draw the radial line
            ax.plot([0, x_planet], [0, y_planet], 'g-', linewidth=2, label='Radial Line')
            
            ax.legend()
            ax.set_title('Binary Star Orbit')
            st.pyplot(fig)
            
            if st.button("Save 2D Orbit"):
                save_current_figure(fig, f"kepler_orbit_2d_{method}.png")
        
        with tab2:
            # 3D interactive orbit
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            
            # Sphere for the star
            star_radius = 0.2
            x_star = star_radius * np.outer(np.cos(u), np.sin(v))
            y_star = star_radius * np.outer(np.sin(u), np.sin(v))
            z_star = star_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Sphere for the planet
            planet_radius = 0.1
            x_planet_sphere = x_planet + planet_radius * np.outer(np.cos(u), np.sin(v))
            y_planet_sphere = y_planet + planet_radius * np.outer(np.sin(u), np.sin(v))
            z_planet_sphere = 0 + planet_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Create 3D orbit
            orbit_x = a * np.cos(theta)
            orbit_y = b * np.sin(theta)
            orbit_z = np.zeros_like(theta)
            
            # Create the 3D plot
            fig = go.Figure()
            
            # Add the star
            fig.add_trace(go.Surface(
                x=x_star, y=y_star, z=z_star,
                colorscale=[[0, 'yellow'], [1, 'yellow']],
                showscale=False,
                name='Star'
            ))
            
            # Add the planet
            fig.add_trace(go.Surface(
                x=x_planet_sphere, y=y_planet_sphere, z=z_planet_sphere,
                colorscale=[[0, 'blue'], [1, 'blue']],
                showscale=False,
                name='Planet'
            ))
            
            # Add the orbit
            fig.add_trace(go.Scatter3d(
                x=orbit_x, y=orbit_y, z=orbit_z,
                mode='lines',
                line=dict(color='white', width=2),
                name='Orbit'
            ))
            
            # Add the radial line
            fig.add_trace(go.Scatter3d(
                x=[0, x_planet], y=[0, y_planet], z=[0, 0],
                mode='lines',
                line=dict(color='green', width=4),
                name='Radial Line'
            ))
            
            fig.update_layout(
                title='3D Visualization of Binary Star Orbit',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                width=800,
                height=800
            )
            
            st.plotly_chart(fig)
            
            if st.button("Save 3D Orbit"):
                save_current_figure(fig, f"kepler_orbit_3d_{method}.png", save_type="plotly")

# B2. Battery Discharge Curve
elif selected_exercise == "B2. Battery Discharge Curve":
    st.header("B2. Battery Discharge Curve (0.5 C)")
    
    st.markdown("""
    This exercise analyzes a battery discharge curve, computing:
    1. Voltage derivatives at specific times using different numerical differentiation methods
    2. The total energy delivered using numerical integration methods
    """)
    
    # Display the data
    st.subheader("Battery Discharge Data")
    t_data = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200]
    V_data = [4.20, 4.09, 3.99, 3.90, 3.83, 3.77, 3.71, 3.66, 3.61, 3.56, 3.51, 3.46, 3.40]
    
    data_df = {"Time (s)": t_data, "Voltage (V)": V_data}
    st.table(data_df)
    
    # Compute results
    if st.button("Analyze Battery Discharge"):
        derivatives, integrals, t_interp, V_interp = analyze_battery(t_data, V_data)
        
        st.subheader("Derivative Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Voltage Rate of Change at t = 1800s")
            st.table({
                "Method": ["Forward", "Backward", "Central"],
                "dV/dt (V/s)": [f"{derivatives['forward_1800']:.6e}", 
                                f"{derivatives['backward_1800']:.6e}", 
                                f"{derivatives['central_1800']:.6e}"]
            })
            
            st.markdown("#### Voltage Rate of Change at t = 5400s")
            st.table({
                "Method": ["Forward", "Backward", "Central"],
                "dV/dt (V/s)": [f"{derivatives['forward_5400']:.6e}", 
                                f"{derivatives['backward_5400']:.6e}", 
                                f"{derivatives['central_5400']:.6e}"]
            })
        
        with col2:
            # Standard visualization - Basic discharge curve
            fig, ax = plt.subplots()
            ax.plot(t_data, V_data, 'bo-', label='Measured Data')
            ax.plot(t_interp, V_interp, 'r-', label='Interpolated Curve', alpha=0.5)
            
            # Mark points where derivatives are calculated
            ax.plot(1800, np.interp(1800, t_data, V_data), 'go', markersize=10, label='dV/dt at 1800s')
            ax.plot(5400, np.interp(5400, t_data, V_data), 'mo', markersize=10, label='dV/dt at 5400s')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title('Battery Discharge Curve')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            if st.button("Save Discharge Curve"):
                save_current_figure(fig, "battery_discharge_curve.png")
        
        # Integration results
        st.subheader("Integration Results - Delivered Energy")
        
        col1, col2 = st.columns(2)
        with col1:
            st.table({
                "Method": ["Trapezoidal Rule", "Simpson's 1/3 Rule"],
                "Energy (J)": [f"{integrals['trapz']:.4f}", f"{integrals['simpson']:.4f}"]
            })
        
        with col2:
            # Advanced visualization - 3D Surface showing the energy
            t_mesh = np.linspace(t_data[0], t_data[-1], 100)
            V_mesh = np.interp(t_mesh, t_data, V_data)
            
            # Create mesh grid for 3D surface
            T, TIME = np.meshgrid(np.linspace(0, 1, 10), t_mesh)
            V_surface = np.zeros_like(T)
            
            for i in range(T.shape[1]):
                V_surface[:, i] = V_mesh * (1 - 0.1*i)  # Just for visualization effect
            
            fig = go.Figure(data=[go.Surface(z=V_surface, x=TIME, y=T, colorscale='Viridis')])
            
            fig.update_layout(
                title='3D Battery Discharge Surface',
                scene=dict(
                    xaxis_title='Time (s)',
                    yaxis_title='Parameter',
                    zaxis_title='Voltage (V)',
                ),
                width=700,
                height=700
            )
            
            st.plotly_chart(fig)
            
            if st.button("Save 3D Discharge Surface"):
                save_current_figure(fig, "battery_discharge_3d.png", save_type="plotly")

# C3. Newton's Cooling of Coffee
elif selected_exercise == "C3. Newton's Cooling of Coffee":
    st.header("C3. Newton's Cooling of Coffee")
    
    st.markdown(r"""
    Newton's Law of Cooling is given by:
    $$T'(t) = -k(T - T_{\infty})$$
    
    With initial conditions:
    - $T(0) = 90°C$ (initial temperature)
    - $T_{\infty} = 25°C$ (ambient temperature)
    - $k = 0.07 \text{ min}^{-1}$ (cooling constant)
    
    The analytical solution is:
    $$T(t) = T_{\infty} + (T_0 - T_{\infty})e^{-kt}$$
    """)
    
    # Inputs with default values
    col1, col2 = st.columns(2)
    with col1:
        T0 = st.number_input("Initial Temperature (°C)", value=90.0)
        T_inf = st.number_input("Ambient Temperature (°C)", value=25.0)
        k = st.number_input("Cooling Constant (min⁻¹)", value=0.07, format="%.4f")
        
    with col2:
        target_temp = st.slider("Target Temperature (°C)", 25.0, 90.0, 60.0)
        t_max = st.slider("Maximum Time (min)", 10, 120, 60)
        
    # Numerical method selection
    methods = st.multiselect(
        "Select Numerical Methods",
        ["Euler", "RK2", "RK4", "RK45"],
        ["Euler", "RK4"]
    )
    
    step_sizes = st.multiselect(
        "Select Step Sizes (min)",
        [0.1, 0.5, 1.0, 2.0, 5.0],
        [1.0]
    )
    
    if st.button("Simulate Cooling"):
        results = simulate_cooling(T0, T_inf, k, t_max, target_temp, methods, step_sizes)
        
        # Display analytical time to reach target
        st.subheader(f"Time to reach {target_temp}°C")
        st.markdown(f"**Analytical Solution:** {results['analytical_time']:.2f} minutes")
        
        for method in methods:
            time_value = results.get(f'{method}_time')
            if time_value is not None:
                st.markdown(f"**{method}:** {time_value:.2f} minutes")
            else:
                st.markdown(f"**{method}:** Target temperature not reached")
        
        # Standard visualization - Temperature vs. Time
        st.subheader("Temperature vs. Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot analytical solution
        t_analytical = results['t_analytical']
        T_analytical = results['T_analytical']
        ax.plot(t_analytical, T_analytical, 'k-', label='Analytical', linewidth=2)
        
        # Plot numerical solutions
        for method in methods:
            for step_size in step_sizes:
                t = results[f't_{method}_{step_size}']
                T = results[f'T_{method}_{step_size}']
                ax.plot(t, T, 'o-', label=f'{method} (h={step_size})', alpha=0.7, markersize=3)
        
        # Mark the target temperature
        ax.axhline(y=target_temp, color='r', linestyle='--', label=f'Target Temp ({target_temp}°C)')
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Coffee Cooling Simulation')
        ax.grid(True)
        ax.legend()
        
        st.pyplot(fig)
        
        if st.button("Save Temperature Plot"):
            save_current_figure(fig, "coffee_cooling_temperature.png")
        
        # Advanced visualization - Error Analysis
        st.subheader("Error Analysis")
        
        # Tab for different visualizations
        tab1, tab2 = st.tabs(["Error vs. Step Size", "3D Heat Visualization"])
        
        with tab1:
            # Error vs step size plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for method in methods:
                error_data = []
                step_sizes_sorted = sorted(step_sizes)
                
                for step_size in step_sizes_sorted:
                    error = results[f'error_{method}_{step_size}']
                    error_data.append(error)
                
                ax.loglog(step_sizes_sorted, error_data, 'o-', label=f'{method}')
            
            # Add reference slopes
            x_ref = np.array([min(step_sizes_sorted), max(step_sizes_sorted)])
            
            # First order reference (h)
            y_ref1 = error_data[-1] * (x_ref / x_ref[-1])
            ax.loglog(x_ref, y_ref1, 'k--', alpha=0.5, label='First Order (h)')
            
            # Second order reference (h²)
            y_ref2 = error_data[-1] * (x_ref / x_ref[-1])**2
            ax.loglog(x_ref, y_ref2, 'k-.', alpha=0.5, label='Second Order (h²)')
            
            # Fourth order reference (h⁴)
            y_ref4 = error_data[-1] * (x_ref / x_ref[-1])**4
            ax.loglog(x_ref, y_ref4, 'k:', alpha=0.5, label='Fourth Order (h⁴)')
            
            ax.set_xlabel('Step Size (h)')
            ax.set_ylabel('Maximum Error')
            ax.set_title('Error vs. Step Size')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            if st.button("Save Error Plot", key="save_error_plot"):
                try:
                    filename = "coffee_cooling_error.png"
                    save_current_figure(fig, filename)
                    
                    # Additional check to verify the file was saved
                    saved_path = os.path.join("charts", filename)
                    if os.path.exists(saved_path):
                        st.success(f"Verified: File exists at {saved_path}")
                        st.success(f"File size: {os.path.getsize(saved_path) / 1024:.1f} KB")
                    else:
                        st.error(f"File was not created at {saved_path}")
                except Exception as e:
                    st.error(f"Error in save button handler: {str(e)}")
        
        with tab2:
            # 3D visualization of heat diffusion
            # Create a cylinder representing the coffee cup
            st.markdown("### 3D Heat Visualization")
            
            # Create time steps for the animation
            t_anim = np.linspace(0, t_max, 100)
            T_anim = T_inf + (T0 - T_inf) * np.exp(-k * t_anim)
            
            # Create normalized temperatures for the color scale (0 to 1)
            T_norm = (T_anim - T_inf) / (T0 - T_inf)
            
            # Create cylinder coordinates
            theta = np.linspace(0, 2*np.pi, 30)
            z = np.linspace(0, 1, 20)
            Theta, Z = np.meshgrid(theta, z)
            X = np.cos(Theta)
            Y = np.sin(Theta)
            
            # Initialize session state for time index and camera position
            if 'heat_time_idx' not in st.session_state:
                st.session_state.heat_time_idx = 0
            if 'camera_position' not in st.session_state:
                st.session_state.camera_position = dict(eye=dict(x=1.5, y=1.5, z=1))
            
            # Define callback for time index slider changes
            def update_time_idx():
                st.session_state.heat_time_idx = st.session_state.time_slider
            
            # Create a slider for time that updates session state
            time_slider = st.slider(
                "Time Index", 
                0, 
                len(t_anim)-1, 
                st.session_state.heat_time_idx,
                key="time_slider",
                on_change=update_time_idx
            )
            
            # Use the time index from session state
            time_idx = st.session_state.heat_time_idx
            
            # Get temperature at selected time
            current_temp = T_anim[time_idx]
            current_t = t_anim[time_idx]
            
            # Display current time and temperature as text (for debugging)
            st.text(f"Current time index: {time_idx}, t={current_t:.1f} min, T={current_temp:.1f}°C")
            
            # Create a 3D plot
            fig = go.Figure()
            
            # Add the cup surface with color based on temperature
            color_vals = T_norm[time_idx] * np.ones_like(Z)  # Uniform temperature at this time
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=color_vals,
                colorscale='Thermal',
                colorbar=dict(title='Temperature', tickvals=[0, 0.5, 1], 
                            ticktext=[f'{T_inf}°C', f'{(T0+T_inf)/2}°C', f'{T0}°C']),
                showscale=True,
                cmin=0, cmax=1
            ))
            
            # Add a circle for the surface of the coffee
            circle_z = 0.8 * np.ones_like(theta)
            circle_x = 0.9 * np.cos(theta)
            circle_y = 0.9 * np.sin(theta)
            
            fig.add_trace(go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode='lines',
                line=dict(color='black', width=4),
                name='Coffee Surface'
            ))
            
            # Update layout while preserving camera position
            fig.update_layout(
                title=f'Coffee Cup at t={current_t:.1f} min, T={current_temp:.1f}°C',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Height',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.5),
                    camera=st.session_state.camera_position,
                ),
                width=700,
                height=700,
                uirevision=True  # This preserves user interactions with the plot
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Use a unique key for the save button
            if st.button("Save 3D Heat Visualization", key="save_3d_heat_viz"):
                try:
                    # Ensure the figure is properly prepared for saving
                    filename = f"coffee_cooling_3d_t{current_t:.1f}.png"
                    save_current_figure(fig, filename, save_type="plotly")
                    
                    # Additional check to verify the file was saved
                    saved_path = os.path.join("charts", filename)
                    if os.path.exists(saved_path):
                        st.success(f"Verified: File exists at {saved_path}")
                        st.success(f"File size: {os.path.getsize(saved_path) / 1024:.1f} KB")
                    else:
                        st.error(f"File was not created at {saved_path}")
                except Exception as e:
                    st.error(f"Error in save button handler: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.markdown("""
This application demonstrates numerical methods for solving three different problems:
- Kepler's equation for binary star orbits (root-finding)
- Battery discharge curve analysis (differentiation & integration)
- Newton's cooling of coffee (ODEs)

""") 
