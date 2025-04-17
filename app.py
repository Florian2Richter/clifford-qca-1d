import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric, plot_spacetime_plotly
from matplotlib.colors import ListedColormap
import time
import hashlib

# Set page configuration
st.set_page_config(
    page_title="1D Clifford QCA Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add version indicator to verify deployment
st.sidebar.markdown("**App Version: 2025-04-15**")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .sidebar-header {
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a hash of all parameters to detect changes
def get_params_hash(n, T_steps, local_rule, initial_state):
    hash_str = f"{n}_{T_steps}_{local_rule.tobytes().hex()}_{initial_state.tobytes().hex()}"
    return hashlib.md5(hash_str.encode()).hexdigest()

# Initialize session state for progressive simulation
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_step = 0
    st.session_state.pauli_strings = []
    st.session_state.states = []
    st.session_state.global_operator = None
    st.session_state.params_hash = ""
    st.session_state.target_steps = 0
    st.session_state.simulation_running = False
    st.session_state.simulation_complete = False

# Main title
st.markdown('<h1 class="main-header">1D Clifford QCA Simulator</h1>', unsafe_allow_html=True)

# Description
st.markdown("""
<div class="description">
This simulator visualizes the evolution of a 1-Dimensional Clifford Quantum Cellular Automaton (QCA). 
The simulation shows how Pauli operators (I, X, Z, Y) propagate through a 1D lattice over time.
</div>
""", unsafe_allow_html=True)

# Sidebar for simulation parameters
st.sidebar.markdown('<h3 class="sidebar-header">Simulation Parameters</h3>', unsafe_allow_html=True)
n = st.sidebar.number_input("Number of cells", min_value=3, value=500, step=1)
T_steps = st.sidebar.number_input("Number of time steps", min_value=1, value=250, step=1)

# Sidebar for local rule matrix input
st.sidebar.markdown('<h3 class="sidebar-header">Local Rule Matrix (2x6 over F2)</h3>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="description">
Enter each row as 6 numbers (0 or 1) separated by spaces.
The local rule determines how each cell updates based on its neighbors.

For example, the identity transformation would be:
```
0 0 1 0 0 0  (first row)
0 0 0 1 0 0  (second row)
```
This leaves each cell's state unchanged as it only uses the identity matrix in the center block.
</div>
""", unsafe_allow_html=True)

row1_input = st.sidebar.text_input("Row 1 (for A_left and A_center)", "1 0 1 1 0 1")
row2_input = st.sidebar.text_input("Row 2 (for A_center and A_right)", "0 1 0 1 1 0")

def parse_matrix_row(row_str):
    try:
        values = [int(x) for x in row_str.split()]
        if len(values) != 6:
            st.sidebar.error("Each row must have exactly 6 numbers.")
            return None
        for v in values:
            if v not in (0, 1):
                st.sidebar.error("Only 0 and 1 are allowed.")
                return None
        return values
    except Exception as e:
        st.sidebar.error("Error parsing the row: " + str(e))
        return None

row1 = parse_matrix_row(row1_input)
row2 = parse_matrix_row(row2_input)

if row1 is not None and row2 is not None:
    local_rule = np.array([row1, row2], dtype=int) % 2
else:
    st.stop()

# Initial state selection
st.sidebar.markdown('<h3 class="sidebar-header">Initial State</h3>', unsafe_allow_html=True)
init_option = st.sidebar.selectbox("Choose initial state:", ["Single active cell", "Random", "Manual"])

def get_single_active_state(n):
    # Create a state vector of length 2*n representing all I's,
    # except for one X (i.e. (1,0)) at the center.
    state = np.zeros(2 * n, dtype=int)
    center = n // 2
    state[2*center] = 1  # Set x-part for an X operator
    return state

if init_option == "Single active cell":
    initial_state = get_single_active_state(n)
    st.sidebar.info("Using a single X operator at the center cell.")
elif init_option == "Random":
    # Generate a random Pauli string of length n from I, X, Z, Y.
    choices = ['I', 'X', 'Z', 'Y']
    random_pauli = ''.join(np.random.choice(choices, size=n))
    st.sidebar.info(f"Random initial state: {random_pauli}")
    initial_state = pauli_string_to_state(random_pauli)
elif init_option == "Manual":
    manual_pauli = st.sidebar.text_input("Pauli string (I, X, Z, Y)", "I"*(n//2) + "X" + "I"*(n - n//2 - 1))
    if len(manual_pauli) != n:
        st.sidebar.error("Pauli string must be of length equal to the number of cells.")
        st.stop()
    valid_chars = set("IXZY")
    if any(ch not in valid_chars for ch in manual_pauli):
        st.sidebar.error("Invalid characters in Pauli string. Use only I, X, Z, Y.")
        st.stop()
    initial_state = pauli_string_to_state(manual_pauli)
else:
    initial_state = get_single_active_state(n)

# Build the global operator from the local rule
global_operator = build_global_operator(n, local_rule)

# Create a placeholder for the plot
plot_placeholder = st.empty()

# Create a placeholder for status messages
status_placeholder = st.empty()

# Calculate parameters hash to detect changes
current_hash = get_params_hash(n, T_steps, local_rule, initial_state)

# Reset simulation if parameters changed
if current_hash != st.session_state.params_hash:
    st.session_state.params_hash = current_hash
    st.session_state.current_step = 0
    st.session_state.pauli_strings = [vector_to_pauli_string(initial_state)]
    st.session_state.states = [initial_state.copy()]
    st.session_state.global_operator = global_operator
    st.session_state.target_steps = T_steps
    st.session_state.simulation_running = True
    st.session_state.simulation_complete = False
    st.session_state.initialized = True

# Function to calculate one time step
def calculate_step(current_state, step_number):
    start_time = time.time()
    
    # Calculate next state - use the stored global operator from session state
    next_state = mod2_matmul(st.session_state.global_operator, current_state) % 2
    
    # Convert to Pauli string
    next_pauli = vector_to_pauli_string(next_state)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    return next_state, next_pauli, calculation_time

# Progressive simulation - calculate one step at a time and update UI
if st.session_state.initialized and st.session_state.simulation_running:
    if st.session_state.current_step < st.session_state.target_steps:
        # Create a progress bar
        # progress_bar = st.progress(0)
        
        # Get starting state
        current_state = st.session_state.states[-1]
        
        # Calculate all remaining time steps one by one
        for step in range(st.session_state.current_step, st.session_state.target_steps):
            # Calculate next time step
            next_step = step + 1
            next_state, next_pauli, calc_time = calculate_step(current_state, next_step)
            
            # Store the results
            st.session_state.states.append(next_state.copy())
            st.session_state.pauli_strings.append(next_pauli)
            
            # Update current state for next iteration
            current_state = next_state
            
            # Increment step counter
            st.session_state.current_step += 1
            
            # Update the plot with current progress
            fig = plot_spacetime_plotly(
                st.session_state.pauli_strings, 
                total_time_steps=st.session_state.target_steps
            )
            
            # Display the updated plot
            plot_placeholder.plotly_chart(fig, use_container_width=False)
            
            # Update progress bar and status message
            #progress_value = st.session_state.current_step / st.session_state.target_steps
            #progress_bar.progress(progress_value)
            #status_placeholder.info(f"Calculating time step {st.session_state.current_step}/{st.session_state.target_steps} ({progress_value*100:.1f}%)")
            
            # Small sleep to allow UI to update (can be adjusted)
            time.sleep(0.005)
        
        # Simulation complete
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        #progress_bar.empty()
        status_placeholder.success("Simulation complete!")
    
    elif not st.session_state.simulation_complete:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")

# If simulation complete, just show the final plot
if st.session_state.simulation_complete:
    fig = plot_spacetime_plotly(st.session_state.pauli_strings)
    plot_placeholder.plotly_chart(fig, use_container_width=False)

# For initial load, show empty plot
if not st.session_state.initialized:
    # Initial state only - show first step
    initial_pauli = vector_to_pauli_string(initial_state)
    fig = plot_spacetime_plotly([initial_pauli], total_time_steps=T_steps)
    plot_placeholder.plotly_chart(fig, use_container_width=False)
    
    # Start simulation
    st.session_state.pauli_strings = [initial_pauli]
    st.session_state.states = [initial_state.copy()]
    st.session_state.global_operator = global_operator
    st.session_state.target_steps = T_steps
    st.session_state.params_hash = current_hash
    st.session_state.simulation_running = True
    st.session_state.initialized = True
