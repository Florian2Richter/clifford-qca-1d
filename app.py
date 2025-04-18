import streamlit as st
import numpy as np
from qca.core import build_global_operator, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric, make_empty_figure, update_figure
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
st.sidebar.markdown("**App Version: 2025-04-19.2 (simplified calculation)**")

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
    st.session_state.fig = None

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
T_steps = st.sidebar.number_input("Number of time steps", min_value=1, value=50, step=1)

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
    state = np.zeros(2 * n, dtype=int)
    center = n // 2
    state[2*center] = 1
    return state

if init_option == "Single active cell":
    initial_state = get_single_active_state(n)
    st.sidebar.info("Using a single X operator at the center cell.")
elif init_option == "Random":
    choices = ['I', 'X', 'Z', 'Y']
    random_pauli = ''.join(np.random.choice(choices, size=n))
    st.sidebar.info(f"Random initial state: {random_pauli}")
    initial_state = pauli_string_to_state(random_pauli)
elif init_option == "Manual":
    manual_pauli = st.sidebar.text_input("Pauli string (I, X, Z, Y)", "I"*(n//2) + "X" + "I"*(n - n//2 - 1))
    if len(manual_pauli) != n:
        st.sidebar.error("Pauli string must be of length equal to the number of cells.")
        st.stop()
    if any(ch not in set("IXZY") for ch in manual_pauli):
        st.sidebar.error("Invalid characters in Pauli string. Use only I, X, Z, Y.")
        st.stop()
    initial_state = pauli_string_to_state(manual_pauli)
else:
    initial_state = get_single_active_state(n)

# Build the global operator from the local rule
global_operator = build_global_operator(n, local_rule)

# Placeholders for plot and status
plot_placeholder = st.empty()
status_placeholder = st.empty()

# Detect parameter changes
current_hash = get_params_hash(n, T_steps, local_rule, initial_state)
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
    st.session_state.fig = None

# Function to calculate one time step
def calculate_step(current_state):
    """Calculate the next state for the QCA simulation."""
    next_state = mod2_matmul(st.session_state.global_operator, current_state) % 2
    next_pauli = vector_to_pauli_string(next_state)
    return next_state, next_pauli

# Progressive simulation
BATCH_SIZE = 5
if st.session_state.initialized and st.session_state.simulation_running:
    if st.session_state.current_step < st.session_state.target_steps:
        # Create the figure once on first batch
        if st.session_state.fig is None:
            st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
            plot_placeholder.plotly_chart(
                st.session_state.fig,
                use_container_width=False,
                key=f"init_plot_{current_hash[:8]}"
            )
            
        for step in range(st.session_state.current_step, st.session_state.target_steps):
            next_step = step + 1
            next_state, next_pauli = calculate_step(st.session_state.states[-1])
            st.session_state.states.append(next_state.copy())
            st.session_state.pauli_strings.append(next_pauli)
            st.session_state.current_step += 1

            if st.session_state.current_step % BATCH_SIZE == 0 or st.session_state.current_step == st.session_state.target_steps:
                # Update the existing figure instead of creating a new one
                update_figure(st.session_state.fig, st.session_state.pauli_strings)
                plot_placeholder.plotly_chart(
                    st.session_state.fig,
                    use_container_width=False,
                    key=f"step_{st.session_state.current_step}_{current_hash[:8]}"
                )
                time.sleep(0.005)

        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")
    elif not st.session_state.simulation_complete:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")

# Final plot 
elif st.session_state.simulation_complete:
    # Safety check if fig doesn't exist for some reason
    if "fig" not in st.session_state or st.session_state.fig is None:
        st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
        update_figure(st.session_state.fig, st.session_state.pauli_strings)
    
    # No need to create a new figure, the last update already has all data
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        key=f"final_plot_{current_hash[:8]}"
    )

# Initial load
elif not st.session_state.initialized:
    initial_pauli = vector_to_pauli_string(initial_state)
    # Create the figure once
    st.session_state.fig = make_empty_figure(n, T_steps)
    update_figure(st.session_state.fig, [initial_pauli])
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        key=f"initial_load_{current_hash[:8]}"
    )
    st.session_state.pauli_strings = [initial_pauli]
    st.session_state.states = [initial_state.copy()]
    st.session_state.global_operator = global_operator
    st.session_state.target_steps = T_steps
    st.session_state.params_hash = current_hash
    st.session_state.simulation_running = True
    st.session_state.initialized = True
