import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state
from qca.visualization import pauli_to_numeric, plot_spacetime_plotly
from matplotlib.colors import ListedColormap

# Set page configuration
st.set_page_config(
    page_title="1D Clifford QCA Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add version indicator to verify deployment
st.sidebar.markdown("**App Version: 2023-04-13-reset**")

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
n = st.sidebar.number_input("Number of cells", min_value=3, value=50, step=1)
T_steps = st.sidebar.number_input("Number of time steps", min_value=1, value=25, step=1)

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

# Build the global operator from the local rule.
global_operator = build_global_operator(n, local_rule)

# Run the simulation.
states, pauli_strings = simulate_QCA(n, T_steps, initial_state, global_operator)

# Display the spacetime diagram using Plotly
st.markdown('<h2 class="sub-header">Spacetime Diagram</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="description">
The diagram shows the evolution of Pauli operators (I, X, Z, Y) over time in the quantum cellular automaton. You can zoom, pan, and hover over cells.
</div>
""", unsafe_allow_html=True)

# Create the plot using the Plotly function
fig = plot_spacetime_plotly(pauli_strings)
# Display the Plotly figure using st.plotly_chart
st.plotly_chart(fig, use_container_width=True)
