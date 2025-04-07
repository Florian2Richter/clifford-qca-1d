import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qca.core import build_global_operator, simulate_QCA, pauli_string_to_state
from qca.visualization import pauli_to_numeric, plot_spacetime
from matplotlib.colors import ListedColormap

st.title("1D Clifford QCA Simulator")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")
n = st.sidebar.number_input("Number of cells", min_value=3, value=31, step=1)
T_steps = st.sidebar.number_input("Number of time steps", min_value=1, value=20, step=1)

# Sidebar for local rule matrix input
st.sidebar.header("Local Rule Matrix (2x6 over F2)")
st.sidebar.write("Enter each row as 6 numbers (0 or 1) separated by spaces.")

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
st.sidebar.header("Initial State")
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
elif init_option == "Random":
    # Generate a random Pauli string of length n from I, X, Z, Y.
    choices = ['I', 'X', 'Z', 'Y']
    random_pauli = ''.join(np.random.choice(choices, size=n))
    st.sidebar.write("Random initial state:", random_pauli)
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

st.subheader("Spacetime Diagram")
fig = plot_spacetime(pauli_strings, n, return_fig=True)
st.pyplot(fig)
