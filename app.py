import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qca.core import build_global_operator, simulate_QCA
from qca.visualization import pauli_to_numeric
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

def get_initial_state(n):
    # Create a state vector of length 2*n: first n for x-part, next n for z-part.
    state = np.zeros(2 * n, dtype=int)
    center = n // 2
    state[center] = 1  # Single active cell: set the x-part at the center to 1.
    return state

if init_option == "Single active cell":
    initial_state = get_initial_state(n)
elif init_option == "Random":
    initial_state = np.random.randint(0, 2, size=2 * n)
elif init_option == "Manual":
    manual_x = st.sidebar.text_input("X-part (binary string of length n)", "0" * (n // 2) + "1" + "0" * (n - n // 2 - 1))
    manual_z = st.sidebar.text_input("Z-part (binary string of length n)", "0" * n)
    if len(manual_x) != n or len(manual_z) != n:
        st.sidebar.error("Binary strings must be of length equal to the number of cells.")
        st.stop()
    try:
        x_part = np.array([int(ch) for ch in manual_x], dtype=int)
        z_part = np.array([int(ch) for ch in manual_z], dtype=int)
    except:
        st.sidebar.error("Invalid binary string. Use only 0 and 1.")
        st.stop()
    initial_state = np.concatenate([x_part, z_part])
else:
    initial_state = get_initial_state(n)

# Build the global operator from the local rule.
global_operator = build_global_operator(n, local_rule)

# Run the simulation.
states, pauli_strings = simulate_QCA(n, T_steps, initial_state, global_operator)

st.subheader("Spacetime Diagram")
# Create a spacetime diagram using matplotlib.
time_steps = len(pauli_strings)
data = np.array([np.array([pauli_to_numeric(s)[i] for i in range(n)]) for s in pauli_strings])

cmap = ListedColormap(["white", "red", "blue", "green"])
fig, ax = plt.subplots(figsize=(0.5 * n, 0.5 * time_steps))
cax = ax.imshow(data, cmap=cmap, interpolation="nearest", aspect="auto")
ax.set_xlabel("Cell position")
ax.set_ylabel("Time step")
ax.set_title("1D Clifford QCA Spacetime Diagram")
fig.colorbar(cax, ticks=[0, 1, 2, 3], label="Pauli")
st.pyplot(fig)

st.subheader("Pauli Strings per Time Step")
for t, s in enumerate(pauli_strings):
    st.text(f"Time {t}: {s}")