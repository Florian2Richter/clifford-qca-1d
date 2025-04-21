import streamlit as st
import numpy as np
from qca.core import build_global_operator, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric, make_empty_figure, update_figure
import hashlib

# Global constants
BATCH_SIZE = 5

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="1D Clifford QCA Simulator",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add version indicator to verify deployment
    st.sidebar.markdown("**App Version: 2025-04-20.3 (Matrix UI Fixed)**")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header { font-size:2.5rem; color:#1E88E5; text-align:center; margin-bottom:1rem; }
        .sub-header { font-size:1.5rem; color:#424242; margin-top:1.5rem; margin-bottom:1rem; }
        .description { font-size:1rem; color:#616161; margin-bottom:1.5rem; }
        .sidebar-header { font-size:24px !important; font-weight:bold !important; margin-top:1rem !important; }
        .stMetric { background-color:#f0f2f6; padding:10px; border-radius:5px; }
        /* Matrix styling */
        .matrix-container { margin-bottom: 1rem; padding: 8px; border-radius: 5px; background-color: #f5f7f9; }
        .matrix-label { font-weight: bold; margin-bottom: 5px; font-size: 16px; color: #333; }
        [data-testid="stNumberInput"] { margin-bottom: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize the session state if it doesn't exist."""
    if 'initialized' not in st.session_state:
        st.session_state.update({
            "initialized": False,
            "current_step": 0,
            "pauli_strings": [],
            "states": [],
            "global_operator": None,
            "params_hash": "",
            "target_steps": 0,
            "simulation_running": False,
            "simulation_complete": False,
            "fig": None
        })

def setup_ui_elements():
    """Set up all UI elements and return input parameters."""
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
    st.sidebar.markdown("""
    <a href="https://florian2richter.github.io/2025/04/15/what-is-cellular-automata.html" target="_blank">What am I seeing here?</a>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<h3 class="sidebar-header">Simulation Parameters</h3>', unsafe_allow_html=True)
    
    # Create two columns for simulation parameters
    col1, col2 = st.sidebar.columns(2)
    n = col1.number_input("Number of cells", min_value=3, value=500, step=1)
    T_steps = col2.number_input("Time steps", min_value=1, value=250, step=1)
    
    # Local rule matrix input
    st.sidebar.markdown('<h3 class="sidebar-header">Choose your Local Rule Matrices</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="description">
    Enter values (0 or 1) for each cell in the three matrices.
    These matrices define how each cell updates based on its left neighbor (M-1), 
    its own state (M0), and its right neighbor (M1).
    </div>
    """, unsafe_allow_html=True)
    
    # Matrix headers in one row
    mat_headers = st.sidebar.columns(3)
    mat_headers[0].markdown("<div style='text-align: center; font-weight: bold;'>M-1 (Left)</div>", unsafe_allow_html=True)
    mat_headers[1].markdown("<div style='text-align: center; font-weight: bold;'>M0 (Center)</div>", unsafe_allow_html=True)
    mat_headers[2].markdown("<div style='text-align: center; font-weight: bold;'>M1 (Right)</div>", unsafe_allow_html=True)
    
    # Create matrices with shared structure
    m_left = np.zeros((2, 2), dtype=int)
    m_center = np.zeros((2, 2), dtype=int)
    m_right = np.zeros((2, 2), dtype=int)
    
    # First row of all matrices
    row1 = st.sidebar.columns(6)
    m_left[0, 0] = row1[0].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_left_00", label_visibility="collapsed")
    m_left[0, 1] = row1[1].number_input("", min_value=0, max_value=1, value=0, step=1, key="m_left_01", label_visibility="collapsed")
    m_center[0, 0] = row1[2].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_center_00", label_visibility="collapsed")
    m_center[0, 1] = row1[3].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_center_01", label_visibility="collapsed")
    m_right[0, 0] = row1[4].number_input("", min_value=0, max_value=1, value=0, step=1, key="m_right_00", label_visibility="collapsed")
    m_right[0, 1] = row1[5].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_right_01", label_visibility="collapsed")
    
    # Second row of all matrices
    row2 = st.sidebar.columns(6)
    m_left[1, 0] = row2[0].number_input("", min_value=0, max_value=1, value=0, step=1, key="m_left_10", label_visibility="collapsed")
    m_left[1, 1] = row2[1].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_left_11", label_visibility="collapsed")
    m_center[1, 0] = row2[2].number_input("", min_value=0, max_value=1, value=0, step=1, key="m_center_10", label_visibility="collapsed")
    m_center[1, 1] = row2[3].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_center_11", label_visibility="collapsed")
    m_right[1, 0] = row2[4].number_input("", min_value=0, max_value=1, value=1, step=1, key="m_right_10", label_visibility="collapsed")
    m_right[1, 1] = row2[5].number_input("", min_value=0, max_value=1, value=0, step=1, key="m_right_11", label_visibility="collapsed")
    
    # Convert the three matrices to the required local rule format
    local_rule = matrices_to_local_rule(m_left, m_center, m_right)
    
    # Initial state selection
    st.sidebar.markdown('<h3 class="sidebar-header">Initial State</h3>', unsafe_allow_html=True)
    init_option = st.sidebar.selectbox("Choose initial state:", ["Single active cell", "Random", "Manual"])
    
    initial_state = create_initial_state(init_option, n)
    
    # Create placeholders for plot and status
    plot_placeholder = st.empty()
    status_placeholder = st.empty()
    
    return n, T_steps, local_rule, initial_state, plot_placeholder, status_placeholder

def parse_local_rule(row1_input, row2_input):
    """Parse and validate the local rule matrix input."""
    row1 = parse_matrix_row(row1_input)
    row2 = parse_matrix_row(row2_input)
    
    if row1 is not None and row2 is not None:
        return np.array([row1, row2], dtype=int) % 2
    else:
        st.stop()

def parse_matrix_row(row_str):
    """Parse a matrix row from a string input."""
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

def matrices_to_local_rule(m_left, m_center, m_right):
    """
    Convert three 2x2 matrices to the required 2x6 local rule format.
    
    Parameters:
    -----------
    m_left : np.ndarray
        2x2 matrix for the left neighbor (M_{-1}).
    m_center : np.ndarray
        2x2 matrix for the center cell (M_{0}).
    m_right : np.ndarray
        2x2 matrix for the right neighbor (M_{1}).
    
    Returns:
    --------
    np.ndarray
        2x6 local rule matrix in the required format.
    """
    # Initialize the 2x6 local rule matrix
    local_rule = np.zeros((2, 6), dtype=int)
    
    # First row: [m_left[0,0], m_left[0,1], m_center[0,0], m_center[0,1], m_right[0,0], m_right[0,1]]
    local_rule[0, 0] = m_left[0, 0]
    local_rule[0, 1] = m_left[0, 1]
    local_rule[0, 2] = m_center[0, 0]
    local_rule[0, 3] = m_center[0, 1]
    local_rule[0, 4] = m_right[0, 0]
    local_rule[0, 5] = m_right[0, 1]
    
    # Second row: [m_left[1,0], m_left[1,1], m_center[1,0], m_center[1,1], m_right[1,0], m_right[1,1]]
    local_rule[1, 0] = m_left[1, 0]
    local_rule[1, 1] = m_left[1, 1]
    local_rule[1, 2] = m_center[1, 0]
    local_rule[1, 3] = m_center[1, 1]
    local_rule[1, 4] = m_right[1, 0]
    local_rule[1, 5] = m_right[1, 1]
    
    return local_rule

def create_initial_state(init_option, n):
    """Create the initial state based on the selected option."""
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
    
    return initial_state

def get_single_active_state(n):
    """Create a state with a single X operator at the center."""
    state = np.zeros(2 * n, dtype=int)
    center = n // 2
    state[2*center] = 1
    return state

def get_params_hash(n, T_steps, local_rule, initial_state):
    """Create a hash of all parameters to detect changes."""
    hash_str = f"{n}_{T_steps}_{local_rule.tobytes().hex()}_{initial_state.tobytes().hex()}"
    return hashlib.md5(hash_str.encode()).hexdigest()

def handle_parameter_changes(n, T_steps, local_rule, initial_state, current_hash):
    """Handle changes in simulation parameters."""
    if current_hash != st.session_state.params_hash:
        st.session_state.params_hash = current_hash
        st.session_state.current_step = 0
        st.session_state.pauli_strings = [vector_to_pauli_string(initial_state)]
        st.session_state.states = [initial_state.copy()]
        st.session_state.global_operator = build_global_operator(n, local_rule)
        st.session_state.target_steps = T_steps
        st.session_state.simulation_running = True
        st.session_state.simulation_complete = False
        st.session_state.initialized = True
        st.session_state.fig = None

def calculate_step(current_state):
    """Calculate the next state for the QCA simulation."""
    next_state = mod2_matmul(st.session_state.global_operator, current_state) % 2
    next_pauli = vector_to_pauli_string(next_state)
    return next_state, next_pauli

def run_simulation(n, plot_placeholder, status_placeholder, current_hash):
    """Run the progressive simulation."""
    if st.session_state.current_step < st.session_state.target_steps:
        # Create the figure once on first batch
        if st.session_state.fig is None:
            st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
            plot_placeholder.plotly_chart(
                st.session_state.fig,
                use_container_width=False,
                config=getattr(st.session_state.fig, '_config', None),
                key=f"init_plot_{current_hash[:8]}",
                theme=None
            )
        
        for step in range(st.session_state.current_step, st.session_state.target_steps):
            next_state, next_pauli = calculate_step(st.session_state.states[-1])
            st.session_state.states.append(next_state.copy())
            st.session_state.pauli_strings.append(next_pauli)
            st.session_state.current_step += 1

            if (st.session_state.current_step % BATCH_SIZE == 0 or 
                    st.session_state.current_step == st.session_state.target_steps):
                # Update the figure
                st.session_state.fig = update_figure(st.session_state.fig, st.session_state.pauli_strings)
                plot_placeholder.plotly_chart(
                    st.session_state.fig,
                    use_container_width=False,
                    config=getattr(st.session_state.fig, '_config', None),
                    key=f"step_{st.session_state.current_step}_{current_hash[:8]}",
                    theme=None
                )
        
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")
        
    elif not st.session_state.simulation_complete:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")

def display_results(n, plot_placeholder, current_hash):
    """Display the final simulation results."""
    # Safety check if fig doesn't exist for some reason
    if "fig" not in st.session_state or st.session_state.fig is None:
        st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
        st.session_state.fig = update_figure(st.session_state.fig, st.session_state.pauli_strings)
    
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"final_plot_{current_hash[:8]}",
        theme=None
    )

def handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash):
    """Handle the initial load of the application."""
    initial_pauli = vector_to_pauli_string(initial_state)
    
    # Create the figure once
    st.session_state.fig = make_empty_figure(n, T_steps)
    st.session_state.fig = update_figure(st.session_state.fig, [initial_pauli])
    
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"initial_load_{current_hash[:8]}",
        theme=None
    )
    
    st.session_state.pauli_strings = [initial_pauli]
    st.session_state.states = [initial_state.copy()]
    st.session_state.global_operator = global_operator
    st.session_state.target_steps = T_steps
    st.session_state.params_hash = current_hash
    st.session_state.simulation_running = True
    st.session_state.initialized = True

# Add a decorator for caching
@st.cache_data(ttl=900, show_spinner=False)
def build_cached_global_operator(n, local_rule):
    """Cached version of build_global_operator to improve performance."""
    return build_global_operator(n, local_rule)

def main():
    """Main function to run the application."""
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    initialize_session_state()
    
    # Setup UI elements
    n, T_steps, local_rule, initial_state, plot_placeholder, status_placeholder = setup_ui_elements()
    
    # Build the global operator (use cached version for better performance)
    global_operator = build_cached_global_operator(n, local_rule)
    
    # Calculate parameters hash
    current_hash = get_params_hash(n, T_steps, local_rule, initial_state)
    
    # Handle parameter changes
    handle_parameter_changes(n, T_steps, local_rule, initial_state, current_hash)
    
    # App execution flow
    with st.spinner("Processing simulation..."):
        if st.session_state.initialized and st.session_state.simulation_running:
            run_simulation(n, plot_placeholder, status_placeholder, current_hash)
        elif st.session_state.simulation_complete:
            display_results(n, plot_placeholder, current_hash)
        elif not st.session_state.initialized:
            handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash)

# Run the application
if __name__ == "__main__":
    main()
