import streamlit as st
import numpy as np
from simulation.core import (
    build_global_operator, 
    pauli_string_to_state, 
    vector_to_pauli_string, 
    mod2_matmul,
    calculate_step, 
    build_cached_global_operator,
    matrices_to_local_rule,
    create_initial_state_custom,
    get_params_hash,
    handle_parameter_changes
)
from qca.visualization import make_empty_figure, update_figure, pauli_strings_to_numeric, generate_hires_plot
from ui.page_config import setup_page_config
from ui.sidebar import setup_sidebar
from ui.main_view import setup_main_view, run_simulation, display_results, handle_initial_load
from config import BATCH_SIZE
import io
import plotly.graph_objects as go

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

def parse_local_rule(row1_input, row2_input):
    """
    Parse input rows into an array for the local rule.
    
    Parameters:
    -----------
    row1_input, row2_input : str
        Comma-separated 0s and 1s
        
    Returns:
    --------
    numpy.ndarray
        A 2x6 array representing the local rule
    """
    row1 = parse_matrix_row(row1_input)
    row2 = parse_matrix_row(row2_input)
    
    # Ensure correct length
    row1 = row1[:6] + [0] * max(0, 6 - len(row1))
    row2 = row2[:6] + [0] * max(0, 6 - len(row2))
    
    return np.array([row1, row2])

def parse_matrix_row(row_str):
    """
    Parse a comma-separated string of 0s and 1s into a list of integers.
    
    Parameters:
    -----------
    row_str : str
        Comma-separated string of 0s and 1s.
    
    Returns:
    --------
    list
        List of integers (0s and 1s).
    """
    try:
        # Remove spaces and split by commas
        values = [int(x.strip()) for x in row_str.replace(' ', '').split(',')]
        
        # Validate that all values are 0 or 1
        for val in values:
            if val not in [0, 1]:
                st.sidebar.error("Only 0 and 1 are allowed.")
                return None
        return values
    except Exception as e:
        st.sidebar.error("Error parsing the row: " + str(e))
        return None

def main():
    """Main function to run the application."""
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    initialize_session_state()
    
    # Setup UI elements
    n, T_steps, local_rule, initial_state = setup_sidebar()
    plot_placeholder, status_placeholder = setup_main_view()
    
    # Build the global operator (use cached version)
    global_operator = build_cached_global_operator(n, local_rule)
    
    # Calculate parameters hash
    current_hash = get_params_hash(n, T_steps, local_rule, initial_state)
    
    # Handle parameter changes
    handle_parameter_changes(n, T_steps, local_rule, initial_state, current_hash)
    
    # App execution flow
    with st.spinner("Processing simulation..."):
        # Check completion first, then check if running, finally handle initial state
        if st.session_state.simulation_complete:
            display_results(n, plot_placeholder, current_hash)
        elif st.session_state.initialized and st.session_state.simulation_running:
            run_simulation(n, plot_placeholder, status_placeholder, current_hash)
        elif not st.session_state.initialized:
            handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash)

# Run the application
if __name__ == "__main__":
    main()
