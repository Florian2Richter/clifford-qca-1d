import streamlit as st
import numpy as np
from qca.core import build_global_operator, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import make_empty_figure, update_figure, pauli_strings_to_numeric, generate_hires_plot
from ui.page_config import setup_page_config
from ui.sidebar import setup_sidebar
from ui.main_view import setup_main_view, run_simulation, display_results, handle_initial_load
import hashlib
import io
import plotly.graph_objects as go

# Global constants
BATCH_SIZE = 5

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

def create_initial_state_custom(n, operators, positions):
    """
    Create the initial state based on the selected operators and positions.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    operators : list of str
        List of Pauli operators ('X', 'Y', 'Z') to place.
    positions : list of int
        Positions where to place the operators (modulo n for periodic boundaries).
    
    Returns:
    --------
    state : numpy.ndarray
        The initial state vector.
    """
    # Create a Pauli string with all 'I' operators
    pauli_string = ['I'] * n
    
    # Place selected operators at their positions (with modulo for periodic boundaries)
    for op, pos in zip(operators, positions):
        pos = pos % n  # Apply modulo for periodic boundaries
        pauli_string[pos] = op
    
    # Convert Pauli string to a state vector
    pauli_str = ''.join(pauli_string)
    state = pauli_string_to_state(pauli_str)
    
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
        
        # Always set these flags at the end of simulation
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        
        # Clear the status placeholder before adding success message
        status_placeholder.empty()
        
        # Force a rerun to ensure display_results is shown
        st.rerun()
        
    elif not st.session_state.simulation_complete:
        # This handles the case where we haven't yet marked the simulation as complete
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.empty()
        # Force a rerun to ensure display_results is shown
        st.rerun()

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
    
    # Add export section with a success message
    st.success("✅ Simulation complete!")
    
    # Create a two-column layout for export options
    col1, col2 = st.columns(2)
    
    # Resolution options dropdown in the second column (no label)
    resolution_options = {
        "1920x1080 (baseline)": (1920, 1080),
        "2560x1440 (QHD)": (2560, 1440),
        "3840x2160 (4K)": (3840, 2160),
        "3440x1440 (ultrawide)": (3440, 1440),
        "1080x1920 (mobile/portrait)": (1080, 1920)
    }
    
    # Export button in the first column
    if col1.button("📥 Export as Wallpaper", use_container_width=True, key="export_button"):
        selected_resolution = col2.selectbox(
            "",
            options=list(resolution_options.keys()),
            index=0,
            key="resolution_selector",
            label_visibility="collapsed"
        )
        
        # Extract width and height from the selected resolution
        width, height = resolution_options[selected_resolution]
        
        with st.spinner(f"Generating {selected_resolution} image..."):
            try:
                wallpaper_bytes = generate_hires_plot(st.session_state.pauli_strings, width, height)
                if wallpaper_bytes:
                    # Parse the resolution for the filename
                    resolution_label = selected_resolution.split(" ")[0]
                    
                    # Create a download button with the generated image
                    st.download_button(
                        label=f"Download {resolution_label} Wallpaper",
                        data=wallpaper_bytes,
                        file_name=f"qca_wallpaper_{resolution_label}_{current_hash[:6]}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.success("Image generated! Click the button above to download.")
                else:
                    st.error("Could not generate image. Please try again.")
            except Exception as e:
                st.error(f"Error generating high-resolution image: {str(e)}")
                st.exception(e)  # Show detailed error information
    else:
        # Show resolution dropdown when button is not yet clicked
        col2.selectbox(
            "",
            options=list(resolution_options.keys()),
            index=0,
            key="resolution_selector",
            label_visibility="collapsed"
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

@st.cache_data(ttl=900, show_spinner=False)
def build_cached_global_operator(n, local_rule):
    """Cached version of build_global_operator."""
    return build_global_operator(n, local_rule)

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
