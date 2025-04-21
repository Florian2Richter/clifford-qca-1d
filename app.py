import streamlit as st
import numpy as np
from qca.core import build_global_operator, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric, make_empty_figure, update_figure, pauli_strings_to_numeric
import hashlib
import io
import plotly.graph_objects as go

# Global constants
BATCH_SIZE = 10

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="1D Clifford QCA Simulator",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add version indicator to verify deployment
    st.sidebar.markdown("**App Version: 2025-04-21.7 (Added FullHD Export)**")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header { font-size:2.5rem; color:#1E88E5; text-align:center; margin-bottom:1rem; }
        .sub-header { font-size:1.5rem; color:#424242; margin-top:1.5rem; margin-bottom:1rem; }
        .description { font-size:1rem; color:#616161; margin-bottom:1.5rem; }
        .sidebar-header { font-size:24px !important; font-weight:bold !important; margin-top:1rem !important; }
        .stMetric { background-color:#f0f2f6; padding:10px; border-radius:5px; }
        
        /* Improved selectbox styling */
        [data-testid="stSelectbox"] {
            margin-bottom: 0 !important;
        }
        
        /* Make the dropdown text more visible */
        .st-bi {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
        }
        
        /* Style for the matrix containers */
        .matrix-section {
            background-color: #f5f7f9;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Matrix background colors */
        .matrix-left {
            background-color: rgba(200, 230, 255, 0.3);
            border-radius: 5px;
            padding: 5px;
        }
        
        .matrix-center {
            background-color: rgba(230, 255, 200, 0.3);
            border-radius: 5px;
            padding: 5px;
        }
        
        .matrix-right {
            background-color: rgba(255, 230, 200, 0.3);
            border-radius: 5px;
            padding: 5px;
        }
        
        /* Enhance dropdown appearance */
        [data-testid="stSelectbox"] > div > div > div {
            font-weight: bold !important;
            text-align: center !important;
            font-size: 16px !important;
        }
        
        /* Better number input styling */
        [data-testid="stNumberInput"] > div {
            flex-direction: row !important;
        }
        
        [data-testid="stNumberInput"] input {
            text-align: center !important;
            font-size: 16px !important;
            font-weight: bold !important;
            width: 50px !important;
        }
        
        /* Improve button appearance */
        button.step-down, button.step-up {
            height: 30px !important;
            width: 30px !important;
            background-color: #f0f2f6 !important;
        }
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
    
    # Add presets dropdown at the top of the sidebar
    st.sidebar.markdown('<h3 class="sidebar-header">Preset Configurations</h3>', unsafe_allow_html=True)
    
    # Define preset configurations
    presets = {
        "Custom": {
            "description": "Custom configuration (current settings)",
            "matrices": {
                "m_left": np.array([[1, 0], [0, 1]]),
                "m_center": np.array([[1, 1], [0, 1]]),
                "m_right": np.array([[0, 1], [1, 0]])
            },
            "initial_state": {
                "num_operators": 1,
                "operators": ["X"],
                "positions": [n//2]
            }
        },
        "Glider": {
            "description": "A 'glider' pattern that propagates through the lattice",
            "matrices": {
                "m_left": np.array([[0, 0], [0, 1]]),
                "m_center": np.array([[0, 1], [1, 0]]),
                "m_right": np.array([[0, 0], [0, 1]])
            },
            "initial_state": {
                "num_operators": 2,
                "operators": ["X", "Z"],
                "positions": [n//2, n//2 + 1]
            }
        },
        "Fractal": {
            "description": "A pattern that creates self-similar fractal structures",
            "matrices": {
                "m_left": np.array([[1, 0], [0, 0]]),
                "m_center": np.array([[1, 1], [1, 0]]),
                "m_right": np.array([[1, 0], [0, 0]])
            },
            "initial_state": {
                "num_operators": 1,
                "operators": ["X"],
                "positions": [250]
            }
        }
    }
    
    # Select a preset
    selected_preset = st.sidebar.selectbox(
        "Choose a preset configuration:",
        list(presets.keys())
    )
    
    if selected_preset != "Custom":
        st.sidebar.info(presets[selected_preset]["description"])
    
    # Local rule matrix input
    st.sidebar.markdown('<h3 class="sidebar-header">Choose your Local Rule Matrices</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="description">
    Enter values (0 or 1) for each cell in the three matrices.
    These matrices define how each cell updates based on its left neighbor (M-1), 
    its own state (M0), and its right neighbor (M1).
    </div>
    """, unsafe_allow_html=True)
    
    # Create matrices in a simple grid layout
    if selected_preset != "Custom":
        # Use matrices from preset
        m_left = presets[selected_preset]["matrices"]["m_left"].copy()
        m_center = presets[selected_preset]["matrices"]["m_center"].copy()
        m_right = presets[selected_preset]["matrices"]["m_right"].copy()
    else:
        # Use default matrices from Custom preset
        m_left = presets["Custom"]["matrices"]["m_left"].copy()
        m_center = presets["Custom"]["matrices"]["m_center"].copy()
        m_right = presets["Custom"]["matrices"]["m_right"].copy()
    
    # Matrix headers
    st.sidebar.markdown("<div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>"
                      "<div style='width: 30%; text-align: center; font-weight: bold;' class='matrix-left'>M-1 (Left)</div>"
                      "<div style='width: 30%; text-align: center; font-weight: bold;' class='matrix-center'>M0 (Center)</div>"
                      "<div style='width: 30%; text-align: center; font-weight: bold;' class='matrix-right'>M1 (Right)</div>"
                      "</div>", unsafe_allow_html=True)
    
    # First row of matrices - use 6 columns side by side
    row1 = st.sidebar.columns(6)
    m_left[0, 0] = row1[0].selectbox("", options=[0, 1], index=int(m_left[0, 0]), key="m_left_00", label_visibility="collapsed")
    m_left[0, 1] = row1[1].selectbox("", options=[0, 1], index=int(m_left[0, 1]), key="m_left_01", label_visibility="collapsed")
    m_center[0, 0] = row1[2].selectbox("", options=[0, 1], index=int(m_center[0, 0]), key="m_center_00", label_visibility="collapsed")
    m_center[0, 1] = row1[3].selectbox("", options=[0, 1], index=int(m_center[0, 1]), key="m_center_01", label_visibility="collapsed")
    m_right[0, 0] = row1[4].selectbox("", options=[0, 1], index=int(m_right[0, 0]), key="m_right_00", label_visibility="collapsed")
    m_right[0, 1] = row1[5].selectbox("", options=[0, 1], index=int(m_right[0, 1]), key="m_right_01", label_visibility="collapsed")
    
    # Second row of matrices - use 6 columns side by side
    row2 = st.sidebar.columns(6)
    m_left[1, 0] = row2[0].selectbox("", options=[0, 1], index=int(m_left[1, 0]), key="m_left_10", label_visibility="collapsed")
    m_left[1, 1] = row2[1].selectbox("", options=[0, 1], index=int(m_left[1, 1]), key="m_left_11", label_visibility="collapsed")
    m_center[1, 0] = row2[2].selectbox("", options=[0, 1], index=int(m_center[1, 0]), key="m_center_10", label_visibility="collapsed")
    m_center[1, 1] = row2[3].selectbox("", options=[0, 1], index=int(m_center[1, 1]), key="m_center_11", label_visibility="collapsed")
    m_right[1, 0] = row2[4].selectbox("", options=[0, 1], index=int(m_right[1, 0]), key="m_right_10", label_visibility="collapsed")
    m_right[1, 1] = row2[5].selectbox("", options=[0, 1], index=int(m_right[1, 1]), key="m_right_11", label_visibility="collapsed")
    
    # Convert the three matrices to the required local rule format
    local_rule = matrices_to_local_rule(m_left, m_center, m_right)
    
    # Initial state selection
    st.sidebar.markdown('<h3 class="sidebar-header">Initial State</h3>', unsafe_allow_html=True)
    
    # Number of non-identity operators
    if selected_preset != "Custom":
        preset_num_operators = presets[selected_preset]["initial_state"]["num_operators"]
        preset_operators = presets[selected_preset]["initial_state"]["operators"]
        preset_positions = presets[selected_preset]["initial_state"]["positions"]
        num_operators = st.sidebar.number_input("Number of non-identity operators", 
                                              min_value=1, max_value=n, value=preset_num_operators, step=1)
    else:
        num_operators = st.sidebar.number_input("Number of non-identity operators", 
                                              min_value=1, max_value=n, value=1, step=1)
    
    # Initialize operator list and positions
    operators = []
    positions = []
    
    # Generate UI for each operator
    for i in range(int(num_operators)):
        op_row = st.sidebar.columns(2)
        
        # Set default values from preset if applicable
        if selected_preset != "Custom" and i < len(preset_operators):
            default_op = preset_operators[i]
            default_pos = preset_positions[i]
        else:
            default_op = "X"
            default_pos = n//2 if i == 0 else 0
        
        # Create operator and position inputs
        operator = op_row[0].selectbox(f"Operator {i+1}", 
                                      options=["X", "Y", "Z"], 
                                      index=["X", "Y", "Z"].index(default_op), 
                                      key=f"op_{i}")
        position = op_row[1].number_input(f"Position {i+1}", 
                                        min_value=0, max_value=n-1, 
                                        value=default_pos, 
                                        key=f"pos_{i}")
        operators.append(operator)
        positions.append(position)
    
    # Create initial state based on operators and positions
    initial_state = create_initial_state_custom(n, operators, positions)
    
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
    
    # Add high-resolution download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Export as FullHD Wallpaper (1920×1080)", use_container_width=True):
            with st.spinner("Generating high-resolution image..."):
                try:
                    wallpaper_bytes = generate_hires_plot(st.session_state.pauli_strings)
                    if wallpaper_bytes:
                        # Create a download button with the generated image
                        st.download_button(
                            label="Download FullHD Wallpaper",
                            data=wallpaper_bytes,
                            file_name=f"qca_wallpaper_{current_hash[:6]}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        st.success("Image generated! Click the button above to download.")
                    else:
                        st.error("Could not generate image. Please try again.")
                except Exception as e:
                    st.error(f"Error generating high-resolution image: {str(e)}")
                    raise e

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

def generate_hires_plot(pauli_strings, width=1920, height=1080):
    """
    Generate a high-resolution (FullHD) plot of the QCA simulation for wallpaper use.
    
    Parameters:
    -----------
    pauli_strings : list of str
        List of Pauli strings representing the state at each calculated time step.
    width : int
        Width of the output image in pixels (default: 1920 for FullHD).
    height : int
        Height of the output image in pixels (default: 1080 for FullHD).
        
    Returns:
    --------
    bytes
        The PNG image as bytes for download.
    """
    # Get dimensions
    time_steps = len(pauli_strings)
    cell_count = len(pauli_strings[0]) if pauli_strings else 0
    
    if time_steps == 0 or cell_count == 0:
        return None
    
    # Convert strings to numeric data using our mapping (I->0, X->1, Z->2, Y->3)
    numeric_data = pauli_strings_to_numeric(pauli_strings)
    
    # Define colors for each Pauli operator (same as in visualization.py)
    color_mapping = [
        [0.0, 'white'],      # I (value 0)
        [0.33, '#008080'],   # X (value 1)
        [0.67, '#FF7F50'],   # Z (value 2)
        [1.0, '#4A4A4A']     # Y (value 3)
    ]
    
    # Create the heatmap with optimized appearance for wallpaper
    fig = go.Figure(data=go.Heatmap(
        z=numeric_data,
        colorscale=color_mapping,
        zmin=0,
        zmax=3,
        showscale=False,  # No colorbar for clean wallpaper
        hoverinfo='none'  # No hover for static image
    ))
    
    # Set clean layout with no axes or labels
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            visible=False
        ),
        xaxis=dict(visible=False)
    )
    
    # Convert to PNG bytes
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
    
    return img_bytes

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
