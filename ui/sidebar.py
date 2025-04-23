import streamlit as st
import numpy as np
from ui.page_config import display_app_version
from simulation.core import matrices_to_local_rule, create_initial_state_custom

def setup_sidebar():
    """
    Set up the sidebar UI components.
    
    Returns:
    --------
    tuple
        (n, T_steps, local_rule, initial_state) - Parameters for the simulation
    """
    # About section first
    display_about_section()
    
    # Add presets dropdown
    presets = get_preset_configurations()
    selected_preset = display_preset_section(presets)
    
    # Matrix input section for local rule
    m_left, m_center, m_right = display_matrix_input_section(presets, selected_preset)
    
    # Convert the matrices to the required local rule format
    local_rule = matrices_to_local_rule(m_left, m_center, m_right)
    
    # Initial state section
    operators, positions = display_initial_state_section(presets, selected_preset)
    
    # Simulation parameters section (moved to bottom of sidebar)
    n, T_steps = display_simulation_parameters()
    
    # Create initial state based on operators and positions
    initial_state = create_initial_state_custom(n, operators, positions)
    
    # Add version indicator at the bottom of the sidebar
    display_app_version()
    
    return n, T_steps, local_rule, initial_state

def display_about_section():
    """Display the 'What is this App about?' section in the sidebar."""
    st.sidebar.markdown('<h3 class="sidebar-header">What is this App about?</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    Hi! Whether you're deep into mathematics and quantum theory — or just here for the eye candy — you're in the right place.

    This app lets you explore a 1D Clifford Quantum Cellular Automaton. You can tweak the rules, lean back, and watch the system evolve into beautiful, fractal-like patterns. Just for fun? Export your favorite result as a high-resolution wallpaper!

    Curious what's really going on under the hood?
    Take a dive into the quantum depths <a href="https://florian2richter.github.io/2025/04/15/what-is-cellular-automata.html" target="_blank">in this blog post</a>, where I explain the science behind Clifford QCAs and how they work.

    Have fun exploring — whatever your angle!
    """, unsafe_allow_html=True)

def get_preset_configurations():
    """Define preset configurations for the QCA simulation."""
    return {
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
                "positions": [250]
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
                "positions": [250, 251]
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

def display_preset_section(presets):
    """
    Display the preset configurations section.
    
    Parameters:
    -----------
    presets : dict
        Dictionary of preset configurations
        
    Returns:
    --------
    str
        Name of the selected preset
    """
    st.sidebar.markdown('<h3 class="sidebar-header">Preset Configurations</h3>', unsafe_allow_html=True)
    
    # Select a preset
    selected_preset = st.sidebar.selectbox(
        "Choose a preset configuration:",
        list(presets.keys())
    )
    
    if selected_preset != "Custom":
        st.sidebar.info(presets[selected_preset]["description"])
    
    return selected_preset

def display_matrix_input_section(presets, selected_preset):
    """
    Display the matrix input section for local rule configuration.
    
    Parameters:
    -----------
    presets : dict
        Dictionary of preset configurations
    selected_preset : str
        Name of the selected preset
        
    Returns:
    --------
    tuple
        (m_left, m_center, m_right) - The 2x2 matrices for the local rule
    """
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
    
    return m_left, m_center, m_right

def display_initial_state_section(presets, selected_preset):
    """
    Display the initial state section in the sidebar.
    
    Parameters:
    -----------
    presets : dict
        Dictionary of preset configurations
    selected_preset : str
        Name of the selected preset
        
    Returns:
    --------
    tuple
        (operators, positions) - Lists of operators and their positions
    """
    st.sidebar.markdown('<h3 class="sidebar-header">Initial State</h3>', unsafe_allow_html=True)
    
    # Number of non-identity operators
    if selected_preset != "Custom":
        preset_num_operators = presets[selected_preset]["initial_state"]["num_operators"]
        preset_operators = presets[selected_preset]["initial_state"]["operators"]
        preset_positions = presets[selected_preset]["initial_state"]["positions"]
        num_operators = st.sidebar.number_input("Number of non-identity operators", 
                                              min_value=1, max_value=500, value=preset_num_operators, step=1)
    else:
        num_operators = st.sidebar.number_input("Number of non-identity operators", 
                                              min_value=1, max_value=500, value=1, step=1)
    
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
            default_pos = 250 if i == 0 else 0
        
        # Create operator and position inputs
        operator = op_row[0].selectbox(f"Operator {i+1}", 
                                      options=["X", "Y", "Z"], 
                                      index=["X", "Y", "Z"].index(default_op), 
                                      key=f"op_{i}")
        position = op_row[1].number_input(f"Position {i+1}", 
                                        min_value=0, max_value=499, 
                                        value=default_pos, 
                                        key=f"pos_{i}")
        operators.append(operator)
        positions.append(position)
    
    return operators, positions

def display_simulation_parameters():
    """
    Display the simulation parameters section.
    
    Returns:
    --------
    tuple
        (n, T_steps) - Number of cells and time steps
    """
    st.sidebar.markdown('<h3 class="sidebar-header">Simulation Parameters</h3>', unsafe_allow_html=True)
    
    # Create two columns for simulation parameters
    col1, col2 = st.sidebar.columns(2)
    n = col1.number_input("Number of cells", min_value=3, value=500, step=1)
    T_steps = col2.number_input("Time steps", min_value=1, value=250, step=1)
    
    return n, T_steps 