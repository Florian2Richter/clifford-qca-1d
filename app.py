import streamlit as st
import numpy as np
from qca.core import build_global_operator, pauli_string_to_state, vector_to_pauli_string, mod2_matmul
from qca.visualization import pauli_to_numeric, make_empty_figure, update_figure
import hashlib
import time

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
    st.sidebar.markdown("**App Version: 2025-04-20.1 (optimized rendering)**")
    
    # Custom CSS for better styling and performance optimizations
    st.markdown("""
    <style>
        .main-header { font-size:2.5rem; color:#1E88E5; text-align:center; margin-bottom:1rem; }
        .sub-header { font-size:1.5rem; color:#424242; margin-top:1.5rem; margin-bottom:1rem; }
        .description { font-size:1rem; color:#616161; margin-bottom:1.5rem; }
        .sidebar-header { font-size:24px !important; font-weight:bold !important; margin-top:1rem !important; }
        .stMetric { background-color:#f0f2f6; padding:10px; border-radius:5px; }
        /* Rendering optimizations */
        .element-container { margin-bottom: 0.5rem !important; }
        .stPlotlyChart > div { margin: 0 !important; padding: 0 !important; }
        iframe { border: none !important; }
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
    st.sidebar.markdown('<h3 class="sidebar-header">Simulation Parameters</h3>', unsafe_allow_html=True)
    n = st.sidebar.number_input("Number of cells", min_value=3, value=500, step=1)
    T_steps = st.sidebar.number_input("Number of time steps", min_value=1, value=250, step=1)
    
    # Local rule matrix input
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
    
    local_rule = parse_local_rule(row1_input, row2_input)
    
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
    start_time = time.time()
    next_state = mod2_matmul(st.session_state.global_operator, current_state) % 2
    next_pauli = vector_to_pauli_string(next_state)
    calc_time = time.time() - start_time
    return next_state, next_pauli, calc_time

def run_simulation(n, plot_placeholder, status_placeholder, current_hash):
    """Run the progressive simulation."""
    if 'timing_metrics' not in st.session_state:
        st.session_state.timing_metrics = {
            'calculation': [],
            'plot_update': [],
            'rendering': [],
            'plot_update_detail': []  # Added for detailed metrics
        }
    
    metrics_placeholder = st.empty()
    plot_detail_metrics = st.empty()  # New placeholder for detailed plot update metrics
    
    if st.session_state.current_step < st.session_state.target_steps:
        # Create the figure once on first batch
        if st.session_state.fig is None:
            st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
            render_start = time.time()
            plot_placeholder.plotly_chart(
                st.session_state.fig,
                use_container_width=False,
                config=getattr(st.session_state.fig, '_config', None),
                key=f"init_plot_{current_hash[:8]}",
                theme=None  # Disable theming for better performance
            )
            render_time = time.time() - render_start
            st.session_state.timing_metrics['rendering'].append(render_time)
        
        total_calc_time = 0
        batch_calc_time = 0
        batch_start = time.time()
        
        for step in range(st.session_state.current_step, st.session_state.target_steps):
            next_step = step + 1
            next_state, next_pauli, calc_time = calculate_step(st.session_state.states[-1])
            total_calc_time += calc_time
            batch_calc_time += calc_time
            st.session_state.timing_metrics['calculation'].append(calc_time)
            
            st.session_state.states.append(next_state.copy())
            st.session_state.pauli_strings.append(next_pauli)
            st.session_state.current_step += 1

            if (st.session_state.current_step % BATCH_SIZE == 0 or 
                    st.session_state.current_step == st.session_state.target_steps):
                # Measure update time with detailed metrics
                update_start = time.time()
                
                # Update the figure with the optimized method
                st.session_state.fig, update_details = update_figure(st.session_state.fig, 
                                                                     st.session_state.pauli_strings)
                    
                update_time = time.time() - update_start
                st.session_state.timing_metrics['plot_update'].append(update_time)
                st.session_state.timing_metrics['plot_update_detail'].append(update_details)
                
                # Measure rendering time
                render_start = time.time()
                plot_placeholder.plotly_chart(
                    st.session_state.fig,
                    use_container_width=False,
                    config=getattr(st.session_state.fig, '_config', None),
                    key=f"step_{st.session_state.current_step}_{current_hash[:8]}",
                    theme=None  # Disable theming for better performance
                )
                render_time = time.time() - render_start
                st.session_state.timing_metrics['rendering'].append(render_time)
                
                # Display timing metrics
                batch_time = time.time() - batch_start
                display_timing_metrics(metrics_placeholder, batch_calc_time, update_time, render_time, batch_time)
                
                # Display detailed plot update metrics
                display_plot_update_details(plot_detail_metrics, update_details)
                
                # Reset batch timing
                batch_calc_time = 0
                batch_start = time.time()
        
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")
        
        # Display final timing statistics
        display_final_timing_stats(metrics_placeholder)
        
    elif not st.session_state.simulation_complete:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.success("Simulation complete!")

def display_timing_metrics(placeholder, calc_time, update_time, render_time, total_time):
    """Display current timing metrics."""
    with placeholder.container():
        cols = st.columns(4)
        cols[0].metric("Calculation time", f"{calc_time*1000:.1f} ms", f"{(calc_time/total_time)*100:.1f}%")
        cols[1].metric("Plot update time", f"{update_time*1000:.1f} ms", f"{(update_time/total_time)*100:.1f}%")
        cols[2].metric("Rendering time", f"{render_time*1000:.1f} ms", f"{(render_time/total_time)*100:.1f}%")
        cols[3].metric("Total batch time", f"{total_time*1000:.1f} ms", None)

def display_plot_update_details(placeholder, update_details):
    """Display detailed timing metrics for plot update function."""
    with placeholder.container():
        st.markdown("#### Plot Update Breakdown")
        cols = st.columns(4)
        
        total = update_details.get('total', 0.001)  # Avoid division by zero
        
        cols[0].metric("Data Preparation", 
                      f"{update_details.get('data_preparation', 0)*1000:.2f} ms", 
                      f"{(update_details.get('data_preparation', 0)/total)*100:.1f}%")
        
        cols[1].metric("Customdata Creation", 
                      f"{update_details.get('customdata_creation', 0)*1000:.2f} ms", 
                      f"{(update_details.get('customdata_creation', 0)/total)*100:.1f}%")
        
        cols[2].metric("Figure Update", 
                      f"{update_details.get('figure_update', 0)*1000:.2f} ms", 
                      f"{(update_details.get('figure_update', 0)/total)*100:.1f}%")
        
        cols[3].metric("Total Plot Update", 
                      f"{total*1000:.2f} ms", 
                      None)

def display_final_timing_stats(metrics_placeholder):
    """Display final timing statistics after simulation completes."""
    if 'timing_metrics' in st.session_state:
        metrics = st.session_state.timing_metrics
        calc_avg = sum(metrics['calculation']) / max(len(metrics['calculation']), 1) * 1000
        update_avg = sum(metrics['plot_update']) / max(len(metrics['plot_update']), 1) * 1000
        render_avg = sum(metrics['rendering']) / max(len(metrics['rendering']), 1) * 1000
        total_avg = calc_avg + update_avg + render_avg
        
        with metrics_placeholder.container():
            st.markdown("### Timing Statistics")
            cols = st.columns(4)
            cols[0].metric("Avg Calculation", f"{calc_avg:.1f} ms", f"{(calc_avg/total_avg)*100:.1f}%")
            cols[1].metric("Avg Plot Update", f"{update_avg:.1f} ms", f"{(update_avg/total_avg)*100:.1f}%")
            cols[2].metric("Avg Rendering", f"{render_avg:.1f} ms", f"{(render_avg/total_avg)*100:.1f}%")
            cols[3].metric("Avg Total Time", f"{total_avg:.1f} ms", None)
            
            if len(metrics['calculation']) > 0:
                st.markdown("#### Detailed Breakdown")
                st.markdown(f"**Steps calculated:** {len(metrics['calculation'])}")
                st.markdown(f"**Plot updates:** {len(metrics['plot_update'])}")
                st.markdown(f"**Total calculation time:** {sum(metrics['calculation'])*1000:.1f} ms")
                st.markdown(f"**Total update time:** {sum(metrics['plot_update'])*1000:.1f} ms")
                st.markdown(f"**Total rendering time:** {sum(metrics['rendering'])*1000:.1f} ms")
                
                # Plot Update Detailed Breakdown
                if metrics['plot_update_detail']:
                    st.markdown("#### Plot Update Component Averages")
                    # Calculate averages for each component
                    data_prep_avg = sum(detail.get('data_preparation', 0) for detail in metrics['plot_update_detail']) / len(metrics['plot_update_detail']) * 1000
                    customdata_avg = sum(detail.get('customdata_creation', 0) for detail in metrics['plot_update_detail']) / len(metrics['plot_update_detail']) * 1000
                    figure_update_avg = sum(detail.get('figure_update', 0) for detail in metrics['plot_update_detail']) / len(metrics['plot_update_detail']) * 1000
                    
                    component_cols = st.columns(3)
                    component_cols[0].metric("Data Preparation", f"{data_prep_avg:.2f} ms", f"{(data_prep_avg/update_avg)*100:.1f}%")
                    component_cols[1].metric("Customdata Creation", f"{customdata_avg:.2f} ms", f"{(customdata_avg/update_avg)*100:.1f}%")
                    component_cols[2].metric("Figure Update", f"{figure_update_avg:.2f} ms", f"{(figure_update_avg/update_avg)*100:.1f}%")

def display_results(n, plot_placeholder, current_hash):
    """Display the final simulation results."""
    metrics_placeholder = st.empty()
    
    # Safety check if fig doesn't exist for some reason
    if "fig" not in st.session_state or st.session_state.fig is None:
        st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
        update_start = time.time()
        st.session_state.fig, update_details = update_figure(st.session_state.fig, st.session_state.pauli_strings)
        update_time = time.time() - update_start
        if 'timing_metrics' in st.session_state:
            st.session_state.timing_metrics['plot_update'].append(update_time)
            st.session_state.timing_metrics['plot_update_detail'].append(update_details)
    
    # Measure rendering time
    render_start = time.time()
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"final_plot_{current_hash[:8]}",
        theme=None  # Disable theming for better performance
    )
    render_time = time.time() - render_start
    if 'timing_metrics' in st.session_state:
        st.session_state.timing_metrics['rendering'].append(render_time)
    
    # Display final timing stats if they exist
    if 'timing_metrics' in st.session_state:
        display_final_timing_stats(metrics_placeholder)

def handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash):
    """Handle the initial load of the application."""
    initial_pauli = vector_to_pauli_string(initial_state)
    
    # Initialize timing metrics
    if 'timing_metrics' not in st.session_state:
        st.session_state.timing_metrics = {
            'calculation': [],
            'plot_update': [],
            'rendering': [],
            'plot_update_detail': []
        }
    
    # Create the figure once
    fig_start = time.time()
    st.session_state.fig = make_empty_figure(n, T_steps)
    fig_time = time.time() - fig_start
    
    # Measure update time
    update_start = time.time()
    st.session_state.fig, update_details = update_figure(st.session_state.fig, [initial_pauli])
    update_time = time.time() - update_start
    st.session_state.timing_metrics['plot_update'].append(update_time)
    st.session_state.timing_metrics['plot_update_detail'].append(update_details)
    
    # Measure rendering time
    render_start = time.time()
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"initial_load_{current_hash[:8]}",
        theme=None  # Disable theming for better performance
    )
    render_time = time.time() - render_start
    st.session_state.timing_metrics['rendering'].append(render_time)
    
    # Display initial timing metrics
    metrics_placeholder = st.empty()
    total_time = fig_time + update_time + render_time
    with metrics_placeholder.container():
        cols = st.columns(4)
        cols[0].metric("Figure creation", f"{fig_time*1000:.1f} ms", f"{(fig_time/total_time)*100:.1f}%")
        cols[1].metric("Plot update time", f"{update_time*1000:.1f} ms", f"{(update_time/total_time)*100:.1f}%")
        cols[2].metric("Rendering time", f"{render_time*1000:.1f} ms", f"{(render_time/total_time)*100:.1f}%")
        cols[3].metric("Total time", f"{total_time*1000:.1f} ms", None)
    
    # Display detailed plot update metrics
    detail_metrics = st.empty()
    display_plot_update_details(detail_metrics, update_details)
    
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
    
    # Optimize Streamlit performance
    st.cache_data.clear()  # Clear on each run to reduce memory usage
    
    # Setup UI elements
    n, T_steps, local_rule, initial_state, plot_placeholder, status_placeholder = setup_ui_elements()
    
    # Build the global operator (use cached version for better performance)
    global_operator = build_cached_global_operator(n, local_rule)
    
    # Calculate parameters hash
    current_hash = get_params_hash(n, T_steps, local_rule, initial_state)
    
    # Handle parameter changes
    handle_parameter_changes(n, T_steps, local_rule, initial_state, current_hash)
    
    # App execution flow - use with blocks for better memory management
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
