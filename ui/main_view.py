import streamlit as st
from qca.visualization import make_empty_figure, update_figure, generate_hires_plot
from qca.core import vector_to_pauli_string
from simulation.core import calculate_step

def setup_main_view():
    """
    Set up the main view UI elements.
    
    Returns:
    --------
    tuple
        (plot_placeholder, status_placeholder) - Placeholders for the main content
    """
    # Main title
    st.markdown('<h1 class="main-header">1D Clifford QCA Simulator</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="description">
    This simulator visualizes the evolution of a 1-Dimensional Clifford Quantum Cellular Automaton (QCA). 
    The simulation shows how Pauli operators (I, X, Z, Y) propagate through a 1D lattice over time.
    </div>
    """, unsafe_allow_html=True)
    
    # Create placeholders for plot and status
    plot_placeholder = st.empty()
    status_placeholder = st.empty()
    
    return plot_placeholder, status_placeholder

def run_simulation(n, plot_placeholder, status_placeholder, current_hash):
    """Run the simulation, updating the plot as we go."""
    if not st.session_state.simulation_running:
        return
    
    # If we're done, mark as complete and exit
    if st.session_state.current_step >= st.session_state.target_steps:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        status_placeholder.empty()
        
        # Force a rerun to ensure display_results is shown
        st.rerun()
        return
    
    # Display running status
    status_placeholder.info(f"Running simulation: step {st.session_state.current_step}/{st.session_state.target_steps}")
    
    # Calculate the next step using our global operator
    next_state, next_pauli = calculate_step(
        st.session_state.states[-1]
    )
    
    # Store the results
    st.session_state.states.append(next_state)
    st.session_state.pauli_strings.append(next_pauli)
    
    # Update the plot with all Pauli strings
    st.session_state.fig = update_figure(st.session_state.fig, st.session_state.pauli_strings)
    
    # Display the updated plot
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"plot_{st.session_state.current_step}_{current_hash[:8]}",
        theme=None
    )
    
    # Increment the step counter
    st.session_state.current_step += 1
    
    # If we've reached the target, mark as complete
    if st.session_state.current_step >= st.session_state.target_steps:
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
    
    # Display the final plot
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
    
    # Selection for resolution
    selected_resolution = col2.selectbox(
        "",
        options=list(resolution_options.keys()),
        index=0,
        key="resolution_selector",
        label_visibility="collapsed"
    )
    
    # Export button in the first column
    if col1.button("📥 Export as Wallpaper", use_container_width=True, key="export_button"):
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

def handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash):
    """Handle the initial load of the application."""
    from qca.core import vector_to_pauli_string
    
    # Convert the initial state to Pauli string
    initial_pauli = vector_to_pauli_string(initial_state)
    
    # Create the figure once
    st.session_state.fig = make_empty_figure(n, T_steps)
    st.session_state.fig = update_figure(st.session_state.fig, [initial_pauli])
    
    # Display the initial plot
    plot_placeholder.plotly_chart(
        st.session_state.fig,
        use_container_width=False,
        config=getattr(st.session_state.fig, '_config', None),
        key=f"initial_load_{current_hash[:8]}",
        theme=None
    )
    
    # Initialize session state for the simulation
    st.session_state.pauli_strings = [initial_pauli]
    st.session_state.states = [initial_state]
    st.session_state.global_operator = global_operator
    st.session_state.target_steps = T_steps
    st.session_state.params_hash = current_hash
    st.session_state.simulation_running = True
    st.session_state.initialized = True 