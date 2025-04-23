import streamlit as st
import numpy as np
from simulation.core import (
    vector_to_pauli_string,
    calculate_step
)
from visualization import make_empty_figure, update_figure, generate_hires_plot
from config import BATCH_SIZE, EXPORT_RESOLUTIONS

def setup_main_view():
    """
    Set up the main view UI components.
    
    Returns:
    --------
    tuple
        A tuple containing placeholders for plot and status updates.
    """
    # Set up the main content area
    st.markdown("## 📊 Simulation Results")
    st.markdown("This plot shows the evolution of Pauli operators on each qubit over time.")
    
    # Create placeholders for the plot and status updates
    plot_placeholder = st.empty()
    status_placeholder = st.empty()
    
    return plot_placeholder, status_placeholder

def run_simulation(n, plot_placeholder, status_placeholder, current_hash):
    """
    Run the progressive simulation.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    plot_placeholder : streamlit.empty
        Placeholder for the plot.
    status_placeholder : streamlit.empty
        Placeholder for status updates.
    current_hash : str
        Hash of current parameters to use as part of keys.
    """
    if st.session_state.current_step < st.session_state.target_steps:
        # Create the figure once on first batch
        if st.session_state.fig is None:
            st.session_state.fig = make_empty_figure(n, st.session_state.target_steps)
        
        # Show a progress bar
        progress_bar = st.progress(0)
        
        # Run all steps without updating the display
        for step in range(st.session_state.current_step, st.session_state.target_steps):
            next_state, next_pauli = calculate_step(st.session_state.states[-1])
            st.session_state.states.append(next_state.copy())
            st.session_state.pauli_strings.append(next_pauli)
            st.session_state.current_step += 1
            
            # Update progress bar
            progress = st.session_state.current_step / st.session_state.target_steps
            progress_bar.progress(progress)
        
        # Update the figure once at the end
        st.session_state.fig = update_figure(st.session_state.fig, st.session_state.pauli_strings)
        plot_placeholder.plotly_chart(
            st.session_state.fig,
            use_container_width=False,
            config=getattr(st.session_state.fig, '_config', None),
            key=f"final_run_{current_hash[:8]}",
            theme=None
        )
        
        # Always set these flags at the end of simulation
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True
        
        # Clear the status placeholder and progress bar
        status_placeholder.empty()
        progress_bar.empty()
        
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
    """
    Display the final simulation results.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    plot_placeholder : streamlit.empty
        Placeholder for the plot.
    current_hash : str
        Hash of current parameters to use as part of keys.
    """
    from visualization import generate_hires_plot
    
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
    # Using EXPORT_RESOLUTIONS from config
    
    # Export button in the first column
    if col1.button("📥 Export as Wallpaper", use_container_width=True, key="export_button"):
        selected_resolution = col2.selectbox(
            "",
            options=list(EXPORT_RESOLUTIONS.keys()),
            index=0,
            key="resolution_selector",
            label_visibility="collapsed"
        )
        
        # Extract width and height from the selected resolution
        width, height = EXPORT_RESOLUTIONS[selected_resolution]
        
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
            options=list(EXPORT_RESOLUTIONS.keys()),
            index=0,
            key="resolution_selector",
            label_visibility="collapsed"
        )

def handle_initial_load(n, T_steps, initial_state, global_operator, plot_placeholder, current_hash):
    """
    Handle the initial load of the application.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    T_steps : int
        Number of simulation steps.
    initial_state : numpy.ndarray
        Initial state vector.
    global_operator : numpy.ndarray
        Global operator for the simulation.
    plot_placeholder : streamlit.empty
        Placeholder for the plot.
    current_hash : str
        Hash of current parameters to use as part of keys.
    """
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