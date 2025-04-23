import numpy as np
import plotly.graph_objects as go

# Global constants
PAULI_MAPPING = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}

def pauli_strings_to_numeric(pauli_strings):
    """
    Vectorized conversion of multiple Pauli strings to numeric arrays.
    
    Parameters:
    -----------
    pauli_strings : list of strings
        List of Pauli strings to convert.
        
    Returns:
    --------
    np.ndarray of shape (len(pauli_strings), len(pauli_strings[0]))
    """
    if not pauli_strings:
        return np.array([])
    
    n_strings = len(pauli_strings)
    string_len = len(pauli_strings[0])
    
    # Pre-allocate the result array
    result = np.zeros((n_strings, string_len), dtype=np.int8)
    
    # Create a character array from the strings
    char_array = np.array([list(s) for s in pauli_strings])
    
    # Vectorized mapping using NumPy
    for char, value in PAULI_MAPPING.items():
        result[char_array == char] = value
    
    return result

def make_empty_figure(cell_count, total_time_steps):
    """
    Create an empty plotly figure with the heatmap structure but initialized with all 'I' operators.
    This creates the figure only once, which can then be updated efficiently.
    
    Parameters:
    -----------
    cell_count : int
        Number of cells in the QCA.
    total_time_steps : int
        Total number of time steps to show in the plot.
    """
    # Initialize empty data arrays
    data = np.zeros((total_time_steps, cell_count), dtype=np.int8)  # All 'I' operators
    customdata = [['I'] * cell_count for _ in range(total_time_steps)]
    
    # Define colors for each Pauli operator with normalized values (0 to 1)
    color_mapping = [
        [0.0, 'white'],      # I (value 0)
        [0.33, '#008080'],   # X (value 1)
        [0.67, '#FF7F50'],   # Z (value 2)
        [1.0, '#4A4A4A']     # Y (value 3)
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=list(range(cell_count)),
        y=list(range(total_time_steps)),
        colorscale=color_mapping,
        zmin=0,
        zmax=3,
        showscale=True,
        colorbar=dict(
            title='Pauli Operator',
            tickvals=[0, 1, 2, 3],          # Use actual values
            ticktext=['I', 'X', 'Z', 'Y'],  # Label order matches values
            lenmode='pixels',
            len=200,
            yanchor='top',
            y=1,
            thickness=25,
            outlinewidth=1,
            outlinecolor='black',
            ticks='outside',
            ticklen=8, 
            tickwidth=2
        ),
        hovertemplate="Time: %{y}<br>Cell: %{x}<br>Operator: %{customdata}<extra></extra>",
        customdata=customdata
    ))
    
    # Set layout
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        yaxis_autorange='reversed',
        xaxis=dict(
            tickmode='linear', 
            dtick=max(1, cell_count // 10),  # Fewer ticks
            constrain='domain'
        ),
        yaxis=dict(
            tickmode='linear', 
            dtick=max(1, total_time_steps // 10),  # Fewer ticks
            constrain='domain'
        ),
        width=800,
        height=500,
        autosize=False,
        # Enable interactive features
        uirevision=True,  # Maintain UI state during updates
        hovermode='closest',
        hoverdistance=10,
        dragmode='zoom',
        modebar=dict(
            orientation='v',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Configure interactivity
    config = {
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
        'responsive': True,
        'staticPlot': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'qca_simulation',
            'height': 500,
            'width': 800,
            'scale': 1
        },
        'modeBarButtonsToRemove': [
            'select2d', 'lasso2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines', 'toggleHover', 'resetViewMapbox'
        ]
    }
    
    # Configure axes
    fig.update_xaxes(showgrid=False, zeroline=False, showspikes=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showspikes=False)
    
    # Attach the config to the figure for use in Streamlit
    fig._config = config
    
    return fig

def update_figure(fig, pauli_strings):
    """
    Update an existing plotly figure with new data.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The existing figure to update.
    pauli_strings : list of strings
        List of Pauli strings representing the state at each calculated time step.
    
    Returns:
    --------
    The updated figure.
    """
    # Get dimensions
    current_time_steps = len(pauli_strings)
    if current_time_steps == 0:
        return fig
    
    cell_count = len(pauli_strings[0])
    total_time_steps = len(fig.data[0].z)
    
    # Convert strings to numeric data
    if current_time_steps > 0:
        # Convert strings to numeric data
        numeric_data = pauli_strings_to_numeric(pauli_strings)
        
        # Prepare the full data array
        max_steps = min(current_time_steps, total_time_steps)
        z_data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
        z_data[:max_steps] = numeric_data[:max_steps]
    else:
        z_data = np.zeros((total_time_steps, cell_count), dtype=np.int8)
    
    # Create customdata for hover information
    customdata = [['I'] * cell_count for _ in range(total_time_steps)]
    for t, s in enumerate(pauli_strings):
        if t < total_time_steps:
            customdata[t] = list(s)
    
    # Update the figure
    fig.plotly_restyle(
        {'z': [z_data], 'customdata': [customdata]},
        trace_indexes=[0]
    )
    
    return fig

def generate_hires_plot(pauli_strings, width=1920, height=1080):
    """
    Generate a high-resolution plot of the QCA simulation for wallpaper use.
    
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
    
    # Create the heatmap for wallpaper
    fig = go.Figure(data=go.Heatmap(
        z=numeric_data,
        colorscale=color_mapping,
        zmin=0,
        zmax=3,
        showscale=False,  # No colorbar for clean wallpaper
        hoverinfo='none',  # No hover for static image
    ))
    
    # Set completely clean layout with absolute zero margins
    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        autosize=False,
    )
    
    # Eliminate all axes, ticks, and constraints
    fig.update_xaxes(
        visible=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        constrain="domain",
        range=[0, cell_count],
        fixedrange=True
    )
    
    fig.update_yaxes(
        visible=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor=None,  # Remove scale anchoring
        constrain="domain",
        range=[0, time_steps],
        fixedrange=True
    )
    
    # Convert to PNG bytes with exact dimensions
    img_bytes = fig.to_image(format="png", width=width, height=height, scale=1)
    
    return img_bytes