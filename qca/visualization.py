import numpy as np
import plotly.graph_objects as go

# Global constants for better performance
PAULI_MAPPING = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    numeric = [PAULI_MAPPING.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

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
    
    # Create the heatmap with optimized performance settings
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
    
    # Set layout with performance optimizations
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
        hovermode='closest',    # Simplify hover behavior
        hoverdistance=10,       # Limit hover distance detection
        dragmode='zoom',        # Enable zoom selection
        modebar=dict(
            orientation='v',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
    )
    
    # Configure interactivity
    config = {
        'displayModeBar': True,
        'scrollZoom': True,     # Enable scroll wheel zooming
        'displaylogo': False,
        'responsive': True,
        'staticPlot': False,    # Allow interactivity
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'qca_simulation',
            'height': 500,
            'width': 800,
            'scale': 1
        },
        'modeBarButtonsToRemove': [
            'select2d', 'lasso2d',  # Keep zoom buttons
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines', 'toggleHover', 'resetViewMapbox'
        ]
    }
    
    # Simplify axes
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
        # Convert strings to numeric data using our mapping (I->0, X->1, Z->2, Y->3)
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
    
    # Update the figure with the integer values and customdata
    fig.plotly_restyle(
        {'z': [z_data], 'customdata': [customdata]},
        trace_indexes=[0]
    )
    
    return fig