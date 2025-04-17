import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import streamlit.components.v1 as components
import streamlit as st

def pauli_to_numeric(pauli_str):
    """
    Convert a Pauli string (e.g., "IXZY...") into an array of numeric codes:
      I -> 0, X -> 1, Z -> 2, Y -> 3.
    Returns a numpy array of shape (len(pauli_str),).
    """
    mapping = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    numeric = [mapping.get(ch, 0) for ch in pauli_str]
    return np.array(numeric)

def plot_spacetime_plotly(pauli_strings, total_time_steps=None):
    """
    Plot the spacetime diagram using Plotly for interactivity.
    
    Parameters:
    -----------
    pauli_strings : list of strings
        List of Pauli strings representing the state at each calculated time step.
    total_time_steps : int, optional
        Total number of time steps to show in the plot. If None, uses len(pauli_strings).
        This is used for progressive visualization to keep axes consistent.
    """
    current_time_steps = len(pauli_strings)
    if current_time_steps == 0:
        return go.Figure()
    
    cell_count = len(pauli_strings[0])
    
    if total_time_steps is None or total_time_steps < current_time_steps:
        total_time_steps = current_time_steps
    
    data = np.zeros((total_time_steps, cell_count), dtype=int)
    for t, s in enumerate(pauli_strings):
        data[t] = pauli_to_numeric(s)
    
    colorscale = [
        [0.0, 'white'],
        [0.25, 'white'],
        [0.25, '#008080'],
        [0.5, '#008080'],
        [0.5, '#FF7F50'],
        [0.75, '#FF7F50'],
        [0.75, '#4A4A4A'],
        [1.0, '#4A4A4A']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=list(range(cell_count)),
        y=list(range(total_time_steps)),
        colorscale=colorscale,
        zmin=0,
        zmax=3,
        showscale=True,
        colorbar=dict(
            title='Pauli Operator',
            tickvals=[0.5, 1.5, 2.5, 3.5],
            ticktext=['I', 'X', 'Z', 'Y'],
            lenmode='pixels',
            len=200,
            yanchor='top',
            y=1
        ),
        hovertemplate="Time: %{y}<br>Cell: %{x}<br>Operator: %{customdata}<extra></extra>",
        customdata=[[pauli_strings[y][x] if y < current_time_steps else 'I' 
                    for x in range(cell_count)] 
                    for y in range(total_time_steps)]
    ))
    
    fig.update_layout(
        title='1D Clifford QCA Spacetime Diagram',
        xaxis_title='Cell Position',
        yaxis_title='Time Step',
        yaxis_autorange='reversed',
        xaxis=dict(tickmode='linear', dtick=max(1, cell_count // 15)),
        yaxis=dict(tickmode='linear', dtick=max(1, total_time_steps // 15)),
        width=800,
        height=500,
        autosize=False
    )
    
    return fig