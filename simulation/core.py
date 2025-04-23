import streamlit as st
import numpy as np
from qca.core import build_global_operator, vector_to_pauli_string, mod2_matmul

def calculate_step(current_state):
    """
    Calculate the next state for the QCA simulation.
    
    Parameters:
    -----------
    current_state : numpy.ndarray
        The current state of the system.
        
    Returns:
    --------
    tuple
        (next_state, next_pauli) - The next state and its Pauli string representation.
    """
    next_state = mod2_matmul(st.session_state.global_operator, current_state) % 2
    next_pauli = vector_to_pauli_string(next_state)
    return next_state, next_pauli

@st.cache_data(ttl=900, show_spinner=False)
def build_cached_global_operator(n, local_rule):
    """
    Cached version of build_global_operator for better performance.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    local_rule : numpy.ndarray
        The local rule matrix.
        
    Returns:
    --------
    numpy.ndarray
        The global operator matrix.
    """
    return build_global_operator(n, local_rule) 