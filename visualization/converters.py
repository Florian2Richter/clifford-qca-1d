"""
Data conversion utilities for visualization.
"""

import numpy as np
from config import PAULI_MAPPING

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