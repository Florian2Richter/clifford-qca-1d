import streamlit as st
import numpy as np
import hashlib

def mod2_matmul(A, v):
    """
    Perform matrix multiplication over F2 (mod 2 arithmetic).
    
    Parameters:
    -----------
    A : numpy.ndarray
        Matrix to multiply.
    v : numpy.ndarray
        Vector to multiply by.
        
    Returns:
    --------
    numpy.ndarray
        Result of A @ v (mod 2).
    """
    return (A.dot(v)) % 2

def build_global_operator(n, local_rule):
    """
    Build the global update operator T for a 1D cellular automaton with n cells.
    Each cell state is in F2^2, and the local rule is a 2x6 matrix over F2.
    
    The local rule is split into three 2x2 blocks:
      local_rule = [A_left | A_center | A_right]
    which determine the contribution of the left neighbor, the cell itself,
    and the right neighbor.
    
    Periodic boundary conditions are assumed.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    local_rule : numpy.ndarray
        2x6 matrix over F2 representing the local rule.
        
    Returns:
    --------
    numpy.ndarray
        A (2*n)x(2*n) numpy array representing the global operator T.
    """
    n = int(n)
    local_rule = np.array(local_rule) % 2
    if local_rule.shape != (2, 6):
        raise ValueError("local_rule must be a 2x6 matrix over F2")
    
    # Split the local rule into three 2x2 blocks.
    A_left   = local_rule[:, :2]
    A_center = local_rule[:, 2:4]
    A_right  = local_rule[:, 4:]
    
    # Initialize the global operator T.
    T = np.zeros((2*n, 2*n), dtype=int)
    for i in range(n):
        left_idx   = (i - 1) % n
        center_idx = i
        right_idx  = (i + 1) % n
        
        T[2*i:2*i+2, 2*left_idx:2*left_idx+2]   = A_left
        T[2*i:2*i+2, 2*center_idx:2*center_idx+2] = A_center
        T[2*i:2*i+2, 2*right_idx:2*right_idx+2]   = A_right
    return T % 2

def vector_to_pauli_string(v):
    """
    Convert a state vector (with each cell stored as (x, z)) into a Pauli string.
    
    Mapping:
      (0,0) -> I, (1,0) -> X, (0,1) -> Z, (1,1) -> Y.
    
    Parameters:
    -----------
    v : numpy.ndarray
        1D numpy array of length 2*n representing the state vector.
        
    Returns:
    --------
    str
        Pauli string representation of the state vector.
    """
    v = v % 2
    n = len(v) // 2
    pauli_str = ""
    for i in range(n):
        x = v[2*i]
        z = v[2*i+1]
        if x == 0 and z == 0:
            pauli_str += "I"
        elif x == 1 and z == 0:
            pauli_str += "X"
        elif x == 0 and z == 1:
            pauli_str += "Z"
        elif x == 1 and z == 1:
            pauli_str += "Y"
    return pauli_str

def pauli_string_to_state(pauli_str):
    """
    Convert a Pauli string (of length n, consisting of I, X, Z, Y)
    into a state vector of length 2*n over F2.
    
    Mapping:
      I -> (0,0), X -> (1,0), Z -> (0,1), Y -> (1,1).
      
    Parameters:
    -----------
    pauli_str : str
        String of Pauli operators (I, X, Z, Y).
        
    Returns:
    --------
    numpy.ndarray
        State vector representation of the Pauli string.
    """
    mapping = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    n = len(pauli_str)
    state = np.zeros(2*n, dtype=int)
    for i, ch in enumerate(pauli_str):
        if ch not in mapping:
            raise ValueError("Invalid character in Pauli string. Must be one of I, X, Z, Y.")
        state[2*i], state[2*i+1] = mapping[ch]
    return state

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
    """
    Create a hash of all parameters to detect changes.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    T_steps : int
        Number of time steps to simulate.
    local_rule : numpy.ndarray
        Local rule matrix.
    initial_state : numpy.ndarray
        Initial state vector.
        
    Returns:
    --------
    str
        MD5 hash of the parameters.
    """
    hash_str = f"{n}_{T_steps}_{local_rule.tobytes().hex()}_{initial_state.tobytes().hex()}"
    return hashlib.md5(hash_str.encode()).hexdigest()

def handle_parameter_changes(n, T_steps, local_rule, initial_state, current_hash):
    """
    Handle changes in simulation parameters.
    
    Parameters:
    -----------
    n : int
        Number of cells in the QCA.
    T_steps : int
        Number of time steps to simulate.
    local_rule : numpy.ndarray
        Local rule matrix.
    initial_state : numpy.ndarray
        Initial state vector.
    current_hash : str
        Hash of the current parameters.
    """
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