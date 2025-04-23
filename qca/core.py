import numpy as np

def mod2_matmul(A, v):
    """Perform matrix multiplication over F2 (mod 2 arithmetic)."""
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
    
    Returns a (2*n)x(2*n) numpy array representing the global operator T.
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
    
    v should be a 1D numpy array of length 2*n.
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
    """
    mapping = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    n = len(pauli_str)
    state = np.zeros(2*n, dtype=int)
    for i, ch in enumerate(pauli_str):
        if ch not in mapping:
            raise ValueError("Invalid character in Pauli string. Must be one of I, X, Z, Y.")
        state[2*i], state[2*i+1] = mapping[ch]
    return state
