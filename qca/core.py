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
        # Determine indices for left neighbor, cell itself, and right neighbor (with periodic BC)
        left_idx   = (i - 1) % n
        center_idx = i
        right_idx  = (i + 1) % n
        
        T[2*i:2*i+2, 2*left_idx:2*left_idx+2]   = A_left
        T[2*i:2*i+2, 2*center_idx:2*center_idx+2] = A_center
        T[2*i:2*i+2, 2*right_idx:2*right_idx+2]   = A_right
    return T % 2

def vector_to_pauli_string(v):
    """
    Convert a state vector over F2 (first half: x-part, second half: z-part)
    into a string of Pauli operators.
    
    Mapping:
      (0,0) -> I, (1,0) -> X, (0,1) -> Z, (1,1) -> Y.
      
    v should be a 1D numpy array of length 2*n.
    """
    v = v % 2
    n = len(v) // 2
    pauli_str = ""
    for i in range(n):
        x = v[i]
        z = v[n + i]
        if x == 0 and z == 0:
            pauli_str += "I"
        elif x == 1 and z == 0:
            pauli_str += "X"
        elif x == 0 and z == 1:
            pauli_str += "Z"
        elif x == 1 and z == 1:
            pauli_str += "Y"
    return pauli_str

def simulate_QCA(n, T_steps, initial_state, global_operator):
    """
    Simulate the 1D QCA for T_steps time steps.
    
    - n: number of cells.
    - initial_state: a numpy array of length 2*n (over F2).
    - global_operator: a (2*n)x(2*n) numpy array representing the update rule.
    
    Returns a tuple:
      (states, pauli_strings)
    where states is a list of state vectors and pauli_strings is the corresponding
    list of Pauli string representations.
    """
    state = initial_state.copy() % 2
    states = [state.copy()]
    pauli_strings = [vector_to_pauli_string(state)]
    for _ in range(T_steps):
        state = mod2_matmul(global_operator, state) % 2
        states.append(state.copy())
        pauli_strings.append(vector_to_pauli_string(state))
    return states, pauli_strings
