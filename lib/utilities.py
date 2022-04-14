import numpy as np
import copy
import warnings
from qiskit import (
    Aer,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    transpile
)
from qiskit.utils.mitigation.fitters import CompleteMeasFitter
from qiskit.ignis.verification.tomography import StateTomographyFitter, state_tomography_circuits
from qiskit.quantum_info import state_fidelity
from scipy.linalg import expm
import os


# Constant objects used in the calculations
X = np.array([[0,1],[1,0]]) 
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
Id = np.eye(2)

# Defining the hamiltonian divided in: 
#       - H1: first two qubits interactions.
#       - H2: second two qubits interactions.
H1 = np.kron(X, np.kron(X,Id)) + np.kron(Y, np.kron(Y,Id)) + np.kron(Z, np.kron(Z,Id)) 
H2 = np.kron(Id, np.kron(X,X)) + np.kron(Id, np.kron(Y,Y)) + np.kron(Id, np.kron(Z,Z)) 

# Loads the Jakarta-adapted Toffoli gate
dirname = os.path.dirname(__file__)
Toffoli_gate = QuantumCircuit.from_qasm_file(os.path.join(dirname, "Toffoli.qasm"))
Toffoli_gate.name = "optimized\ntoffoli"
# B is the permutation matrix used in the HSD 
B = np.array(
   [[0,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1]]
)
# look up table of B
state_permutations_B = {'110':'110', '111':'111', '101': '101', '000':'011', '011':'100', '100':'000', '001':'010', '010':'001'}

# B quantum circuit: this code is reported from the final commit.
# some functions need it.
_B_qr=QuantumRegister(3, name="q-")
_B_qc=QuantumCircuit(_B_qr, name="B")
_B_qc.x(_B_qr[2])
_B_qc.cx(_B_qr[1],_B_qr[0])
_B_qc.cx(_B_qr[2],_B_qr[1])
_B_qc.cx(_B_qr[1],_B_qr[0])
_B_qc.x([_B_qr[0],_B_qr[1],_B_qr[2]])
_B_qc.append(Toffoli_gate,[_B_qr[0],_B_qr[1],_B_qr[2]])
_B_qc.x([_B_qr[0],_B_qr[1]])

def trotter_step_matrix(time, n_steps):
    """Computes numerically the trotter step"""
    return expm(-time/n_steps*H1*1j).dot(expm(-time/n_steps*H2*1j))

def trotterized_matrix(time, n_steps):
    """Computes the trotter_step**n_steps"""
    return np.linalg.matrix_power(trotter_step_matrix(time, n_steps), n_steps)

def get_gates_parameters(U, initial_state={"110": 1.0}):
    """Finds the parameters of the gates based on the system of equations
    defined by the numerical evolution matrix, for the SSD.

    Since the evolution is trivial in the magnetization 0 and 3 subspaces
    the procedure is done only for mag==1 and mag==2

    Note: 
        we suggest the implementation for a generic state 
        but the only one tested is the inital_state of the challenge '110'.

    Args
    ----
        U : np.ndarray
            the trotterized evolution matrix
        initial_state : dict
            the initial state to be evolved in format {"state": amplitude}
    Returns
    ----
        theta_1, theta_2, phi_1, phi_2, omega_1, omega_2 (tuple(float)): 
    
    """
    if initial_state != {"110": 1.0}:
        warnings.warn("Any initial_state different from '110' has not been sufficiently tested.")
    # Builds up the array associated to the initial state
    state = np.zeros(8, dtype=np.complex)

    # Creates the 8-dimensional vector associated to the state
    # checking that it is in a magnetization eigenspace
    magnetization = sum([int(_) for _ in list(initial_state.keys())[0]])
    for base_vector in initial_state:
        if sum([int(_) for _ in base_vector]) != magnetization:
            raise ValueError("States must have the same magnetization!")
        state[int(base_vector, 2)] = initial_state[base_vector]

    # Sends an (a, b, c) state of fixed magnetization
    # into a (alpha, beta, gamma) state of the same mag
    state = U.dot(state)
    # print(f"get_gates_parameters() - U*initial_state is \n{state}")

    # Now takes the relevant components and solves the associated system of eqs.
    if magnetization == 2:

        subspace_coords = np.array([3, 5, 6])
        alpha, beta, gamma = state[subspace_coords]

        theta_1 = 0.5 * (np.angle(alpha) + np.angle(gamma))
        theta_2 = 0

        phi_1 = 0.5 * (np.angle(gamma) - np.angle(beta) - np.pi)
        phi_2 = 0.5 * (np.angle(beta) - np.angle(alpha) + np.pi)

        omega_1 = np.arccos(np.abs(gamma))
        omega_2 = np.arccos(np.abs(beta) / np.sin(omega_1))

    elif magnetization == 1:
        raise NotImplementedError("The magnetization==1 case has not been sufficiently tested")
        subspace_coords = np.array([3, 5, 6])
        alpha, beta, gamma = state[subspace_coords]

        theta_1 = 0.5 * (-np.angle(alpha) - np.angle(gamma))
        theta_2 = 0

        phi_1 = 0.5 * (np.angle(beta) - np.angle(gamma))
        phi_2 = 0.5 * (np.angle(alpha) - np.angle(beta)) - phi_1

        omega_1 = np.arcos(np.abs(gamma))
        omega_2 = np.arccos(np.abs(beta) / np.sin(omega_1))

    return theta_1, theta_2, phi_1, phi_2, omega_1, omega_2

def get_calibration_circuits(qc, method="NIC", eigenvector=None):
    '''
    Returns a list of calibration circuits for all the methods: CIC, NIC and qiskit calibration matrix.

    Args
    ----
        qc (QuantumCircuit): the quantum circuit you wont to calibrate.
        method (string): the method of calibration. Can be CIC, NIC or qiskit.
        eigenvector (string): is a string of binary, example "111". Is the prepared state in the case
                              NIC mitigation tecnique. For CIC and qiskit calibraitions is useless.

    Return
    ----
        calib_circuits (list of QuantumCircuit): list of calibration circuits.
    '''

    calib_circuits = []
    state_labels = ['000', '001', '010', '011', '100', '101', '110', '111']  

    for state in state_labels:
        cr_cal = ClassicalRegister(3, name = "c")
        qr_cal = QuantumRegister(3, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        if method == "NIC": 
            # first we prepare the eigenstate (if method == "NIC").
            for qubit in range(3):
                if eigenvector[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
            # then we append the circuit
            qc_cal.append(qc, qr_cal)
            # than we append the gate that bring the eigenstate to the computational basis.
            for qubit in range(3):
                if eigenvector[::-1][qubit] == "1" and state[::-1][qubit] == "0":
                    qc_cal.x(qr_cal[qubit])
                elif eigenvector[::-1][qubit] == "0" and state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
        # CIC case: first we prepare the initial state than we append the evolution.
        if method == "CIC": 
            # first we prepare the state.
            for qubit in range(3):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
            # than we append the circuit
            qc_cal.append(qc, qr_cal)
        # measure all
        qc_cal.measure(qr_cal, cr_cal)
        calib_circuits.append(qc_cal)

    return calib_circuits

def matrix_from_circuit(qc, simulator = "unitary_simulator", phase=0):

    """
    Return the matrix of the circuit:

    Args 
    ----

        - qc (QuantumCircuit): the quantum circuit, without final measuraments, which you wont know the matrix.
        - simulator (string): the simulator used for the aim:
                                - "unitary_simulator": you will get the exact unitary matrix of the circuit.
                                - "aer_simulator": you will get the probability matrices of the circuit in the computational base.
        - phase (float): global phase

    Return
    -----

        - the matrix of the circuit
        
    """

    backend = Aer.get_backend(simulator)
    job = execute(qc, backend, shots=32000)
    result = job.result()
    circuit_matrix = result.get_unitary(qc, decimals=40) * np.exp(1j * phase)
    return circuit_matrix

def fidelity_count(result, qcs, target_state):
    '''
    given job result, tomography circuits and targhet state it returns the fidelity score.
    '''
    tomo_ising = StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method="lstsq")
    fid=state_fidelity(rho_fit_ising, target_state)
    return fid



def get_evolution_circuit(time, n_steps, method="HSD", initial_state={"110": 1}):
    '''
    Returns the evolution circuit with the associated QuantumRegister.

    Args:
    ----

        time (float): the evolution time.
        n_steps (int): number of trotter steps.
        method (string): decomposition method, can be "HSD" or "SSD".
        initial_state (dict): {"state": amplitude, ...}. Is the initiat state, used only for the SSD
                              to compute the circuit parameters.
                              Warning: functions are not optimized for any initial state,
                              everythink works with "110", can be problems with other initial states.

    Returns
    ----

        QuantumCircuit, QuantumRegister : circuit and quantum register of the evolution circuit.
        
    '''
    if method == "HSD":
        return get_HSD_circuit(time, n_steps)
    elif method == "SSD":
        return get_SSD_circuit(time, n_steps, initial_state=initial_state)
    raise ValueError("The decomposition method 'method' must be chosen between 'HSD' or 'SSD'.")

def get_HSD_circuit(time, n_steps):
    '''
    returns the evolution circuit with the Hilbert Space Decomposition, prepared in the state |000>
    '''
    T = trotterized_matrix(time, n_steps)
    T_b = np.linalg.multi_dot([B, T, B.transpose() ])

    D = T_b[0:4, 0:4]
    # Transpile the D operator and build the evolution circuit
    D_qc = QuantumCircuit(2, name="D")
    D_qc.unitary(D,[0,1])    
    D_qc = transpile(D_qc, basis_gates=['cx','x','sx','rz']) # Jackarta basis gates

    qr_HSD = QuantumRegister(3, name="q_")
    qc_HSD = QuantumCircuit(qr_HSD, name="D")
    # here the state isn't preparated. 
    qc_HSD.append(D_qc, [qr_HSD[0], qr_HSD[1]])
    qc_HSD.append(_B_qc.inverse(), qr_HSD)

    return qc_HSD, qr_HSD

def get_SSD_circuit(time, n_steps, initial_state={"110": 1}):
    '''
    returns the evolution circuit obtained with the Single State Decomposition.
    '''
    # getting the parameters for the gates M1 and M2, solving the equations described in 1.1).
    theta_1, theta_2, phi_1, phi_2, omega_1, omega_2 = get_gates_parameters(trotterized_matrix(time, n_steps), initial_state=initial_state)
    # build M1 and M2
    M1_qc = _get_M(theta_1, phi_1, omega_1)
    M2_qc = _get_M(theta_2, phi_2, omega_2)
    # define the circuit of U
    qr_U = QuantumRegister(3 ,name="q_")
    qc_U = QuantumCircuit(qr_U, name="U")
    # append the gates
    qc_U.append(M1_qc, [qr_U[0], qr_U[1]])
    qc_U.append(M2_qc, [qr_U[1], qr_U[2]])
    # transpile and draw the circuit
    qc_U=transpile(qc_U, basis_gates=["cx","rz","x","sx"])
    return qc_U, qr_U

def fast_tomography_calibration_MeasFitters(calibration_results, method="NIC", U_ideal=None):
    '''
    builds a list of CompleteMeasFitter objects, for each tomography basis.
    
    Args:
    ----

        calibration_results (job.result()): the results of the calibration NIC or CIC.
        method (string): "NIC" or "CIC", chose the calibration technique.
        U_ideal (np.array): unitary matrix of the circuit. Used only for CIC.

    Returns:
    ----

        meas_fitters (list of CompleteMeasCal objects): one for each tomography basis.
        
    '''
    state_labels = ['000', '001', '010', '011', '100', '101', '110', '111']
    meas_fitter = CompleteMeasFitter(calibration_results, state_labels=state_labels)
    # copy the measured probability matrix by the calibration circuits.
    U_tilde = meas_fitter.cal_matrix
    #defining the tomography basis circuits.
    qr_basi = QuantumRegister(3)
    qc_basi = QuantumCircuit(qr_basi)
    tomography_basis = state_tomography_circuits(qc_basi, qr_basi)
    # computing the calibration matrix in the computational basis.
    if method == "NIC":
        C = U_tilde
    elif method == "CIC":
        # computing the ideal probability matrix of the circuit.
        if U_ideal is None: print("expected the U_ideal unitary matrix of the circuit.")
        U_ideal_abs = np.abs(U_ideal)**2
        U_ideal_abs_inv = np.linalg.inv(U_ideal_abs)
        C = np.dot(U_tilde, U_ideal_abs_inv)
    # building the fast tomography circuits calibration: a list of 27 CompleteMeasFitter objects, one for each basis.
    meas_fitters = []
    for basis in tomography_basis:
        # compute the tomography unitary basis matrix and the inverse.
        basis.remove_final_measurements()
        base_matrix_amplitudes = matrix_from_circuit(basis)
        base_matrix_amplitudes_inverse = np.linalg.inv(base_matrix_amplitudes)
        # compute the probability matrix of the base changing.
        base_matrix = np.abs(base_matrix_amplitudes)**2
        base_matrix_inverse = np.abs(base_matrix_amplitudes_inverse)**2
        # compute the calibration matrix in the new basis.
        C_base = np.linalg.multi_dot([base_matrix, C,  base_matrix_inverse])
        # define a new object CompleteMeasFitter.
        meas_fitter_aus = copy.deepcopy(meas_fitter)
        meas_fitter_aus._tens_fitt.cal_matrices[0]=C_base
        meas_fitters.append(meas_fitter_aus)
    return meas_fitters

def _get_M(theta, phi, omega, name="M"): # defining the M matrix
    '''
    returns the fixed 2-qubits magnetization gate. 
    This is a copy of the function in the notebook, 
    some functions need it here.
    '''
    qr=QuantumRegister(2, name="q_")
    M_qc=QuantumCircuit(qr, name=name)

    M_qc.rz(2*theta,qr[1])
    M_qc.h(qr[0])
    M_qc.cx(qr[0],qr[1])
    M_qc.ry(omega,qr)
    M_qc.cx(qr[0],qr[1])
    M_qc.h(qr[0])
    M_qc.rz(2*phi,qr[1])

    return M_qc

def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)

def occurrences_to_vector(occurrences_dict):
    """Converts the occurrences dict to vector.

    Args:
    ----

        occurrences_dict (dict) : dict returned by BaseJob.results.get_counts() 
    
    Returns:
    ----

        counts_vector (np.array): the vector of counts.

    """
    counts_vector = np.zeros(8)
    for i, state in enumerate(occurrences_dict):
        counts_vector[i] = occurrences_dict[state]
    return counts_vector