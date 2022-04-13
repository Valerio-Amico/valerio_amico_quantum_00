from logging import warning
import numpy as np
import copy
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


def trotter_step_matrix(time, n_steps):
    """Computes numerically the trotter step"""
    return expm(-time/n_steps*H1*1j).dot(expm(-time/n_steps*H2*1j))

def trotterized_matrix(time, n_steps):
    """Computes the trotter_step**n_steps"""
    return np.linalg.matrix_power(trotter_step_matrix(time, n_steps), n_steps)

def get_gates_parameters(U, initial_state={"110": 1.0}):
    """Finds the parameters of the gates based on the system of equations
    defined by the numerical evolution matrix.

    Since the evolution is trivial in the magnetization 0 and 3 subspaces
    the procedure is done only for mag==1 and mag==2

    Args
    ----
        U : np.ndarray
            the trotterized evolution matrix
        initial_state : dict
            the initial state to be evolved in format {"state": amplitude}
    """

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

        r1 = 0.5 * (np.angle(alpha) + np.angle(gamma))
        r2 = 0

        f1 = 0.5 * (np.angle(gamma) - np.angle(beta) - np.pi)
        f2 = 0.5 * (np.angle(beta) - np.angle(alpha) + np.pi)

        a1 = np.arccos(np.abs(gamma))
        a2 = np.arccos(np.abs(beta) / np.sin(a1))

    elif magnetization == 1:
        raise NotImplementedError("The magnetization==1 case has not been sufficiently tested")
        subspace_coords = np.array([3, 5, 6])
        alpha, beta, gamma = state[subspace_coords]

        r1 = 0.5 * (-np.angle(alpha) - np.angle(gamma))
        r2 = 0

        f1 = 0.5 * (np.angle(beta) - np.angle(gamma))
        f2 = 0.5 * (np.angle(alpha) - np.angle(beta)) - f1

        a1 = np.arcos(np.abs(gamma))
        a2 = np.arccos(np.abs(beta) / np.sin(a1))

    return r1, r2, f1, f2, a1, a2

def get_calibration_circuits(qc, method="NIC"):
    '''
    Returns a list of calibration circuits for all the methods: CIC, NIC and qiskit calibration matrix.

    Args
    ----
        qc (QuantumCircuit): the quantum circuit you wont to calibrate.
        method (string): the method of calibration. Can be CIC, NIC or qiskit.

    Return
    ----
        calib_circuits (list of QuantumCircuit): list of calibration circuits.
    '''
    print("invertire lo stato iniziale se la decomposizione è HSD.")
    calib_circuits = []
    state_labels = ['000', '001', '010', '011', '100', '101', '110', '111']  

    for state in state_labels:
        cr_cal = ClassicalRegister(3, name = "c")
        qr_cal = QuantumRegister(3, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        # first we append the circuit (if method == "NIC").
        if method == "NIC": qc_cal.append(qc, qr_cal)
        # than we prepare the state.
        for qubit in range(3):
            if state[::-1][qubit] == "1":
                qc_cal.x(qr_cal[qubit])
        # then we append the circuit (if method == "CIC").
        if method == "CIC": qc_cal.append(qc, qr_cal)
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
    given the job result and the targhet state it returns the fidelity
    '''
    tomo_ising = StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method="lstsq")
    fid=state_fidelity(rho_fit_ising, target_state)
    return fid





def get_evolution_circuit(time, n_steps, method="HSD"):
    '''
    Returns the evolution circuit with the associated QuantumRegister.

    Returns
    ----

        QuantumCircuit, QuantumRegister
        
    '''
    print("attenzione al caso != HSD o SSD.  -get_evolution_circuit")
    if method == "HSD":
        return get_HSD_circuit(time, n_steps)
    if method == "SSD":
        return get_SSD_circuit(time, n_steps)
    return "specify a decomposition technique (HSD or SSD)"

def get_HSD_circuit(time, n_steps, initial_state={"110": 1}):
    print("attenzione! funziona solo per lo stato 110")
    # Build the permutation operator
    B_qr=QuantumRegister(3, name="q_")
    B_qc=QuantumCircuit(B_qr, name="B")
    B_qc.x(B_qr[2])
    B_qc.cx(B_qr[1],B_qr[0])
    B_qc.cx(B_qr[2],B_qr[1])
    B_qc.cx(B_qr[1],B_qr[0])
    B_qc.x([B_qr[0],B_qr[1],B_qr[2]])
    B_qc.append(Toffoli_gate, [B_qr[0],B_qr[1],B_qr[2]])
    B_qc.x([B_qr[0],B_qr[1]])

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

    T = trotterized_matrix(time, n_steps)
    T_b = np.linalg.multi_dot([ B, T, B.transpose() ])

    D = T_b[0:4, 0:4]

    from qiskit import transpile
    # Transpile the D operator and build the evolution circuit
    D_qc = QuantumCircuit(2, name="D")
    D_qc.unitary(D,[0,1])    
    D_qc = transpile(D_qc, basis_gates=['cx','x','sx','rz']) # Jackarta basis gates

    qr_HSD = QuantumRegister(3, name="q_")
    qc_HSD = QuantumCircuit(qr_HSD, name="D")

    # the permutation of 110 is 110 itself, so we prepare it.
    initial_state = "110"

    for qubit in range(3):
        if initial_state[::-1][qubit] == "1":
            qc_HSD.x(qr_HSD[qubit])

    qc_HSD.append(D_qc, [qr_HSD[0], qr_HSD[1]])
    qc_HSD.append(B_qc.inverse(), qr_HSD)

    return qc_HSD, qr_HSD

def get_SSD_circuit(time, n_steps, initial_state={"110": 1}):
    '''
    returns the evolution circuit with the Single State Decomposition.
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
    from qiskit import transpile
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