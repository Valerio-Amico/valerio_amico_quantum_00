import numpy as np
from copy import deepcopy
from qiskit import (
    Aer,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute
)
from qiskit.ignis.verification.tomography import StateTomographyFitter
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




def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)