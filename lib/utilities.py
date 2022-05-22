import string
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
from qiskit.quantum_info import state_fidelity, Operator
from scipy.linalg import expm
import os


def get_calibration_circuits(qc, method="NIC", eigenvector=None):
    '''
    Returns a list of calibration circuits for all the methods: CIC, NIC, LIC and qiskit calibration matrix.

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
    N_qubits = len(qc.qubits)
    state_labels = bin_list(N_qubits)

    for state in state_labels:
        cr_cal = ClassicalRegister(N_qubits, name = "c")
        qr_cal = QuantumRegister(N_qubits, name = "q_")
        qc_cal = QuantumCircuit(qr_cal, cr_cal, name=f"mcalcal_{state}")
        if method == "NIC": 
            # first we prepare the eigenstate (if method == "NIC").
            for qubit in range(N_qubits):
                if eigenvector[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
            # then we append the circuit
            qc_cal.append(qc, qr_cal)
            # than we append the gate that bring the eigenstate to the computational basis.
            for qubit in range(N_qubits):
                if eigenvector[::-1][qubit] == "1" and state[::-1][qubit] == "0":
                    qc_cal.x(qr_cal[qubit])
                elif eigenvector[::-1][qubit] == "0" and state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
        # CIC case: first we prepare the initial state than we append the evolution.
        elif method == "CIC": 
            # first we prepare the state.
            for qubit in range(N_qubits):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
            # than we append the circuit
            qc_cal.append(qc, qr_cal)
        
        elif method == "LIC":
        
        elif method == "qiskit":
            for qubit in range(N_qubits):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
        else:
            warnings.warn("a mitigation tecnique must be specified: NIC, CIC or qiskit.")
        # measure all
        qc_cal.measure(qr_cal, cr_cal)
        calib_circuits.append(qc_cal)

    return calib_circuits, state_labels

def get_tensorized_calibration_circuits(qc, method="NIC", eigenvector=None):
    '''
    Returns a list of calibration circuits for all the methods: T-CIC, T-NIC and T-qiskit calibration matrix.

    Args
    ----
        qc (QuantumCircuit): the quantum circuit you wont to calibrate.
        method (string): the method of calibration. Can be CIC, NIC or qiskit.
        eigenvector (string): is a string of binary, example "111". Is the prepared state in the case
                              NIC mitigation tecnique. For CIC and qiskit calibraitions is useless.

    Return
    ----
        calib_circuits (list of QuantumCircuit): list of calibration circuits: 2*N circuits.
        state_labels (list of strings)
    '''

    calib_circuits = []
    state_labels = []

    for Qubit in range(len(qc.qubits)):
        for label in ['0','1']:
            state = ['0', '0', '0']
            state[Qubit] = label
            state = "".join(state)[::-1]
            state_labels.append(state)
            cr_cal = ClassicalRegister(1, name = "c")
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
            qc_cal.measure(qr_cal[Qubit], cr_cal)
            calib_circuits.append(qc_cal)

    return calib_circuits, state_labels


def fidelity_count(result, qcs, target_state):
    '''
    given job result, tomography circuits and targhet state it returns the fidelity score.
    '''
    tomo_ising = StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method="lstsq")
    fid=state_fidelity(rho_fit_ising, target_state)
    return fid

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
        base_matrix_amplitudes = Operator(basis)
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

def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r
