import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator
from lib import utilities
import warnings

def get_calibration_circuits(qc, method="LIC", eigenvector=None):
    '''
    Returns a list of calibration circuits for all the methods: CIC, NIC, LIC and qiskit calibration matrix.
    Args
    ----
        qc (QuantumCircuit): the quantum circuit you wont to calibrate.
        method (string): the method of calibration. Can be CIC, NIC, LIC or qiskit.
        eigenvector (string): is a string of binary, example "111". Is the prepared state in the case
                              NIC mitigation tecnique. For CIC and qiskit calibraitions is useless.
    Return
    ----
        calib_circuits (list of QuantumCircuit): list of calibration circuits.
    '''
    calib_circuits = []
    N_qubits = len(qc.qubits)
    state_labels = utilities.bin_list(N_qubits)
    if method == "LIC":
        qc_LIC = utilities.LIC_calibration_circuit(qc)
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
            qc_cal.append(qc_LIC, qr_cal)
            for qubit in range(N_qubits):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
        elif method == "qiskit":
            for qubit in range(N_qubits):
                if state[::-1][qubit] == "1":
                    qc_cal.x(qr_cal[qubit])
        else:
            warnings.warn("a mitigation tecnique must be specified: NIC, CIC, LIC or qiskit.")
        # measure all
        qc_cal.measure(qr_cal, cr_cal)
        calib_circuits.append(qc_cal)
    return calib_circuits, state_labels

