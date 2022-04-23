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
from numpy.linalg import multi_dot 
from numpy import kron as kr
from sympy import Matrix
import os

def permutation_to_matrix(Number_of_qubits, permutations_dict):
    '''
    this function computes the permutation matrix of a permutation of qubits, 
    expressed as a dict: {"starting_state": "permutated"}. for example: {"000":"100", "100":"000"}
    '''
    permutation_matrix = np.eye(2**Number_of_qubits)
    for key in permutations_dict.keys():
        permutation_matrix[int(key, 2),int(key, 2)] = 0
        permutation_matrix[int(permutations_dict[key], 2),int(key, 2)] = 1

    return permutation_matrix