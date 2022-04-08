import numpy as np
from copy import deepcopy
from sympy import *
from sympy.physics.quantum import TensorProduct as Tp
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity


def fixed_magnetization_two_qubit_gate(phase1,phase2,ry_arg):
    ''' 
    '''

    qr=QuantumRegister(2)
    M_qc=QuantumCircuit(qr, name="M")

    M_qc.rz(2*phase1,qr[1])
    M_qc.h(qr[0])
    M_qc.cx(qr[0],qr[1])
    M_qc.ry(ry_arg,qr)
    M_qc.cx(qr[0],qr[1])
    M_qc.h(qr[0])
    M_qc.rz(2*phase2,qr[1])

    return M_qc

def get_gates_parameters(U, initial_state={"110": 1.0}):
    """Finds the parameters of the gates based on the system of equations
    defined by the numerical evolution matrix.
    
    Args
    ----
        U : np.ndarray
            the trotterized evolution matrix
        initial_state : dict
            the initial state to be evolved
    """

    # Builds up the array associated to the initial state
    state = np.zeros(8)
    
    # Creates the 8-dimensional vector associated to the state
    # checking that it is in a magnetization eigenspace
    magnetization = sum([int(_) for _ in initial_state.keys()[0]])
    for base_vector, amplitude in initial_state:
        if sum([int(_) for _ in base_vector]) != magnetization:
            raise ValueError("States must have the same magnetization!")
        state[int(base_vector, 2)] = amplitude
    print(f"get_gates_parameters() - the vector is {state}")

    # Sends an (alpha, beta, gamma) state of fixed magnetization 
    # into a (alpha_prime, beta_prime, gamma_prime) state of the same mag
    state = U.dot(state)

    if magnetization == 2:
        # Checks if all the components are in the mag==2 subspace
        if np.arange(8)[state != 0] != [3,5,6]:
            raise RuntimeError("Something went wrong! State has wrong components")

        alpha_prime, beta_prime, gamma_prime = state[state != 0.0]

        r1 = 0.5*(np.angle(alpha_prime) + np.angle(gamma_prime))
        r2 = 0

        f1 = 0.5*(np.angle(gamma_prime) + np.angle(beta_prime) - np.pi)
        f2 = 0.5*(np.angle(gamma_prime) - np.angle(alpha_prime)) - f1

        a1 = np.arccos(np.abs(gamma_prime))
        a2 = np.arccos(np.abs(beta_prime)/np.sin(a1))

    else: 
        if magnetization == 1:
            A0 = U[4*8+column]
            A1 = U[2*8+column]
            A2 = U[1*8+column]

            r1=float(-atan2(im(A0),re(A0))-atan2(im(A2),re(A2)))/2
            r2=0
            f1=float(atan2(im(A1),re(A1))-atan2(im(A2),re(A2)))/2
            f2=float((atan2(im(A0),re(A0))-atan2(im(A1),re(A1)))/2-f1)
            a1=float(acos(abs(A2)))
            a2=float(acos(abs(A1)/sin(a1)))


    return r1, r2, f1, f2, a1, a2

def jobs_result(job_evolution, reps=1, ancillas=[]):

    backend_sim = Aer.get_backend('qasm_simulator')

    qr=QuantumRegister(7)
    qc=QuantumCircuit(qr)
    qcs = state_tomography_circuits(qc, [qr[1],qr[3],qr[5]])
    for qc in qcs:
        cr = ClassicalRegister(len(ancillas))
        qc.add_register(cr)
        i=0
        for j in ancillas:
            qc.measure(qr[j],cr[i])
            i+=1

    jobs_evo_res = []
    for i in range(reps):
    
        job=execute(qcs,backend=backend_sim, shots=10)
        results = job.result()
        
        for j in range(27):
            results.results[j].data.counts = job_evolution.result().get_counts()[i*27+j]
    
        jobs_evo_res.append(results)

    return jobs_evo_res

def ry(alpha):   # generic ry gate matrix
        return Matrix([ 
            [cos(alpha/2),-sin(alpha/2)],
            [sin(alpha/2),cos(alpha/2)]
        ])

def rz(alpha):   # generic rz gate matrix
    return Matrix([ 
        [exp(-1j*(alpha/2)),0],
        [0,exp(1j*(alpha/2))]
    ])

def H():         # hadamard gate matrix
    return Matrix([ 
        [1/sqrt(2),1/sqrt(2)],
        [1/sqrt(2),-1/sqrt(2)]
    ])

def cx_01():     # c-not(0,1) gate matrix
    return Matrix([
        [1,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0]
    ])


def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r

def DecimalToBinary(num):
    return bin(num).replace("0b", "")

def Toffoli_gate():
    """Builds a modified Toffoli gate adapted to Jakarta geometry"""
    qr=QuantumRegister(3, name="q")
    qc=QuantumCircuit(qr, name="Toffoli")
    qc.t([qr[0],qr[1]])
    qc.h(qr[2])
    qc.t(qr[2])
    qc.cx(qr[0],qr[1])
    qc.cx(qr[1],qr[2])
    qc.tdg(qr[1])
    qc.t(qr[2])
    qc.cx(qr[0],qr[1])
    qc.cx(qr[1],qr[2])
    qc.cx(qr[0],qr[1])
    qc.tdg(qr[2])
    qc.cx(qr[1],qr[2])
    qc.tdg(qr[2])
    qc.cx(qr[0],qr[1])
    qc.cx(qr[1],qr[2])
    qc.h(qr[2])

    return qc
