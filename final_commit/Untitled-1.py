    # %%
from lib.functions0 import *
from lib.utility import *
import numpy as np
from scipy.linalg import expm
from qiskit.utils.mitigation.fitters import CompleteMeasFitter
from qiskit.ignis.mitigation.measurement import  complete_meas_cal
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.opflow import Zero, One, I, X, Y, Z
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
import matplotlib.pyplot as plt
from qiskit.opflow import Zero, One, I, X, Y, Z
import warnings
warnings.filterwarnings('ignore')
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-community',
                            group='ibmquantumawards', 
                            project='open-science-22')

backend_sim_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))
backend_real_jakarta = provider.get_backend('ibmq_jakarta')
backend_sim = Aer.get_backend('qasm_simulator')

# %%
import qiskit
qiskit.utils.mitigation.fitters.__file__

Passi=1
F_raw=np.zeros(Passi)
F_Qiskit=np.zeros(Passi)
F_Identity=np.zeros(Passi)
F_circuit=np.zeros(Passi)

for m in range(Passi): # %%
    n_steps=100
    time=np.pi*(m+0.0001)/Passi
    #initial_state={"110": 1}
    shots = 8192
    backend = backend_sim_jakarta

    X = np.array([[0,1],[1,0]])  #defining the pauli matrices
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    Id = np.eye(2)

    # defining the hamiltonian divided in: 
    #       - H1: first two qubits interactions.
    #       - H2: second two qubits interactions.

    H1 = np.kron(X, np.kron(X,Id)) + np.kron(Y, np.kron(Y,Id)) + np.kron(Z, np.kron(Z,Id)) 
    H2 = np.kron(Id, np.kron(X,X)) + np.kron(Id, np.kron(Y,Y)) + np.kron(Id, np.kron(Z,Z)) 

    # building numerically the trotter step matrix, and the whole operator (trotter step)^n_steps.

    trotter_step_matrix_= expm(-time/n_steps*H1*1j).dot(expm(-time/n_steps*H2*1j))
    trotterized = np.linalg.matrix_power(trotter_step_matrix_, n_steps)
    Matrix(trotterized).n(3, chop=True)

    # %%
    B = Matrix([
        [0,0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ])
    B

    # %%
    qr=QuantumRegister(3, name="q")
    B_qc=QuantumCircuit(qr, name="B")
    B_qc.x(qr[2])
    B_qc.cx(qr[1],qr[0])
    B_qc.cx(qr[2],qr[1])
    B_qc.cx(qr[1],qr[0])
    B_qc.x([qr[0],qr[1],qr[2]])
    B_qc.append(Toffoli_gate,[qr[0],qr[1],qr[2]])
    B_qc.x([qr[0],qr[1]])

    B_qc.draw(output="mpl")

    # %%
    transpile(B_qc, basis_gates=["cx", "x", "rz", "sx"]).draw(output="mpl")

    # %%
    n_steps = 42
    time = np.pi
    precision = 40

    numeric_evolution_matrix=eye(8)

    #for _ in range(n_steps): # here is computed the evolution operator numerically, with n_steps trotter steps.
    #
    #    numeric_evolution_matrix=(numeric_evolution_matrix*trotter_step_matrix_(2*time/n_steps)).evalf(precision)


    M_N = B*trotterized*B.H
    M_N = Matrix([M_N[0:4],M_N[8:12],M_N[16:20],M_N[24:28]])
    M_N.evalf(5)

    # %%
    qc=QuantumCircuit(2, name="$M^N$")
    qc.unitary(M_N,[0,1])    
    M_N_qc=transpile(qc,basis_gates=['cx','x','sx','rz']) 

    M_N_qc.draw(output="mpl")

    # %%
    initial_state="000"

    qr_U = QuantumRegister(3, name="q")
    qc_U = QuantumCircuit(qr, name="evo")

    ### preparing the initial state

    l=0
    for k in [2,1,0]:
        if initial_state[l]=='1':
            qc.x(qr[k])
        l+=1

    ### appending the evolution

    qc_U.append(M_N_qc,[qr[0],qr[1]])
    qc_U.append(B_qc,[qr[0],qr[1],qr[2]])

    qc_U.draw(output="mpl")

    # %%
    U_ideal=matrix_from_circuit(qc_U, type="numpy")
    Matrix(U_ideal)

    # %%
    qr_evo = QuantumRegister(3, name="q")
    qc_evo = QuantumCircuit(qr_evo, name="U")

    qc_evo.x([qr_evo[1],qr_evo[2]])
    qc_evo.append(qc_U, qr_evo)

    qcs_tomo = state_tomography_circuits(qc_evo, qr_evo)
    qcs_tomo[3].draw(output="mpl")

    # %%
    qr_cal = QuantumRegister(3)
    cal_circ, state_labels = complete_meas_cal(qubit_list=[0,1,2], qr=qr_cal, circlabel='mcal')

    # %%
    cal_circ[0].draw()

    # %%
    state_labels

    # %%
    calib_circuits_identity = []
    calib_circuits_itself = []

    for i in state_labels:

        cr_cal_itself = ClassicalRegister(3)
        qr_cal_itself = QuantumRegister(3)
        qc_cal_itself = QuantumCircuit(qr_cal_itself, cr_cal_itself, name=f"mcalcal_{i}")

        cr_cal_id = ClassicalRegister(3)
        qr_cal_id = QuantumRegister(3)
        qc_cal_id = QuantumCircuit(qr_cal_id, cr_cal_id, name=f"mcalcal_{i}")

        qc_cal_id.x(qr_cal_id)
        qc_cal_id.append(qc_U, qr_cal_id)

        for k in range(3):
            if i[::-1][k] == "1":
                qc_cal_itself.x(qr_cal_itself[k])
            else:
                qc_cal_id.x(qr_cal_id[k])
            
        qc_cal_itself.append(qc_U, qr_cal_itself)
        
        qc_cal_id.measure(qr_cal_id, cr_cal_id)
        qc_cal_itself.measure(qr_cal_itself, cr_cal_itself)

        calib_circuits_identity.append(qc_cal_id)
        calib_circuits_itself.append(qc_cal_itself)

    # %%
    calib_circuits_identity[1].draw()

    # %%
    calib_circuits_itself[1].draw()

    # %%
    job_tomo=execute(qcs_tomo, backend, shots=shots, initial_layout=[1,3,5])

    job_cal_our_identity=execute(calib_circuits_identity, backend = backend, shots=shots, initial_layout=[1,3,5])

    job_cal_our_itself=execute(calib_circuits_itself, backend = backend, shots=shots, initial_layout=[1,3,5])

    job_cal_qiskit=execute(cal_circ, backend, shots=shots, initial_layout=[1,3,5])

    # %%
    meas_fitter_our_identity = CompleteMeasFitter(job_cal_our_identity.result(), state_labels=state_labels)
    meas_fitter_our_itself = CompleteMeasFitter(job_cal_our_itself.result(), state_labels=state_labels)
    meas_fitter_qiskit = CompleteMeasFitter(job_cal_qiskit.result(), state_labels=state_labels)

    # %%
    Matrix(meas_fitter_qiskit.cal_matrix)

    # %%
    Matrix(meas_fitter_our_identity.cal_matrix)

    # %%
    U_tilde_identity=meas_fitter_our_identity.cal_matrix
    U_tilde_itself=meas_fitter_our_itself.cal_matrix
    U_tilde_qiskit=meas_fitter_qiskit.cal_matrix

    # %%
    def matrix_from_cirquit(qc, phase=0, type="sympy"):

        backend = Aer.get_backend('unitary_simulator')
        job = execute(qc, backend, shots=32000)
        result = job.result()
        A=result.get_unitary(qc, decimals=10)*np.exp(1j*phase)
        if type=="sympy":
            return Matrix(A)
        else:
            return A

    # %%
    qr_basi = QuantumRegister(3)
    qc_basi = QuantumCircuit(qr_basi)

    qcs_basis = state_tomography_circuits(qc_basi, qr_basi)

    qcs_basis[0].remove_final_measurements()

    qcs_basis[16].draw()

    # %%
    meas_fitter_qiskit.cal_matrix

    # %%
    C_matrices_itself = []
    C_matrices_identity = []

    C_itself = np.matmul(U_tilde_itself, np.linalg.inv(U_ideal))
    C_identity = U_tilde_identity

    for base in qcs_basis:
        
        base.remove_final_measurements()

        base_matrix = np.matrix(matrix_from_cirquit(base, type="numpy"))
        base_matrix_H = base_matrix.getH()

        C_aus_itself = np.linalg.multi_dot([U_tilde_qiskit, base_matrix, np.linalg.inv(U_tilde_qiskit), U_tilde_itself, np.linalg.inv(U_ideal),  base_matrix_H])
        #C_aus_itself = np.linalg.multi_dot([base_matrix, U_tilde_qiskit, np.linalg.inv(U_tilde_qiskit), C_itself])
        C_aus_identity = np.matmul(base_matrix, np.matmul(C_identity, base_matrix_H))

        C_matrices_identity.append(np.asarray(C_aus_identity))
        C_matrices_itself.append(np.asarray(C_aus_itself))

    # %%
    from copy import deepcopy

    meas_fitters_identity = []
    meas_fitters_itself = []

    for C_new in C_matrices_identity:
        meas_fitter_our_aus = deepcopy(meas_fitter_our_identity)
        meas_fitter_our_aus._tens_fitt.cal_matrices[0]=C_new

        meas_fitters_identity.append(meas_fitter_our_aus)

    for C_new in C_matrices_itself:
        meas_fitter_our_aus = deepcopy(meas_fitter_our_itself)
        meas_fitter_our_aus._tens_fitt.cal_matrices[0]=C_new

        meas_fitters_itself.append(meas_fitter_our_aus)

    # %%
    target_state = (One^One^Zero).to_matrix()
    #target_state = (Zero^One^One).to_matrix()


    fids=np.zeros(4)

    raw_res=deepcopy(job_tomo.result())
    qiskit_res=deepcopy(raw_res)
    identity_res=deepcopy(raw_res)
    itself_res=deepcopy(raw_res)

    # %%
    raw_res.get_counts(-1)

    # %%
    for i in range(27):

        old_counts=raw_res.get_counts(i)
        new_counts_qiskit = meas_fitter_qiskit.filter.apply(old_counts, method="least_squares")
        qiskit_res.results[i].data.counts = new_counts_qiskit

        new_counts_id = meas_fitters_identity[i].filter.apply(old_counts, method="least_squares")
        identity_res.results[i].data.counts = new_counts_id

        new_counts_it = meas_fitters_itself[i].filter.apply(old_counts, method="least_squares")
        itself_res.results[i].data.counts = new_counts_it

    # %%
    itself_res.get_counts(-1)

    # %%
    identity_res.get_counts(-1)

    # %%
    fids[0] = fidelity_count(raw_res, qcs_tomo, target_state)
    fids[1] = fidelity_count(qiskit_res, qcs_tomo, target_state)
    fids[2] = fidelity_count(identity_res, qcs_tomo, target_state)
    fids[3] = fidelity_count(itself_res, qcs_tomo, target_state)
    F_raw[m]=fids[0]
    F_Qiskit[m]=fids[1]
    F_Identity[m]=fids[2]
    F_circuit[m]=fids[3]
    print(m)





def H_heis3():
    # Interactions (I is the identity matrix; X, Y, and Z are Pauli matricies; ^ is a tensor product)
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    
    # Sum interactions
    H = XXs + YYs + ZZs
    
    # Return Hamiltonian
    return H

def U_heis3(t):
    # Compute XXX Hamiltonian for 3 spins in a line
    H = H_heis3()
    
    # Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
    return (t * H).exp_i()

ts = np.linspace(0+0.001, np.pi, Passi)

# Define initial state |110>
initial_state = One^One^Zero

# Compute probability of remaining in |110> state over the array of time points
 # ~initial_state gives the bra of the initial state (<110|)
 # @ is short hand for matrix multiplication
 # U_heis3(t) is the unitary time evolution at time t
 # t needs to be wrapped with float(t) to avoid a bug
 # (...).eval() returns the inner product <110|U_heis3(t)|110>
 #  np.abs(...)**2 is the modulus squared of the innner product which is the expectation value, or probability, of remaining in |110>
probs_110 = [np.abs((~initial_state @ U_heis3(float(t)) @ initial_state).eval())**2 for t in ts]


t=np.linspace(0+0.0001,np.pi,Passi)
plt.figure(figsize=(13,10))

plt.plot(ts, probs_110)
plt.plot(t,np.abs(F_raw),linestyle='',marker='*',label='Raw')
plt.plot(t,np.abs(F_Qiskit),linestyle='',marker='*',label='Qiskit')
plt.plot(t,np.abs(F_Identity),linestyle='',marker='*',label='Identity')
plt.plot(t,np.abs(F_circuit),linestyle='',marker='*',label='Ciruit')
plt.legend()
plt.grid()
plt.show()

    # %%


    # %%


    # %%



