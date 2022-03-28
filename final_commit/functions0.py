import numpy as np
from utility import *
from copy import deepcopy
from sympy import *
from sympy.physics.quantum import TensorProduct as Tp
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

def trotter_step_matrix(parameter):
    
    '''
    Here is computed the matrix of a single trotter step. It can be done numerically or symbolcally.

    Args:

        - parameter: can be a sympy Symbol or a double.

    Returns:

        - trotter_step_matrix: single trotter steps matrix (symbolic or numeric) with parameter=2*time/N_steps.

    '''

    m = Matrix([
        [exp(-I*parameter),0,0,0],
        [0,cos(parameter),-I*sin(parameter),0],
        [0,-I*sin(parameter),cos(parameter),0],
        [0,0,0,exp(-I*parameter)]
    ])

    trotter_step_matrix = Tp(m, eye(2)) * Tp(eye(2), m) * exp(I*parameter)

    return trotter_step_matrix

def evolution_cirquit(n_steps=10, time=np.pi, initial_state="110", precision=40):

    '''
    This function computes numerically the operator obtained with the composition of "steps" trotter steps,
    and than builds the evolution cirquit with the best decomposition (4 c-not), see "decomposition.ipynb".
    
    Args:

        - n_steps (integer): is the number of trotter steps.
        - time (double): is the total evolution time.
        - initial_state (string): the 3-qubit initial state, from right to left, the characters are associated with qubits 1, 3 and 5 respectively.
        - precision (integer): is the digit where every operation will be troncated.
    
    Returns:

        - qc (QuantumCirquit): evolution circuit.
        
    '''

    numeric_evolution_matrix = eye(8)
    
    for _ in range(n_steps): # here is computed the evolution operator numerically, with n_steps trotter steps.
        numeric_evolution_matrix=(numeric_evolution_matrix*trotter_step_matrix(2*time/n_steps)).evalf(precision)

    # here are computed the parameters of the gates as described in "decomposition.ipynb" file.
    r1, r2, f1, f2, a1, a2 = gates_parameters(initial_state=initial_state, U=numeric_evolution_matrix)

    # defining the two qubits gates that preserve the magnetization when applyed, with the parameters just computed.
    M1_qc = fixed_magnetization_two_qubit_gate(r1,f1,a1)
    M2_qc = fixed_magnetization_two_qubit_gate(r2,f2,a2)

    # defining and building the quantum cirquit.
    qr = QuantumRegister(7 ,name="q")
    qc = QuantumCircuit(qr, name="U")

    # initializing the state as choosen in "initial_state".
    l=0
    for k in [5,3,1]:
        if initial_state[l]=='1':
            qc.x(qr[k])
        l+=1
    
    qc.append(M1_qc, [qr[1],qr[3]])
    qc.append(M2_qc, [qr[3],qr[5]])

    return qr, qc

def add_check(qc, qr_control_qubits, qr_ancillas, type="copy_check"):

    '''
    this function appends a "copy check" with 3 or four qubits to the cirquit given. 
    see the file "Ancillas_Error_mitigation_Git_Hub.ipynb" to understand better how does them work.

    Args:

        qc (QuantumCircuit): The quantum circuit which will be appended the check.
        qr_control_qubits (list of QuantumRegister): list of control qubits
        qr_ancillas (list of QuantumRegister): list of ancillas
        type (string): type of check, options:
                                - "copy_check": copy check with 3 ancillas
                                - "4copy_check": copy check with 3 ancillas
    
    Returns:

        qc (QuantumCircuit): the quantum circuit given in input with appended the check.

    '''

    if type=="copy_check":
        qr_ch=QuantumRegister(6)
        qc_ch=QuantumCircuit(qr_ch,name ='copy_check')

        qc_ch.cx(qr_ch[3],qr_ch[0])
        qc_ch.cx(qr_ch[5],qr_ch[2])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])

        qc.append(qc_ch, qr_ancillas + qr_control_qubits)

    if type=="4copy_check":
        
        qr_ch=QuantumRegister(7)
        qc_ch=QuantumCircuit(qr_ch,name ='4copy_check')

        qc_ch.cx(qr_ch[4],qr_ch[0])
        qc_ch.cx(qr_ch[6],qr_ch[2])
        qc_ch.cx(qr_ch[4],qr_ch[1])
        qc_ch.cx(qr_ch[5],qr_ch[4])
        qc_ch.cx(qr_ch[4],qr_ch[1])
        qc_ch.cx(qr_ch[5],qr_ch[4])
        qc_ch.cx(qr_ch[6],qr_ch[3])

        qc.append(qc_ch, qr_ancillas + qr_control_qubits)
        
    return qc

def calibration_cirquits(type="", q_anc=[], N=0, check="no", check_type="copy_check"):

    c_qc = calibration_cirquit(type=type, N=N)

    qubits = q_anc+[1,3,5]
    N_qubits = len(qubits)
    pos_init = bin_list(N_qubits)

    qcs=[]
    meas=[]

    for i in range(2**N_qubits):

        cr=ClassicalRegister(N_qubits)
        qr=QuantumRegister(N_qubits)
        qc=QuantumCircuit(qr,cr,name='%scal_%s' % ('', DecimalToBinary(i,N_qubits)))

        cr_1=ClassicalRegister(N_qubits)
        qr_1=QuantumRegister(N_qubits)
        qc_1=QuantumCircuit(qr_1,cr_1,name='%scal_%s' % ('', DecimalToBinary(i,N_qubits)))

        l=0
        qubits.reverse()
        for k in qubits:
            if pos_init[i][l]=='1':
                qc.x(qr[k])
                qc_1.x(qr_1[k])
            l+=1
        qubits.reverse()

        qc.append(c_qc, [qr[1],qr[3],qr[5]])

        if check == "yes":

            #qc.barrier()
            l=0
            qubits.reverse()
            for k in qubits:
                if pos_init[i][l]=='1':
                    qc.x(qr[k])
                    #qc_1.x(qr_1[k])
                l+=1
            qubits.reverse()

            if check_type=="copy_check":
                qc = add_check(qc, [qr[1],qr[3],qr[5]], [qr[0],qr[2],qr[4]], type=check_type)
            if check_type=="4copy_check":
                qc = add_check(qc, [qr[1],qr[3],qr[5]], [qr[0],qr[2],qr[4],qr[6]], type=check_type)

            l=0
            qubits.reverse()
            for k in qubits:
                if pos_init[i][l]=='1':
                    qc.x(qr[k])
                    #qc_1.x(qr_1[k])
                l+=1
            qubits.reverse()

        qc.barrier()
        qc_1.barrier()

        i=0
        for k in qubits:
            qc.measure(qr[k],cr[i])
            qc_1.measure(qr_1[k],cr_1[i])
            i+=1

        qcs.append(qc)
        meas.append(qc_1)
    
    return qcs, meas

def mitigate(raw_results, Measure_Mitig="yes", ancillas_conditions=[], meas_fitter=0):

    '''
    This function computes the mitigated results of a job.

    '''

    N_ancillas=len(ancillas_conditions[0]) # number of ancillas qubits
    N_qubit=N_ancillas+3 # total number of qubits
    new_result = deepcopy(raw_results)
    new_result_nm = deepcopy(raw_results)

    ################create the list of the total possible outcomes
    r=bin_list(N_qubit=N_qubit)
    r_split=[]
    for j in range(2**(N_qubit)):
        X_aus=''
        X_aus+=r[j][:N_ancillas]
        X_aus+=' '
        X_aus+=r[j][N_ancillas:]
        r_split.append(X_aus)

    for i in range(len(raw_results.results)): 
        
        old_counts = raw_results.get_counts(i)
        new_counts = {}
        new_counts_nm = {}

        new_result.results[i].header.creg_sizes = [new_result.results[i].header.creg_sizes[0]]
        new_result.results[i].header.clbit_labels = new_result.results[i].header.clbit_labels[0:-1]
        new_result.results[i].header.memory_slots = 3

        new_result_nm.results[i].header.creg_sizes = [new_result_nm.results[i].header.creg_sizes[0]]
        new_result_nm.results[i].header.clbit_labels = new_result_nm.results[i].header.clbit_labels[0:-1]
        new_result_nm.results[i].header.memory_slots = 3

        if N_ancillas>0:
            for reg_key in old_counts:
                reg_bits = reg_key.split(' ')
                if reg_bits[0] in ancillas_conditions:
                    if reg_bits[1] not in new_counts_nm:
                        new_counts_nm[reg_bits[1]]=0
                    new_counts_nm[reg_bits[1]]+=old_counts[reg_key]
        else:
            for reg_key in old_counts:
                new_counts_nm[reg_key]=old_counts[reg_key]

        new_result_nm.results[i].data.counts = new_counts_nm

        if Measure_Mitig=="yes" and meas_fitter != 0:
            if N_ancillas>0:
                for j in range(2**N_qubit):
                    if r_split[j] in old_counts.keys():
                        old_counts[r[j]] = old_counts.pop(r_split[j])
            
            old_counts = meas_fitter.filter.apply(old_counts, method='least_squares') # 'least_squares' or 'pseudo_inverse'

            if N_ancillas>0:
                for j in range(2**N_qubit):
                    if r[j] in old_counts.keys():
                        old_counts[r_split[j]] = old_counts.pop(r[j])

            if N_ancillas>0:
                for reg_key in old_counts:
                    reg_bits = reg_key.split(' ')
                    if reg_bits[0] in ancillas_conditions:
                        if reg_bits[1] not in new_counts:
                            new_counts[reg_bits[1]]=0
                        new_counts[reg_bits[1]]+=old_counts[reg_key]
            else:
                for reg_key in old_counts:
                    new_counts[reg_key]=old_counts[reg_key]
        
            new_result.results[i].data.counts = new_counts

    if Measure_Mitig=="yes" and meas_fitter != 0:
        return new_result, new_result_nm
    else:
        return new_result_nm

def fidelity_count(result, qcs, target_state):
    tomo_ising=StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method='lstsq')
    fid=(state_fidelity(rho_fit_ising, target_state))
    return fid

def circuits_without_ancillas_measuraments(job):
    qcs_without_ancillas = []
    
    for cir in job.circuits():
        circuit=cir.copy()
        circuit.remove_final_measurements()
        c=ClassicalRegister(3, name="c")
        circuit.add_register(c)
        circuit.barrier()
        circuit.measure([1,3,5],c)

        qcs_without_ancillas.append(circuit)
    
    return qcs_without_ancillas
