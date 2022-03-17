import numpy as np
from copy import deepcopy
from sympy import *
from sympy.physics.quantum import TensorProduct as Tp
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

def trotter_step(a):
    
    m = Matrix([
        [exp(-I*a),0,0,0],
        [0,cos(a),-I*sin(a),0],
        [0,-I*sin(a),cos(a),0],
        [0,0,0,exp(-I*a)]
    ])

    return Tp(m, eye(2)) * Tp(eye(2), m) * exp(I*a)

def evolution_cirquit(steps=10, time=np.pi, initial_state="110", precision=40):

    ### This function computes numerically the operator obtained with the composition of "steps" trotter steps.
    ### "time" is the total evolution time
    ### "precision" is the digit where every operation will be troncated.

    a1, r1, f1 = symbols("a1 r1 f1")
    a2, r2, f2 = symbols("a2 r2 f2")

    U = eye(8)
    a = symbols("a")

    Trotter_Step = trotter_step(a)

    for _ in range(steps):
        U=U*Trotter_Step
        U=U.subs(a,2*time/steps)
        U=U.evalf(precision)
    
    r1=float(angolo(U[3*8+6])+angolo(U[6*8+6]))/2
    r2=0
    f1=float(angolo(U[6*8+6])-angolo(U[5*8+6])-np.pi)/2
    f2=float((angolo(U[6*8+6])-angolo(U[3*8+6]))/2-f1)
    a1=float(acos(abs(U[6*8+6])))
    a2=float(acos(abs(U[5*8+6])/sin(a1)))

    qr1=QuantumRegister(2)
    M1_qc=QuantumCircuit(qr1, name="M1")

    M1_qc.rz(2*r1,qr1[1])
    M1_qc.h(qr1[0])
    M1_qc.cx(qr1[0],qr1[1])
    M1_qc.ry(a1,qr1)
    M1_qc.cx(qr1[0],qr1[1])
    M1_qc.h(qr1[0])
    M1_qc.rz(2*f1,qr1[1])

    qr2=QuantumRegister(2)
    M2_qc=QuantumCircuit(qr2, name="M2")

    #M2_qc.rz(2*r2,qr2[1])
    M2_qc.h(qr2[0])
    M2_qc.cx(qr2[0],qr2[1])
    M2_qc.ry(a2,qr2)
    M2_qc.cx(qr2[0],qr2[1])
    M2_qc.h(qr2[0])
    M2_qc.rz(2*f2,qr2[1])

    qr = QuantumRegister(7 ,name="q")
    qc = QuantumCircuit(qr, name="U")

    l=0
    for k in [5,3,1]:
        if initial_state[l]=='1':
            qc.x(qr[k])
        l+=1

    qc.append(M1_qc, [qr[1],qr[3]])
    qc.append(M2_qc, [qr[3],qr[5]])

    return qr, qc

def add_check(qc, qr_q, qr_anc, type="copy_check"):

    if type=="copy_check":
        qr_ch=QuantumRegister(6)
        qc_ch=QuantumCircuit(qr_ch,name ='copy_check')

        qc_ch.cx(qr_ch[3],qr_ch[0])
        qc_ch.cx(qr_ch[5],qr_ch[2])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])

        qc.append(qc_ch, qr_anc + qr_q)

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

        qc.append(qc_ch, qr_anc + qr_q)

    if type=="4copy_checkDD": 

        qr_ch=QuantumRegister(7)
        qc_ch=QuantumCircuit(qr_ch,name ='4copy_checkDD')

        qc_ch.x([qr_ch[6],qr_ch[5]])
        qc_ch.cx(qr_ch[4],qr_ch[0])
        qc_ch.cx(qr_ch[4],qr_ch[1])

        qc_ch.barrier()
        qc_ch.x([qr_ch[5]])
        qc_ch.cx(qr_ch[5],qr_ch[4])
        qc_ch.cx(qr_ch[4],qr_ch[1])
        qc_ch.x([qr_ch[1]])
        qc_ch.cx(qr_ch[5],qr_ch[4])
        qc_ch.x([qr_ch[5]])

        qc_ch.barrier()
        qc_ch.x(qr_ch[6])
        qc_ch.cx(qr_ch[6],qr_ch[3])
        qc_ch.barrier()
        qc_ch.cx(qr_ch[6],qr_ch[2])
        qc_ch.x([qr_ch[1]])
        qc_ch.x([qr_ch[5]])

        qc.append(qc_ch, qr_anc + qr_q)
        
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

            qc.barrier()
            l=0
            qubits.reverse()
            for k in qubits:
                if pos_init[i][l]=='1':
                    qc.x(qr[k])
                    qc_1.x(qr_1[k])
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
                    qc_1.x(qr_1[k])
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

    N_ancillas=len(ancillas_conditions[0])
    N_qubit=N_ancillas+3
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



        #if Measure_Mitig=="yes" and meas_fitter != 0:
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
            
            old_counts = meas_fitter.filter.apply(old_counts, method='least_squares')

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





def calibration_cirquit(type="", N=0):
    if type=="complete_evolution":
        #circuito calibrazione per 14-cnot
        qr=QuantumRegister(3)
        qc=QuantumCircuit(qr)    
        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])  

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1]) 

        qc.cx(qr[0],qr[1]) 

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])    

        qc.cx(qr[0],qr[1])   

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])    

        qc.x(qr[1])
        qc.barrier()
        qc.x(qr[1])   

        qc.cx(qr[0],qr[1])    

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])    

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])    

        qc.cx(qr[0],qr[1])   

        qc.barrier()    

        qc.x(qr[2])
        qc.y(qr[2])
        qc.x(qr[2])
        qc.y(qr[2])    

        qc.cx(qr[1],qr[2])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[2])
        qc.cx(qr[0],qr[1])
        qc.barrier()
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[2])
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[2])    

        return qc
    
    if type=="column_evolution_remake":
        qr=QuantumRegister(3)
        qc=QuantumCircuit(qr)

        qc.x([qr[0],qr[1]])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.cx(qr[0],qr[1])
        qc.x([qr[0],qr[1]])
        qc.sx([qr[0],qr[1]])
        qc.barrier()
        qc.sx([qr[0],qr[1]])
        
        qc.cx(qr[0],qr[1])
        qc.x([qr[0],qr[1]])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.barrier()
        qc.x([qr[1],qr[2]])
        qc.sx(qr[1])
        qc.barrier()
        qc.sx(qr[1])
        qc.x(qr[2])
        qc.cx(qr[1],qr[2])
        qc.x([qr[1],qr[2]])
        qc.sx([qr[1],qr[2]])
        qc.barrier()
        qc.sx([qr[1],qr[2]])
        
        qc.cx(qr[1],qr[2])
        qc.x([qr[1],qr[2]])
        qc.sx(qr[1])
        qc.barrier()
        qc.sx(qr[1])
        qc.x(qr[2])

        return qc

    if type=="column_evolution":
        #circuito calibrazione per 4-cnot
        qr=QuantumRegister(3)
        qc=QuantumCircuit(qr)
        qc.x(qr[0])
        qc.y(qr[0])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.y([qr[0],qr[1]])
        qc.barrier()
        qc.x(qr[1])
        qc.y(qr[1])    
        qc.cx(qr[0],qr[1])
        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier()
        qc.x(qr[0])
        qc.y(qr[0])
        qc.cx(qr[0],qr[1])    
        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.barrier()
        qc.y(qr[0])
        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier()
        qc.x(qr[1])
        qc.y(qr[1]) 
        qc.x(qr[1])
        qc.barrier()   
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])  
        qc.x(qr[2])
        qc.y(qr[2])
        qc.barrier()
        qc.x(qr[2])
        qc.y(qr[2])
        qc.cx(qr[1],qr[2])    
        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier() 
        qc.x(qr[1])
        qc.y(qr[1])    
        qc.cx(qr[1],qr[2])    
        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier()
        qc.x(qr[1])
        qc.y(qr[1])    
        qc.x(qr[2])
        qc.y(qr[2])
        qc.barrier()
        qc.x(qr[2])
        qc.y(qr[2])    
        
        return qc

    #if type=="trotter_steps":

    #    return circ_cal_tot(N)


    return "error"

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

def H():         # hadamart gate matrix
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

def angolo(x):
    alpha=re(x)
    beta=im(x)
    if alpha>0:
        return atan(beta/alpha)  
    if alpha<0:
        if beta>=0:
            return atan(beta/alpha)+np.pi
        else:
            return atan(beta/alpha)-np.pi
    if alpha==0:
        if beta>0:
            return np.pi/2
        else:
            return -np.pi/2
    return 0

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r

def DecimalToBinary(num, N_bit):
    b=''
    if num==0:
        b='0'
    while(num>0):
        b=("% s" % (num%2))+b
        num=num//2
    while len(b)<N_bit:
        b='0'+b
    return b


