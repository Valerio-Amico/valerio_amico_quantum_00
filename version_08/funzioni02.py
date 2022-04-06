from audioop import reverse
from copy import deepcopy
import funzioni01 as f1
from sympy import *
from sympy.physics.quantum import TensorProduct as Tp
import qiskit.ignis.mitigation.measurement as mc
import numpy as np
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

def Trotter_N_approx(steps=10, tempo=np.pi, precision=10):

    ### Funzione che calcola numericamente la matrice di evoluzione,
    ### con un numero di trotter steps pari a steps e un tempo di evoluzione pari a tempo,
    ### la precision è la cifra meno signivicativa dove si troncano i risultati di tutte le operazioni.

    q = Symbol("q", positive = True) #q=2*t/N, t=tempo, N=steps
    e = Symbol("e", positive = True)

    Id=eye(2)

    cx_01= Matrix([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
    ])

    m = Matrix([
    [e**(-1j*q),0,0,0],
    [0,e**(-1j*q),0,0],
    [0,0,cos(q),-1j*sin(q)],
    [0,0,-1j*sin(q),cos(q)]
    ])


    Trotter_step=Tp(Id,cx_01*m*cx_01)*Tp(cx_01*m*cx_01,Id)
    U= Tp(Id,Tp(Id,Id))

    for _ in range(steps):
        # print(steps,i)
        U=U*Trotter_step
        U=U.subs(q,2*tempo/steps)
        U=U.subs(e,np.e)
        U=U.evalf(precision)

    return U

def simplyfied_gates_matricies(U, precision=20):

    e=Symbol("e", positive=True)

    r_1 = Symbol("r1")#, positive = True)
    phi_1 = Symbol("f1")#, positive = True)
    a_1 = Symbol("a1")#, positive = True)

    r_2 = Symbol("r2")#, positive = True)
    phi_2 = Symbol("f2")#, positive = True)
    a_2 = Symbol("a2")#, positive = True)

    cx_01= Matrix([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0]
    ])

    m_1 = Matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,e**(1j*a_1)*cos(r_1),e**(1j*phi_1)*sin(r_1)],
    [0,0,-e**(-1j*phi_1)*sin(r_1),e**(-1j*a_1)*cos(r_1)]
    ])

    m_2 = Matrix([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,e**(1j*a_2)*cos(r_2),e**(1j*phi_2)*sin(r_2)],
    [0,0,-e**(-1j*phi_2)*sin(r_2),e**(-1j*a_2)*cos(r_2)]
    ])

    ### considero la colonna relativa allo stato iniziale desiderato
    ### in questo caso |110>  !!!!!DEVO IMPLEMENTARE QUESTA PARTE PER TUTTI GLI STATI!!!!!!!

    b0=im(U[3*8+6])
    b1=im(U[5*8+6])
    alp0=re(U[3*8+6])
    alp1=re(U[5*8+6])
    alp2=re(U[6*8+6])

    ### calcolo i coefficienti per i gate a 2 qubit

    r1=acos(sqrt(alp2**2+b1**2))
    a1=angolo(alp2,b1)
    a2=0
    phi1=np.pi+a2-angolo(alp1,b1)
    r2=acos(sqrt(alp1**2+b1**2)/sin(r1))
    phi2=-phi1-angolo(alp0,b0)
    
    ### calcolo numericamente i due gate

    gate_1=cx_01*m_1*cx_01
    gate_1=gate_1.subs(r_1,r1)
    gate_1=gate_1.subs(a_1,a1)
    gate_1=gate_1.subs(phi_1,phi1)
    gate_1=gate_1.subs(e,np.e)

    gate_1=gate_1.evalf(precision)

    gate_2=cx_01*m_2*cx_01
    gate_2=gate_2.subs(r_2,r2)
    gate_2=gate_2.subs(a_2,a2)
    gate_2=gate_2.subs(phi_2,phi2)
    gate_2=gate_2.subs(e,np.e)

    gate_2=gate_2.evalf(precision)

    return gate_1, gate_2

def angolo(alpha, beta):

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

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r

def calibration_measure_mitigation(qubit=[1,3,5],backend_calibration="", shots=32000):
    if backend_calibration=="":
        print("Error!!")
        print("Please, give a backend for the simulation, with the label \"backend_calibration=...\" ")
    
    #qubit_aus=qubit
    #for j in range

    print("Procedura di calibrazione in corso!")
    meas_calibs, state_lables = mc.complete_meas_cal(qubit_list=qubit)
    job_cal=execute(meas_calibs, backend=backend_calibration, shots=shots)
    meas_fitter = mc.CompleteMeasFitter(job_cal.result(), state_labels=state_lables)
    print("Calibrazione completata!")
    return meas_fitter

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

def Mdg_circquit():
    qr=QuantumRegister(3, name="q")
    Mdg_qc=QuantumCircuit(qr, name="Mdg")
    Mdg_qc.x(qr[2])
    Mdg_qc.cx(qr[1],qr[0])
    Mdg_qc.cx(qr[2],qr[1])
    Mdg_qc.cx(qr[1],qr[0])
    Mdg_qc.x([qr[0],qr[1],qr[2]])
    Mdg_qc.append(Toffoli_gate(),[qr[0],qr[1],qr[2]])
    Mdg_qc.x([qr[0],qr[1]])
    
    return Mdg_qc

def semplyfied_gates(U, type, precision):

    if type=="transpile":
        ### building gate_1 cirquit
        gate_1, gate_2 = simplyfied_gates_matricies(U, precision=precision)

        qc_aus=QuantumCircuit(2, name="gate_1")
        qc_aus.unitary(gate_1, [0,1])

        gate_1=transpile(qc_aus,basis_gates=["cx","rz","x","sx"])

        ### building gate_2 cirquit

        qc_aus=QuantumCircuit(2, name="gate_2")
        qc_aus.unitary(gate_2, [0,1])

        gate_2=transpile(qc_aus,basis_gates=["cx","rz","x","sx"])

        return gate_1, gate_2

    if type == "esatto":

        ### considero la colonna relativa allo stato iniziale desiderato
        ### in questo caso |110>  !!!!!DEVO IMPLEMENTARE QUESTA PARTE PER TUTTI GLI STATI!!!!!!!

        b0=im(U[3*8+6])
        b1=im(U[5*8+6])
        alp0=re(U[3*8+6])
        alp1=re(U[5*8+6])
        alp2=re(U[6*8+6])

        ### calcolo i coefficienti per i gate a 2 qubit

        r1=acos(sqrt(alp2**2+b1**2))
        a1=angolo(alp2,b1)
        a2=0
        phi1=np.pi+a2-angolo(alp1,b1)
        r2=acos(sqrt(alp1**2+b1**2)/sin(r1))
        phi2=-phi1-angolo(alp0,b0)

        a_1_1 = -float(- a1 + phi1 - np.pi)/2
        a_2_1 = float(a1 + phi1 - np.pi)/2
        a_3_1 = float(r1)

        a_1_2 = -float(- a2 + phi2 - np.pi)/2
        a_2_2 = float(a2 + phi2 - np.pi)/2
        a_3_2 = float(r2)

        qr1=QuantumRegister(2)
        qc1=QuantumCircuit(qr1, name="gate_1")

        qc1.rz(a_1_1*2,qr1[1])
        qc1.h(qr1[0])
        qc1.cx(qr1[0],qr1[1])
        qc1.ry(-a_3_1,qr1[0])
        qc1.ry(-a_3_1,qr1[1])
        qc1.cx(qr1[0],qr1[1])
        qc1.h(qr1[0])
        qc1.rz(a_2_1*2,qr1[1])

        qr=QuantumRegister(2)
        qc=QuantumCircuit(qr, name="gate_2")

        qc.rz(a_1_2*2,qr[1])
        qc.h(qr[0])
        qc.cx(qr[0],qr[1])
        qc.ry(-a_3_2,qr[0])
        qc.ry(-a_3_2,qr[1])
        qc.cx(qr[0],qr[1])
        qc.h(qr[0])
        qc.rz(a_2_2*2,qr[1])

        return qc1, qc
    
    if type == "DD":

        b0=im(U[3*8+6])
        b1=im(U[5*8+6])
        alp0=re(U[3*8+6])
        alp1=re(U[5*8+6])
        alp2=re(U[6*8+6])

        ### calcolo i coefficienti per i gate a 2 qubit

        r1=acos(sqrt(alp2**2+b1**2))
        a1=angolo(alp2,b1)
        a2=0
        phi1=np.pi+a2-angolo(alp1,b1)
        r2=acos(sqrt(alp1**2+b1**2)/sin(r1))
        phi2=-phi1-angolo(alp0,b0)

        a_1_1 = -float(- a1 + phi1 - np.pi)/2
        a_2_1 = float(a1 + phi1 - np.pi)/2
        a_3_1 = float(r1)

        a_1_2 = -float(- a2 + phi2 - np.pi)/2
        a_2_2 = float(a2 + phi2 - np.pi)/2
        a_3_2 = float(r2)

        qr1=QuantumRegister(3)
        qc1=QuantumCircuit(qr1, name="gate_1")

        qc1.rz(a_1_1*2,qr1[1])
        qc1.h(qr1[0])

        qc1.cx(qr1[0],qr1[1])
        qc1.ry(-a_3_1,qr1[0])
        qc1.ry(-a_3_1,qr1[1])

        qc1.cx(qr1[0],qr1[1])
        qc1.h(qr1[0])
        qc1.rz(a_2_1*2,qr1[1])

        for _ in range(3):
            qc1.x(qr1[2])
            qc1.barrier(qr1[2])
            qc1.x(qr1[2])
            qc1.barrier(qr1[2])

        qr=QuantumRegister(3)
        qc=QuantumCircuit(qr, name="gate_2")

        qc.rz(a_1_2*2,qr[1])
        qc.h(qr[0])

        qc.cx(qr[0],qr[1])
        qc.ry(-a_3_2,qr[0])
        qc.ry(-a_3_2,qr[1])

        qc.cx(qr[0],qr[1])
        qc.h(qr[0])
        qc.rz(a_2_2*2,qr[1])

        for _ in range(5):
            qc.x(qr[2])
            qc.barrier(qr[2])
            qc.x(qr[2])
            qc.barrier(qr[2])

        return qc1, qc

    return "error"

def column_evolution_tomo(steps, tempo, precision, initial_state='110', check=[0]):

    ### check is a list:   check=["check_type", qubits_ancilla=[]]
    ### if check==[0] there is no check
    if initial_state != "110": 
        print("warning! the state is always initialide as 110. Change the function!")

    ### Hisemberg evolution with single column decomposition

    U = Trotter_N_approx(steps=steps, tempo=tempo, precision=precision)

    gate_1, gate_2 = semplyfied_gates(U, type="esatto", precision=precision)

    ### building the evolution cirquit

    qr=QuantumRegister(7, name="q")
    qc=QuantumCircuit(qr, name="U")

    ### preparing the initial state

    #l=0
    #for k in [5,3,1]:
    #    if initial_state[l]=='1':
    #        qc.x(qr[k])
    #    l+=1
    if initial_state=="000":
    ### appending the evolution

        qc.append(gate_1, [qr[1],qr[3]])
        qc.barrier()
        qc.append(gate_2, [qr[3],qr[5]])

    else:
            
        qc.x(qr[3])

        ### appending the evolution

        qc.append(gate_1, [qr[1],qr[3]])
        qc.barrier()
        qc.x(qr[5])
        qc.append(gate_2, [qr[3],qr[5]])

    ### macking the tomography if there is no check

    if check==[0] or check==0 or check == []:
        qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
        return qcs

    ### else append the check
    anc=check[1]
    N_anc=len(anc)

    qc.barrier()
    qc = add_check(qc, [qr[1],qr[3],qr[5]], anc, type=check[0])

    ### macking the tomography

    qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
    qcs_na = deepcopy(qcs)

    for qc_ in qcs:
        cr_anc=ClassicalRegister(N_anc)
        qc_.add_register(cr_anc)
        qc_.barrier()
        qc_.measure(anc,cr_anc)

    return qcs, qcs_na

def complete_evolution_tomo(steps, tempo, precision=20, initial_state='110', check=[0]):
    ### check is a list:   check=["check_type", qubits_ancilla=[]]
    ### if check==[0] there is no check


    ### Hisemberg evolution with complete decomposition

    U = Trotter_N_approx(steps=steps, tempo=tempo, precision=precision)

    U2=[
        [U[3*8+3],U[3*8+5],U[3*8+6],0],
        [U[5*8+3],U[5*8+5],U[5*8+6],0],
        [U[6*8+3],U[6*8+5],U[6*8+6],0],
        [0,0,0,1]
    ]

    qc=QuantumCircuit(2, name="U")
    qc.unitary(U2,[0,1])    
    trans_qc=transpile(qc,basis_gates=['cx','x','sx','rz']) 

    ### building the evolution cirquit

    qr=QuantumRegister(7, name="q")
    qc=QuantumCircuit(qr, name="U")

    ### preparing the initial state

    l=0
    for k in [5,3,1]:
        if initial_state[l]=='1':
            qc.x(qr[k])
        l+=1

    ### appending the evolution

    qc.append(trans_qc,[qr[1],qr[3]])
    qc.append(Mdg_circquit().inverse(),[qr[1],qr[3],qr[5]])

    ### macking the tomography if there is no check

    if check==[0] or check==0 or check==[]:
        qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
        return qcs

    ### else append the check

    anc=check[1]
    N_anc=len(anc)

    qc = add_check(qc, [qr[1],qr[3],qr[5]], anc, type=check[0])

    ### macking the tomography

    qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
    qcs_na = deepcopy(qcs)

    for qc_ in qcs:
        cr_anc=ClassicalRegister(N_anc)
        qc_.add_register(cr_anc)
        qc_.barrier()
        qc_.measure(anc,cr_anc)

    return qcs, qcs_na

def evolution_tomo(type, N_steps, tempo, precision=20, initial_state='110', check=[0]):
    if type == "column_evolution":
        return column_evolution_tomo(steps=N_steps, tempo=tempo, precision=precision, initial_state=initial_state, check=check)
    if type == "complete_evolution":
        return complete_evolution_tomo(steps=N_steps, tempo=tempo, precision=precision, initial_state=initial_state, check=check)
    if type == "trotter_steps":
        if check != []:
            return "with type evolution = \"trotter_steps\" is not possible to add checks yet"
        return f1.U_approx_tomo(steps=N_steps,trot_type="our",time=tempo,checks=[],initial_state=initial_state[::-1])
    return "errore"
    
def mitigate(raw_results, ancillas_conditions=[], meas_fitter=None):

    if meas_fitter is None:
        # Mitigazione senza fitter
        pass

    if "__tens_fitt" in dir(meas_fitter):
        # Mitigazione con fitter singolo
        pass

    if '__iter__' in dir(meas_fitter):
        # Mitigazione con lista di filtri
        pass

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

        # cambia il numero di classical bit dei reults

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

        if meas_fitter is not None:
            if N_ancillas>0:
                for j in range(2**N_qubit):
                    if r_split[j] in old_counts.keys():
                        old_counts[r[j]] = old_counts.pop(r_split[j])

            print(i)

            if "_tens_fitt" in dir(meas_fitter):
                #print(old_counts)
                print("tens_fitt")
                old_counts = meas_fitter.filter.apply(old_counts, method='least_squares')
                
            
            if '__iter__' in dir(meas_fitter):
                print(i, np.shape(meas_fitter[i].filter._cal_matrix))
                meas_fitter_aus = deepcopy(meas_fitter[i])
                print("deepcopy")
                
                old_counts = meas_fitter_aus.filter.apply(old_counts, method='least_squares')

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

    if meas_fitter is not None:
        return new_result, new_result_nm
    else:
        return new_result_nm




def fidelity_count(result, qcs, target_state):
    tomo_ising=StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method='lstsq')
    #fid=(state_fidelity(rho_fit_ising, target_state))
    print(rho_fit_ising)
    fid=state_fidelity(target_state, rho_fit_ising)
    print("attenzione ho invertito gli arg di state_fidelity")
    return fid

def circ_cal_trot():
    qr=QuantumRegister(3)
    qc=QuantumCircuit(qr)
    qc.x(qr[0])
    qc.y(qr[0])
    qc.y(qr[0])
    qc.x(qr[0])    
    qc.x(qr[1])
    qc.y(qr[1])
    qc.y(qr[1])
    qc.x(qr[1])    
    qc.barrier()    
    qc.x(qr[1])
    qc.y(qr[1])
    qc.y(qr[1])
    qc.x(qr[1])    
    qc.cx(qr[1],qr[0])
    qc.cx(qr[0],qr[1])
    qc.cx(qr[1],qr[0])    
    qc.barrier()
    qc.cx(qr[1],qr[0])
    qc.cx(qr[0],qr[1])
    qc.cx(qr[1],qr[0])    
    qc.barrier()    
    qc.sx(qr[2])
    qc.barrier()
    qc.sx(qr[2])
    qc.y(qr[2])
    qc.sx(qr[2])
    qc.barrier()
    qc.sx(qr[2])
    qc.y(qr[2])    
    qc.cx(qr[2],qr[1])
    qc.cx(qr[1],qr[2])
    qc.cx(qr[2],qr[1])    
    qc.barrier()
    qc.cx(qr[2],qr[1])
    qc.cx(qr[1],qr[2])
    qc.cx(qr[2],qr[1])    
    
    return qc

def circ_cal_tot(steps):
    qr=QuantumRegister(3)
    qc=QuantumCircuit(qr)
    cal=circ_cal_trot()
    for i in range(int(steps/2)):
        qc.append(cal,[qr[0],qr[1],qr[2]])    
    
    return qc

def calibration_cirquit(type="", N_steps=0, time=np.pi):
    from qiskit import Aer, QuantumCircuit, QuantumRegister, execute

    if type=="itself":
        
        a=Symbol("a", real=True)

        U = eye(8)
        Trotter_Step = trotter_step_matrix(a)

        for _ in range(N_steps):
            U=U*Trotter_Step
            U=U.subs(a,2*time/N_steps)
            U=U.evalf(40)

        U.evalf(4)

        r1_=float(atan2(im(U[3*8+6]),re(U[3*8+6]))+atan2(im(U[6*8+6]),re(U[6*8+6])))/2
        r2_=0
        f1_=float(atan2(im(U[6*8+6]),re(U[6*8+6]))-atan2(im(U[5*8+6]),re(U[5*8+6]))-np.pi)/2
        f2_=float((atan2(im(U[6*8+6]),re(U[6*8+6]))-atan2(im(U[3*8+6]),re(U[3*8+6])))/2-f1_)
        a1_=float(acos(abs(U[6*8+6])))
        a2_=float(acos(abs(U[5*8+6])/sin(a1_)))

        qr1=QuantumRegister(2)
        M1_qc=QuantumCircuit(qr1, name="M1")

        M1_qc.rz(2*r1_,qr1[1])
        M1_qc.h(qr1[0])
        M1_qc.cx(qr1[0],qr1[1])
        M1_qc.ry(a1_,qr1)
        M1_qc.cx(qr1[0],qr1[1])
        M1_qc.h(qr1[0])
        M1_qc.rz(2*f1_,qr1[1])

        qr2=QuantumRegister(2)
        M2_qc=QuantumCircuit(qr2, name="M2")

        #M2_qc.rz(2*r2,qr2[1])
        M2_qc.h(qr2[0])
        M2_qc.cx(qr2[0],qr2[1])
        M2_qc.ry(a2_,qr2)
        M2_qc.cx(qr2[0],qr2[1])
        M2_qc.h(qr2[0])
        M2_qc.rz(2*f2_,qr2[1])

        qr = QuantumRegister(3 ,name="q")
        qc = QuantumCircuit(qr, name="U")

        qc.append(M1_qc, [qr[0],qr[1]])
        qc.append(M2_qc, [qr[1],qr[2]])

        return qc

    if type=="itself_whole":
        U = Trotter_N_approx(steps=N_steps, tempo=time, precision=45)

        U2=[
            [U[3*8+3],U[3*8+5],U[3*8+6],0],
            [U[5*8+3],U[5*8+5],U[5*8+6],0],
            [U[6*8+3],U[6*8+5],U[6*8+6],0],
            [0,0,0,1]
        ]
        from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister
        qc=QuantumCircuit(2)
        qc.unitary(U2,[0,1])    
        trans_qc=transpile(qc,basis_gates=['cx','x','sx','rz']) 

        ### building the evolution cirquit

        qr=QuantumRegister(3, name="q")
        qc=QuantumCircuit(qr, name="U")


        ### appending the evolution

        qc.append(trans_qc,[qr[0],qr[1]])
        qc.append(Mdg_circquit().inverse(),[qr[0],qr[1],qr[2]])

        return qc



    if type=="complete_evolution":
        #circuito calibrazione per 14-cnot
        qr=QuantumRegister(3)
        qc=QuantumCircuit(qr)    
        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])  

        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier(qr[1])
        qc.x(qr[1])
        qc.y(qr[1]) 

        qc.cx(qr[0],qr[1]) 
        qc.barrier()

        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])    

        qc.cx(qr[0],qr[1])   
        qc.barrier()

        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])    

        qc.x(qr[1])
        qc.barrier()
        qc.x(qr[1])   

        qc.cx(qr[0],qr[1])    

        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])    

        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])    

        qc.cx(qr[0],qr[1])   
        qc.barrier()

        qc.x(qr[2])
        qc.y(qr[2])
        qc.barrier(qr[2])
        qc.x(qr[2])
        qc.y(qr[2])    

        qc.cx(qr[1],qr[2])
        qc.cx(qr[0],qr[1])
        qc.barrier()
        qc.cx(qr[1],qr[2])
        qc.cx(qr[0],qr[1])
        qc.barrier()
        qc.cx(qr[0],qr[1])
        qc.cx(qr[1],qr[2])
        qc.barrier()
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

    if type=="trotter_steps":

        return circ_cal_tot(N)


    return "error"

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

def calibration_cirquits(type="", n_steps=0, time=np.pi, q_anc=[], check="no", check_type="copy_check"):
    ### attenzione correggere n_step: è 0 a posta per dare errori.
    precision=45
    initial_state = "110"

    c_qc = calibration_cirquit(type=type, N_steps=n_steps, time=time)

    qubits = [1,3,5]+q_anc
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
                #qc_1.x(qr_1[k])
            l+=1
        qubits.reverse()

        qc.append(c_qc, [qr[1],qr[3],qr[5]])

        ######## parte nuova in cui aggiungo i c-not del check e le x opportune affinchè il circuito rimanga una identità
        if check == "yes":

            qc.barrier()
            
            l=0
            if type!="itself" and type!="itself_whole":
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
            if type!="itself" and type!="itself_whole":
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

def Toffoli_gate():
    qr=QuantumRegister(3)
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

def matrix_from_cirquit(qc, phase=0, type="sympy"):

    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend, shots=32000)
    result = job.result()
    A=result.get_unitary(qc, decimals=10)*np.exp(1j*phase)
    if type=="sympy":
        return Matrix(A)
    else:
        return A

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



def BinaryToDecimal(bin):
    d=0
    for i in range(len(bin)):
        if bin[-1-i]=='1':
            d+=2**i
    return d

def Magnetization(bin):
    p=0
    for i in range(len(bin)):
        if bin[i]=='1':
            p+=1
    return p

def get_gates_parameters(initial_state, U):

    #a1, r1, f1_, a2, r2, f2_ = symbols("a1 r1 f1 a2 r2 f2", real=True)

    magnetization = Magnetization(initial_state)
    column = BinaryToDecimal(initial_state)

    if magnetization == 0:
        A0 = U[3*8+column]
        A1 = U[5*8+column]
        A2 = U[6*8+column]

        r1j=float(atan2(im(A0),re(A0))+atan2(im(A2),re(A2)))/2
        r2=0
        f1_=float(atan2(im(A2),re(A2))-atan2(im(A1),re(A1))-np.pi)/2
        f2_=float((atan2(im(A2),re(A2))-atan2(im(A0),re(A0)))/2-f1_)
        a1=float(acos(abs(A2)))
        a2=float(acos(abs(A1)/sin(a1)))

    else: 
        if magnetization == 1:
            A0 = U[4*8+column]
            A1 = U[2*8+column]
            A2 = U[1*8+column]

            r1j=float(-atan2(im(A0),re(A0))-atan2(im(A2),re(A2)))/2
            r2=0
            f1_=float(atan2(im(A1),re(A1))-atan2(im(A2),re(A2)))/2
            f2_=float((atan2(im(A0),re(A0))-atan2(im(A1),re(A1)))/2-f1_)
            a1=float(acos(abs(A2)))
            a2=float(acos(abs(A1)/sin(a1)))

    return r1j, r2, f1_, f2_, a1, a2

def ZNE_cirquits(type, N_steps, tempo, points_fit=4, precision=20, initial_state='110', check=[0]):

    if initial_state != "110": 
        print("warning! the state is always initialide as 110. Change the function!")

    ### Hisemberg evolution with single column decomposition

    U = Trotter_N_approx(steps=N_steps, tempo=tempo, precision=precision)
    gate_1, gate_2 = semplyfied_gates(U, type="esatto", precision=precision)
    id_cal = calibration_cirquit(type=type)

    ### building the evolution cirquit

    zne_qcs=[]
    zne_qcs_na=[]

    for i in range(points_fit):

        qr=QuantumRegister(7, name="q")
        qc=QuantumCircuit(qr, name="U")

        
        ### appending the evolution
        qc.x(qr[3])
        qc.append(gate_1, [qr[1],qr[3]])
        qc.barrier()
        qc.x(qr[5])
        qc.append(gate_2, [qr[3],qr[5]])

        for _ in range(i):
            qc.append(id_cal,[qr[1],qr[3],qr[5]])
        
        if check==[0] or check==0 or check == []:
            qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
            zne_qcs=zne_qcs+qcs
        
        else:
            anc=check[1]
            N_anc=len(anc)

            qc = add_check(qc, [qr[1],qr[3],qr[5]], anc, type=check[0])

            ### macking the tomography

            qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
            qcs_na = deepcopy(qcs)

            for qc_ in qcs:
                cr_anc=ClassicalRegister(N_anc)
                qc_.add_register(cr_anc)
                qc_.barrier()
                qc_.measure(anc,cr_anc)

            zne_qcs=zne_qcs+qcs
            zne_qcs_na=zne_qcs_na+qcs_na

    if check==[0] or check==0 or check == []:
        return zne_qcs
    else:
        return zne_qcs, zne_qcs_na

def DD_cirquits(N_steps, tempo, precision=20, initial_state='110', check=[0]):
    if initial_state != "110": 
        print("warning! the state is always initialide as 110. Change the function!")

    ### Hisemberg evolution with single column decomposition

    U = Trotter_N_approx(steps=N_steps, tempo=tempo, precision=precision)
    gate_1, gate_2 = semplyfied_gates(U, type="DD", precision=precision)

    ### building the evolution cirquit

    qr=QuantumRegister(7, name="q")
    qc=QuantumCircuit(qr, name="U")

    
    ### appending the evolution
    qc.x(qr[3])
    qc.append(gate_1, [qr[1],qr[3],qr[5]])
    #qc.barrier()
    qc.x(qr[5])
    qc.append(gate_2, [qr[3],qr[5],qr[1]])

    if check==[0] or check==0 or check == []:
        qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
   
    else:
        anc=check[1]
        N_anc=len(anc)

        qc = add_check(qc, [qr[1],qr[3],qr[5]], anc, type=check[0])

        ### macking the tomography

        qcs=state_tomography_circuits(qc,[qr[1],qr[3],qr[5]])
        qcs_na = deepcopy(qcs)

        for qc_ in qcs:
            cr_anc=ClassicalRegister(N_anc)
            qc_.add_register(cr_anc)
            qc_.barrier()
            qc_.measure(anc,cr_anc)

    if check==[0] or check==0 or check == []:
        return qcs
    else:
        return qcs, qcs_na

