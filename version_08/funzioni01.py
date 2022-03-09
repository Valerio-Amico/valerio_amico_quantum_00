import qiskit.ignis.mitigation.measurement as mc
from IPython.display import clear_output
import qiskit.quantum_info as qi
from distutils.log import error
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
#from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.providers.ibmq.job import job_monitor
#from qiskit.providers.ibmq import least_busy
#from qiskit.opflow import Zero, One, I, X, Y, Z
#from qiskit.providers.aer import AerSimulator, QasmSimulator
from copy import deepcopy
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.circuit import Parameter
#from qiskit.result import marginal_counts
import numpy as np
from math import pi as pi

#N=Parameter('N') # Trotter steps

def Trotter_step(type="our", N=10, time = np.pi):
    ############################################### making the trotter step
    if type=="our":

        Trot_qr=QuantumRegister(3)
        Trot_qc=QuantumCircuit(Trot_qr,name="Trotter_step")

        Trot_qc.rz(-pi/2,Trot_qr[1])
        Trot_qc.cx(Trot_qr[1],Trot_qr[0])
        Trot_qc.ry(-pi/2-2*time/N,Trot_qr[1])
        Trot_qc.rz(2*time/N+pi/2,Trot_qr[0])
        Trot_qc.cx(Trot_qr[0],Trot_qr[1])
        Trot_qc.ry(2*time/N+pi/2,Trot_qr[1])
        Trot_qc.cx(Trot_qr[1],Trot_qr[0])
        Trot_qc.rz(pi/2,Trot_qr[0])

        Trot_qc.rz(-pi/2,Trot_qr[2])
        Trot_qc.cx(Trot_qr[2],Trot_qr[1])
        Trot_qc.ry(-pi/2-2*time/N,Trot_qr[2])
        Trot_qc.rz(2*time/N+pi/2,Trot_qr[1])
        Trot_qc.cx(Trot_qr[1],Trot_qr[2])
        Trot_qc.ry(2*time/N+pi/2,Trot_qr[2])
        Trot_qc.cx(Trot_qr[2],Trot_qr[1])
        Trot_qc.rz(pi/2,Trot_qr[1])

        return Trot_qc

    if "decomposed" in type: 
        st="%d"%N
        f = open("output/U"+st+".txt", "r")
    
        gates=[]
        i=0
        
        for x in f:
            gates.append([])
            y=x.split(", ")
            y[0]=y[0].split(" ")
            y[1]=y[1].replace("\n","")
            y[1]=y[1].replace("q[","")
            y[1]=y[1].replace("]","")
            
            y[0][1]=y[0][1].replace("]","")
            y[0][1]=y[0][1].replace("q[","")
            y[0][1]=fpos(int(y[0][1]))
            gates[i].append(y[0][0])
            gates[i].append(y[0][1])
            gates[i].append(y[1])
            
            #print(y)
            i+=1
        

        qr=QuantumRegister(3, "q")
        qc=QuantumCircuit(qr, name = "U%d_trotter"%N)

        for gate in gates:
            if gate[0] == 'rz':
                qc.rz(float(gate[2]),qr[gate[1]])

            if gate[0] == 'ry':
                qc.ry(float(gate[2]),qr[gate[1]])

            if gate[0] == 'rx':
                qc.rx(float(gate[2]),qr[gate[1]])
            
            if gate[0] == 'cnot':
                targhet=fpos(int(gate[2]))
                if targhet == 1 and gate[1]==5:
                    qc.cx(qr[3],qr[targhet])
                    qc.cx(qr[5],qr[3])
                    qc.cx(qr[3],qr[targhet])
                    qc.cx(qr[5],qr[3])
                    
                else:
                    if targhet == 5 and gate[1]==1:
                        qc.cx(qr[3],qr[targhet])
                        qc.cx(qr[1],qr[3])
                        qc.cx(qr[3],qr[targhet])
                        qc.cx(qr[1],qr[3])
                        
                    else:
                        qc.cx(qr[gate[1]],qr[targhet])
        return qc

    if type == "IBM":

        t=time/N

        # Build a subcircuit for ZZ(t) two-qubit gate
        ZZ_qr = QuantumRegister(2)
        ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

        ZZ_qc.cnot(0,1)
        ZZ_qc.rz(2 * t, 1)
        ZZ_qc.cnot(0,1)

        # Build a subcircuit for YY(t) two-qubit gate
        YY_qr = QuantumRegister(2)
        YY_qc = QuantumCircuit(YY_qr, name='YY')

        YY_qc.rx(np.pi/2,[0,1])
        YY_qc.cnot(0,1)
        YY_qc.rz(2 * t, 1)
        YY_qc.cnot(0,1)
        YY_qc.rx(-np.pi/2,[0,1])

        # Build a subcircuit for XX(t) two-qubit gate
        XX_qr = QuantumRegister(2)
        XX_qc = QuantumCircuit(XX_qr, name='XX')

        XX_qc.ry(np.pi/2,[0,1])
        XX_qc.cnot(0,1)
        XX_qc.rz(2 * t, 1)
        XX_qc.cnot(0,1)
        XX_qc.ry(-np.pi/2,[0,1])

        #Trot_qr = QuantumRegister(3)
        Trot_qr=QuantumRegister(3)
        Trot_qc=QuantumCircuit(Trot_qr,name="Trotter_step")

        for i in range(2):
                Trot_qc.append(XX_qc,[i,i+1])
                Trot_qc.append(YY_qc,[i,i+1])
                Trot_qc.append(ZZ_qc,[i,i+1])

        return Trot_qc
    print("errore!!! nessua trotterizzazione specificata")
    return

def final_state_vector(N_steps, time, initial_state= "110"):

    qc_gate  = Trotter_step(type="IBM", N=N_steps, time = time)

    qr=QuantumRegister(3)
    qc=QuantumCircuit(qr,name="U")

    ### preparing the initial state

    l=0
    for k in [2,1,0]:
        if initial_state[l]=='1':
            qc.x(qr[k])
        l+=1

    for _ in range(N_steps):
        qc.append(qc_gate,qr)

    return qi.Statevector.from_instruction(qc)

def CS_gate():
    qr=QuantumRegister(2)
    CS_qc=QuantumCircuit(qr,name='C-S')

    CS_qc.cx(qr[1],qr[0])
    CS_qc.tdg(qr[0])#(-pi/4,qr[0])
    CS_qc.cx(qr[1],qr[0])
    CS_qc.t(qr[0])#CS_qc.rz(pi*2/3,qr[0])
    CS_qc.t(qr[1])#CS_qc.rz(pi*2/3,qr[1])

    return CS_qc

def simmetry_check(type="simmetry1"):
    #################################################### making the simmerty check
    ##### qubit 0 gets the mesurament
    if type=="copy_check":
        qr_ch=QuantumRegister(6)
        qc_ch=QuantumCircuit(qr_ch,name ='copy_check')

        qc_ch.cx(qr_ch[3],qr_ch[0])
        qc_ch.cx(qr_ch[5],qr_ch[2])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])
        qc_ch.cx(qr_ch[3],qr_ch[1])
        qc_ch.cx(qr_ch[4],qr_ch[3])

        return qc_ch

    if type=="simmetry1":
        s_qr = QuantumRegister(4)
        s_qc = QuantumCircuit(s_qr,name="simmetry_check_1")
        s_qc.cx(1,0)
        s_qc.swap(1,2)
        s_qc.cx(1,0)
        s_qc.swap(3,2)
        s_qc.swap(1,2)
        s_qc.cx(1,0)

        return s_qc
    
    if type=="simmetry2":
        s_qr = QuantumRegister(5)
        s_qc = QuantumCircuit(s_qr,name="simmetry_check_2")
        s_qc.cx(2,0)
        s_qc.cx(2,1)
        s_qc.swap(2,3)
        s_qc.cx(2,0)
        s_qc.cx(2,1)
        s_qc.swap(3,4)
        s_qc.swap(2,3)
        s_qc.cx(2,0)
        s_qc.cx(2,1)

        return s_qc

    if type=="magnetization1":

        CS_qc=CS_gate()

        qr=QuantumRegister(5)
        CheckM_qc=QuantumCircuit(qr,name='CheckM_1')

        CheckM_qc.h(qr[1])
        CheckM_qc.cx(qr[2],qr[0])
        CheckM_qc.append(CS_qc,[qr[1],qr[2]])
        CheckM_qc.swap(qr[2],qr[3])
        CheckM_qc.cx(qr[2],qr[0])
        CheckM_qc.append(CS_qc,[qr[1],qr[2]])
        CheckM_qc.swap(qr[4],qr[3])
        CheckM_qc.swap(qr[2],qr[3])
        CheckM_qc.cx(qr[2],qr[0])
        CheckM_qc.append(CS_qc,[qr[1],qr[2]])
        CheckM_qc.h(qr[1])

        return CheckM_qc

    if type=="magnetization2":

        CS_qc=CS_gate()

        qr=QuantumRegister(7)
        DoubleCheckM_qc=QuantumCircuit(qr,name='CheckM')

        DoubleCheckM_qc.h(qr[1])
        DoubleCheckM_qc.h(qr[5])
        DoubleCheckM_qc.cx(qr[4],qr[6])
        DoubleCheckM_qc.cx(qr[2],qr[0])
        DoubleCheckM_qc.append(CS_qc,[qr[1],qr[2]])
        DoubleCheckM_qc.append(CS_qc,[qr[5],qr[4]])
        DoubleCheckM_qc.swap(qr[2],qr[3])
        DoubleCheckM_qc.cx(qr[2],qr[0])
        DoubleCheckM_qc.append(CS_qc,[qr[1],qr[2]])
        DoubleCheckM_qc.swap(qr[4],qr[3])
        DoubleCheckM_qc.cx(qr[4],qr[6])
        DoubleCheckM_qc.append(CS_qc,[qr[5],qr[4]])
        DoubleCheckM_qc.swap(qr[2],qr[3])
        DoubleCheckM_qc.cx(qr[2],qr[0])
        DoubleCheckM_qc.append(CS_qc,[qr[1],qr[2]])
        DoubleCheckM_qc.swap(qr[4],qr[3])
        DoubleCheckM_qc.cx(qr[4],qr[6])
        DoubleCheckM_qc.append(CS_qc,[qr[5],qr[4]])
        DoubleCheckM_qc.h(qr[1])
        DoubleCheckM_qc.h(qr[5])

        return DoubleCheckM_qc

def U_approx_tomo(steps=10,trot_type="our",checks=[],initial_state='011',time=np.pi):

    ############################################### making the trotter step
    Trot_qc=Trotter_step(type=trot_type, N=steps, time=time)

    ###################################################################
    N_bit=3
    sim_check=[]
    check_type=[]
    qubit_check=[]
    reset=[]

    for i in range(len(checks)):
        N_bit+=len(checks[i][2])
        sim_check.append(checks[i][0])
        check_type.append(checks[i][1])
        qubit_check.append(checks[i][2])
        reset.append(checks[i][3])

    cra_len=0
    for g in qubit_check:
        cra_len+=len(g)

    qr=QuantumRegister(7, name="qr")
    cr=ClassicalRegister(3, name="cr")
    
    ising=QuantumCircuit(qr,cr,name="U")
    if len(qubit_check)>0:
        cr_a = ClassicalRegister(cra_len)
        ising.add_register(cr_a)

    #############################building the initial state
    if initial_state[0]=='1':
        ising.x(qr[1])
    if initial_state[1]=='1':
        ising.x(qr[3])
    if initial_state[2]=='1':
        ising.x(qr[5])

    h=0
    m=0
    order="standard"  ###order will take the qubit order, that can swich after the checks.

    if trot_type in ["our", "IBM"]:
        for i in range(steps+1):   ##### Buliding of the cirquit
        # ising.barrier()
            if h<len(sim_check):
                while sim_check[h]==i:
             
                    ising, m, order = add_check(ising, qr, cr_a, check_type[h], qubit_check[h], reset[h], m, order, i, steps)

                    h+=1
                    if h>=len(sim_check):
                        break
            
            if steps>0 and trot_type == 'our':
                if i == 0:
                    ising.rz(-pi/2,qr[3])
                if i == steps:
                    ising.rz(pi/2,qr[3]) 
                

            if i<steps:
                if order=="standard":
                    ising.append(Trot_qc,[qr[1],qr[3],qr[5]])
                else:
                    if order=="converted":
                        ising.append(Trot_qc,[qr[5],qr[3],qr[1]])
                    else:
                        print("errore!")
  
    if trot_type == "decomposed": 
        if len(sim_check) > 0:
            while sim_check[h] == 0:
                ising, m, order = add_check(ising, qr, cr_a, check_type[h], qubit_check[h], reset[h], m, order, 0, steps)
                h+=1
                if h>=len(sim_check):
                    h=h-1
                    break

        if order=="standard":
            ising.append(Trot_qc,[qr[1],qr[3],qr[5]])
        else:
            if order=="converted":
                ising.append(Trot_qc,[qr[5],qr[3],qr[1]])
            else:
                print("errore!")
        if len(sim_check) > 0:
            while sim_check[h] == steps:
                ising, m, order = add_check(ising, qr, cr_a, check_type[h], qubit_check[h], reset[h], m, order, steps, steps)
                h+=1
                if h>=len(sim_check):
                    break

    ################################# making the tomography circuits
    ising.barrier()

    if order=="standard":
        st_qcs = cirquits_tomography(ising, [qr[1],qr[3],qr[5]],[cr[0],cr[1],cr[2]])
    else:
        if order=="strange":
            st_qcs = cirquits_tomography(ising, [qr[3],qr[1],qr[5]],[cr[0],cr[1],cr[2]])
        else:
            st_qcs = cirquits_tomography(ising, [qr[5],qr[3],qr[1]],[cr[0],cr[1],cr[2]])

    if len(sim_check)>0:
        if check_type[0]=="copy_check":
            for ccc in st_qcs:
                ccc.barrier()
                ccc.measure([qr[0],qr[2],qr[4]],cr[3:6])
    #################################### making the tomography with and without the ancillas    
    
    if len(sim_check)>0:
        st_qcs_na = U_approx_tomo(steps=steps,trot_type=trot_type,checks=[],initial_state=initial_state)
        return st_qcs, st_qcs_na
    else:
        return st_qcs

def mitigate(raw_results, Measure_Mitig="yes", ancillas_condition='', backend_calibration="", qubit_ancilla=[], meas_fitter=0):

    N_ancillas=len(ancillas_condition)
    N_qubit=N_ancillas+3
    new_result = deepcopy(raw_results)
    new_result_nm = deepcopy(raw_results)

    if Measure_Mitig=="yes" and meas_fitter==0:
        qubit=qubit_ancilla
        qubit.append(1)
        qubit.append(3)
        qubit.append(5)
        meas_fitter=calibration_measure_mitigation(qubit=qubit,backend_calibration=backend_calibration, shots=32000)

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
                if reg_bits[0]==ancillas_condition:
                    new_counts_nm[reg_bits[1]]=old_counts[reg_key]
        else:
            for reg_key in old_counts:
                new_counts_nm[reg_key]=old_counts[reg_key]

        new_result_nm.results[i].data.counts = new_counts_nm

        if Measure_Mitig=="yes":
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
                    if reg_bits[0]==ancillas_condition:
                        new_counts[reg_bits[1]]=old_counts[reg_key]
            else:
                for reg_key in old_counts:
                    new_counts[reg_key]=old_counts[reg_key]
        
            new_result.results[i].data.counts = new_counts
    if Measure_Mitig=="yes":
        return new_result, new_result_nm
    else:
        return new_result_nm

def add_check(ising, qr, cr_a, check_type, qubit_check, reset, m, order, i, steps):
    if check_type=="simmetry1":
        s_qc=simmetry_check(type="simmetry1")

        if qubit_check[0]==0 or qubit_check[0]==2:
            ising.append(s_qc,[qr[qubit_check[0]],qr[1],qr[3],qr[5]])
        else:
            ising.append(s_qc,[qr[qubit_check[0]],qr[5],qr[3],qr[1]])

        ##############aggiungo la misura sulle ancille seguite dal reset

        ising.measure(qr[qubit_check[0]],cr_a[m])
        for _ in range(reset[0]):
            ising.reset(qr[qubit_check[0]])
        #ising.barrier()
        m+=1

        ################ tengo conto dello shifting tra i qubit

        if order=="standard":
            order="converted"
        else:
            order="standard"

    if (check_type=="simmetry2" or check_type=="magnetization1"):
        s_qc=simmetry_check(type=check_type)

        if (qubit_check[0]==0 and qubit_check[1]==2) or (qubit_check[0]==2 and qubit_check[1]==0):
            ising.append(s_qc,[qr[qubit_check[0]],qr[qubit_check[1]],qr[1],qr[3],qr[5]])
        else:
            if (qubit_check[0]==4 and qubit_check[1]==6) or (qubit_check[0]==6 and qubit_check[1]==4):
                ising.append(s_qc,[qr[qubit_check[0]],qr[qubit_check[1]],qr[5],qr[3],qr[1]])
            else:
                print("Check-qubits choise error!! Be careful on the processor geometry")
                return error

        ##############aggiungo la misura sulle ancille seguite dal reset

        ising.measure([qr[qubit_check[0]],qr[qubit_check[1]]],[cr_a[m],cr_a[m+1]])
        for _ in range(reset[0]):
            ising.reset(qr[qubit_check[0]])
        for _ in range(reset[1]):
            ising.reset(qr[qubit_check[1]])
        #ising.barrier()
        m+=2

        ################ tengo conto dello shifting tra i qubit

        if order=="standard":
            order="converted"
        else:
            order="standard"

    if check_type=="magnetization2":
        s_qc=simmetry_check(type=check_type)
        
        ising.append(s_qc,[qr[qubit_check[0]],qr[qubit_check[1]],qr[1],qr[3],qr[5],qr[qubit_check[2]],qr[qubit_check[3]]])

        ################ tengo conto dello shifting tra i qubit

        if i==steps:
            order="strange"
        else:
            ising.swap(qr[3],qr[5])

            if order=="standard":
                order="converted"
            else:
                order="standard"

        ##############aggiungo la misura sulle ancille seguite dal reset

        ising.measure([qr[qubit_check[0]],qr[qubit_check[1]],qr[qubit_check[2]],qr[qubit_check[3]]],[cr_a[m],cr_a[m+1],cr_a[m+2],cr_a[m+3]])
        for jj in range(4):
            for _ in range(reset[jj]):
                ising.reset(qr[qubit_check[jj]])
        #ising.barrier()
        m+=4
    
    if check_type=="copy_check":
        s_qc=simmetry_check(type=check_type)
        
        ising.append(s_qc,[qr[qubit_check[0]],qr[qubit_check[1]],qr[qubit_check[2]],qr[1],qr[3],qr[5]])

    return ising, m, order

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

def cirquits_tomography(qc, qr, cr):
    N_qubit=len(qr)

    pc=possible_circuits(N_qubit)

    qcs=[]
    for i in range(3**N_qubit):
        qcs.append(qc.copy())
        x=[]

        for j in range(N_qubit):
            if pc[i][j]=='I':
                x.append('Z')
                qcs[i].measure(qr[j],cr[j])
            else:
                if pc[i][j]=='H':
                    x.append('X')
                    qcs[i].h(qr[j])
                    qcs[i].measure(qr[j],cr[j])
                else:
                    if pc[i][j]=='S':
                        x.append('Y')
                        qcs[i].sdg(qr[j])
                        qcs[i].h(qr[j])
                        qcs[i].measure(qr[j],cr[j])
        qcs[i].name=str((x[0],x[1],x[2]))


    return qcs

def possible_circuits(N):
    r=[]
    for i in range(3**N):
        r.append(DecimalToIHS(i,N))
    return r

def DecimalToIHS(num, N_bit):

    cir=['H','S','I']
    b=''

    if num==0:
        b='H'
    while(num>0):
        x=num%3
        b=cir[x]+b
        num=num//3
    while len(b)<N_bit:
        b='H'+b
    return b

def fpos(i):
    r=0
    for s in [0,1,2]:
        if i==r:
            return s
        r+=1
    print("errore")
    return   