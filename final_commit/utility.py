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

def gates_parameters(initial_state, U):

    parity = Parity(initial_state)
    column = BinaryToDecimal(initial_state)

    if parity == 2:
        A0 = U[3*8+column]
        A1 = U[5*8+column]
        A2 = U[6*8+column]
    else:
        A0 = U[3*8+column]
        A1 = U[5*8+column]
        A2 = U[6*8+column]

    r1=float(angolo(U[3*8+6])+angolo(U[6*8+6]))/2
    r2=0
    f1=float(angolo(U[6*8+6])-angolo(U[5*8+6])-np.pi)/2
    f2=float((angolo(U[6*8+6])-angolo(U[3*8+6]))/2-f1)
    a1=float(acos(abs(U[6*8+6])))
    a2=float(acos(abs(U[5*8+6])/sin(a1)))

    return r1, r2, f1, f2, a1, a2

def jobs_result(job_evolution, reps=1):

    backend_sim = Aer.get_backend('qasm_simulator')

    qr=QuantumRegister(7)
    qc=QuantumCircuit(qr)
    qcs = state_tomography_circuits(qc, [qr[1],qr[3],qr[5]])
    for qc in qcs:
        cr = ClassicalRegister(4)
        qc.add_register(cr)
        qc.measure([qr[0],qr[2],qr[4],qr[6]],cr)

    jobs_evo_res = []
    for i in range(reps):
    
        job=execute(qcs,backend=backend_sim, shots=10)
        results = job.result()
        
        for j in range(27):
            results.results[j].data.counts = job_evolution.result().get_counts()[i*27+j]
    
        jobs_evo_res.append(results)

    return jobs_evo_res

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

def angolo(x): ## QUESTA FUNZIONE NON SERVE A NIENTE!!!!! usare atan2 di sympy.
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

def BinaryToDecimal(bin):
    d=0
    for i in range(len(bin)):
        if bin[-1-i]=='1':
            d+=2**i
    return d

def parity(bin):
    p=0
    for i in range(len(bin)):
        if bin[i]=='1':
            p+=1
    return p


