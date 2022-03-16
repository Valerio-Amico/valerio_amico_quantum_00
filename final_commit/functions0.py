import numpy as np
from sympy import *
from sympy.physics.quantum import TensorProduct as Tp
from qiskit import Aer, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, transpile, execute
 
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

def trotter_step(a):
    
    m = Matrix([
        [exp(-I*a),0,0,0],
        [0,cos(a),-I*sin(a),0],
        [0,-I*sin(a),cos(a),0],
        [0,0,0,exp(-I*a)]
    ])

    return Tp(m, eye(2)) * Tp(eye(2), m) * exp(I*a)

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
        U=U.evalf(40)
    
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

    return qc


    