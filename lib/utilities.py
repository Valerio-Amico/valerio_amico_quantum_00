import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

def LIC_calibration_circuit(quantum_circuit):
    '''
    returns the calibration circuit of a 'quantum_circuit'
    '''
    splitted_qasm = quantum_circuit.qasm().split(";\n")
    qr_LIC = QuantumRegister(len(quantum_circuit.qubits), name="q")
    qc_LIC = QuantumCircuit(qr_LIC, name="LIC")
    for element in splitted_qasm:
        if "cx" in element:
            el = element.replace("q[", "").replace("]", "")
            control_target = el.split(" ")[1].split(",")
            add_random_coupling_cnot(qc_LIC, qr_LIC[int(control_target[0])], qr_LIC[int(control_target[1])])
    return qc_LIC

def add_random_coupling_cnot(qc, control, target):
    '''
    appends to a circuit 'qc' a 2-qubit gate, which has |00> as eigenstate,
    builted in the following way:
    two single qubit gates are applied to the qubits 'control' and 'target', than is applied a c-not
    and after are applied two other single qubit gates.

    explain better what it does!!!!!!!!!!!!!!!!!!!!!!!!!

    '''
    choices = [
        "IIII", "YIYX", "XIXX", "ZIZI", 
        "IXIX", "YXYI", "XXXI", "ZXZX", 
        "IYZY", "YYXZ", "XYYZ", "ZYIY", 
        "IZZZ", "YZXY", "XZYY", "ZZIZ"
    ]
    pqrs = np.random.choice(choices)
    if pqrs[0]=='X':
        qc.x(control)
    elif pqrs[0]=='Y':
        qc.y(control)
    elif pqrs[0]=='Z':
        qc.z(control)
    
    if pqrs[1]=='X':
        qc.x(target)
    elif pqrs[1]=='Y':
        qc.y(target)
    elif pqrs[1]=='Z':
        qc.z(target)
    
    qc.cx(control, target)

    if pqrs[2]=='X':
        qc.x(control)
    elif pqrs[2]=='Y':
        qc.y(control)
    elif pqrs[2]=='Z':
        qc.z(control)
    
    if pqrs[3]=='X':
        qc.x(target)
    elif pqrs[3]=='Y':
        qc.y(target)
    elif pqrs[3]=='Z':
        qc.z(target)

    return
    
def DecimalToBinary(num, number_of_qubits):
    """Converts a decimal to a binary string of length ``number_of_qubits``."""
    return f"{num:b}".zfill(number_of_qubits)

def bin_list(N_qubit):
    r=[]
    for i in range(2**N_qubit):
        r.append(DecimalToBinary(i,N_qubit))
    return r