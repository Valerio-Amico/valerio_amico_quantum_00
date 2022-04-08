import numpy as np
from .utility import *
from copy import deepcopy
import sympy
from sympy import Matrix
from sympy.physics.quantum import TensorProduct as Tp
from qiskit import (
    Aer,
    assemble,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    IBMQ,
    transpile,
    execute,
)
from qiskit.ignis.verification.tomography import (
    state_tomography_circuits,
    StateTomographyFitter,
)
from qiskit.quantum_info import state_fidelity


def trotter_step_matrix(parameter):

    """
    Here is computed the matrix of a single trotter step. It can be done numerically or symbolcally.

    Args:

        - parameter: can be a sympy Symbol or a double.

    Returns:

        - trotter_step_matrix: single trotter steps matrix (symbolic or numeric) with parameter=time/N_steps.

    """

    X = Matrix([[0, 1], [1, 0]])  # defining the pauli matrices
    Y = Matrix([[0, -sympy.I], [sympy.I, 0]])
    Z = Matrix([[1, 0], [0, -1]])
    Id = eye(2)

    H1 = Tp(X, X, Id) + Tp(Y, Y, Id) + Tp(Z, Z, Id)
    H2 = Tp(Id, X, X) + Tp(Id, Y, Y) + Tp(Id, Z, Z)

    trotter_step_matrix_ = exp(-parameter * H1 * sympy.I) * exp(
        -parameter * H2 * sympy.I
    )

    return trotter_step_matrix_


def symmetry_check(type="copy_check"):

    """
    this function appends a "copy check" with 3 or four qubits to the circuit given.
    see the file "Ancillas_Error_mitigation_Git_Hub.ipynb" to understand better how does them work.

    Args:

        type (string): type of check, options:
                                - "copy_check": copy check with 3 ancillas
                                - "4copy_check": copy check with 3 ancillas

    Returns:

        qc (QuantumCircuit): the quantum circuit of the check.

    """

    if type == "copy_check":
        qr_ch = QuantumRegister(6, name="q")
        qc_ch = QuantumCircuit(qr_ch, name="copy_check")

        qc_ch.cx(qr_ch[3], qr_ch[0])
        qc_ch.cx(qr_ch[5], qr_ch[2])
        qc_ch.cx(qr_ch[3], qr_ch[1])
        qc_ch.cx(qr_ch[4], qr_ch[3])
        qc_ch.cx(qr_ch[3], qr_ch[1])
        qc_ch.cx(qr_ch[4], qr_ch[3])

    if type == "4copy_check":

        qr_ch = QuantumRegister(7, name="q")
        qc_ch = QuantumCircuit(qr_ch, name="4copy_check")

        qc_ch.cx(qr_ch[4], qr_ch[0])
        qc_ch.cx(qr_ch[6], qr_ch[2])
        qc_ch.cx(qr_ch[4], qr_ch[1])
        qc_ch.cx(qr_ch[5], qr_ch[4])
        qc_ch.cx(qr_ch[4], qr_ch[1])
        qc_ch.cx(qr_ch[5], qr_ch[4])
        qc_ch.cx(qr_ch[6], qr_ch[3])

    return qc_ch


def build_circuit(circuit_list, number_of_qubits, name=None):
    """Builds a circuit described by a list of tuples of gates and qubits index

    The circuit_list must be given in format:

    [["gate_1", [qubits_args] ], ["gate_2", [qubits_args]] ]

    where qubits_args is
        [int, int]      for control gates
        [int]           for single qubit gates
        [float, int]    for single qubit gates with parameter
    """
    register = QuantumRegister(number_of_qubits)
    circuit = QuantumCircuit(register, name=name)
    for element in circuit_list:
        gate_name, qubits = element
        qubits = [
            register[qubit_index] if isinstance(qubit_index, int) else qubit_index
            for qubit_index in qubits
        ]
        gate = getattr(circuit, gate_name)
        gate(*qubits)
    return circuit


def add_symmetry_check(qc, qr_control_qubits, qr_ancillas, type="copy_check"):

    """
    this function appends a "copy check" with 3 or four qubits to the circuit given.
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

    """

    if type == "copy_check":
        qr_ch = QuantumRegister(6)
        qc_ch = QuantumCircuit(qr_ch, name="copy_check")

        qc_ch.cx(qr_ch[3], qr_ch[0])
        qc_ch.cx(qr_ch[5], qr_ch[2])
        qc_ch.cx(qr_ch[3], qr_ch[1])
        qc_ch.cx(qr_ch[4], qr_ch[3])
        qc_ch.cx(qr_ch[3], qr_ch[1])
        qc_ch.cx(qr_ch[4], qr_ch[3])

        qc.append(qc_ch, qr_ancillas + qr_control_qubits)

    if type == "4copy_check":

        qr_ch = QuantumRegister(7)
        qc_ch = QuantumCircuit(qr_ch, name="4copy_check")

        qc_ch.cx(qr_ch[4], qr_ch[0])
        qc_ch.cx(qr_ch[6], qr_ch[2])
        qc_ch.cx(qr_ch[4], qr_ch[1])
        qc_ch.cx(qr_ch[5], qr_ch[4])
        qc_ch.cx(qr_ch[4], qr_ch[1])
        qc_ch.cx(qr_ch[5], qr_ch[4])
        qc_ch.cx(qr_ch[6], qr_ch[3])

        qc.append(qc_ch, qr_ancillas + qr_control_qubits)

    return qc


# funzioni quasi verificate


def evolution_circuit_single_state(
    time=np.pi, n_steps=42, initial_state={"110": 1}, precision=40
):

    """
    This function computes numerically the operator obtained with the composition of "n_steps" trotter steps,
    and than builds the evolution circuit with the best decomposition (4 c-not),
    with a evolution time equals to "time" (see "decomposition.ipynb").

    Args:

        - n_steps (integer): is the number of trotter steps.
        - time (double): is the total evolution time.
        - initial_state (string): the 3-qubit initial state, from right to left, the characters are associated with qubits 1, 3 and 5 respectively.
        - precision (integer): is the digit where every numerical operation will be troncated.

    Returns:

        - qc (Quantumcircuit): evolution circuit.

    """

    numeric_evolution_matrix = eye(8)

    for _ in range(
        n_steps
    ):  # here is computed the evolution operator numerically, with n_steps trotter steps.
        numeric_evolution_matrix = (
            numeric_evolution_matrix * trotter_step_matrix(2 * time / n_steps)
        ).evalf(precision)

    # here are computed the parameters of the gates as described in "decomposition.ipynb" file.
    phase_1_1, phase_2_1, phase_2_1, phase_2_2, a1, a2 = get_gates_parameters(
        initial_state=initial_state, U=numeric_evolution_matrix
    )

    # defining the two qubits gates that preserve the magnetization when applyed, with the parameters just computed.
    M1_qc = fixed_magnetization_two_qubit_gate(phase_1_1, phase_2_1, a1)
    M2_qc = fixed_magnetization_two_qubit_gate(phase_2_1, phase_2_2, a2)

    # defining and building the quantum circuit.
    qr = QuantumRegister(7, name="q")
    qc = QuantumCircuit(qr, name="U")

    # initializing the state as chosen in "initial_state".
    initial_state = "110"

    l = 0
    for k in [5, 3, 1]:
        if initial_state[l] == "1":
            qc.x(qr[k])
        l += 1

    qc.append(M1_qc, [qr[1], qr[3]])
    qc.append(M2_qc, [qr[3], qr[5]])

    return qr, qc


def get_calibration_circuit(type="", n_steps=0, time=np.pi):
    """Generates the calibration circuit for the given calibration procedure."""

    # THis part literally bulds the circuit again, it' unnecessary
    """if type == "itself":
        initial_state = "110"
        precision = 45

        print("cambiare initial_state e time")

        numeric_evolution_matrix = eye(8)

        for _ in range(
            n_steps
        ):  # here is computed the evolution operator numerically, with n_steps trotter steps.
            numeric_evolution_matrix = (
                numeric_evolution_matrix * trotter_step_matrix(time / n_steps)
            ).evalf(precision)

        # here are computed the parameters of the gates as described in "decomposition.ipynb" file.
        phase_1_1, phase_2_1, phase_1_2, phase_2_2, a1, a2 = get_gates_parameters(
            initial_state=initial_state, U=numeric_evolution_matrix
        )

        M1_qc = fixed_magnetization_two_qubit_gate(phase_1_1, phase_1_2, a1)
        M2_qc = fixed_magnetization_two_qubit_gate(phase_2_1, phase_2_2, a2)

        qr3 = QuantumRegister(3, name="q")
        qc3 = QuantumCircuit(qr3, name="U")

        qc3.append(M1_qc, [qr3[0], qr3[1]])
        qc3.append(M2_qc, [qr3[1], qr3[2]])

        qc3 = transpile(qc3, basis_gates=["cx", "rz", "x", "sx"])

        return qc3"""

    if type == "complete_evolution":
        # circuito calibrazione per 14-cnot
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])

        qc.cx(qr[0], qr[1])

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])

        qc.cx(qr[0], qr[1])

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.barrier()
        qc.x(qr[1])

        qc.cx(qr[0], qr[1])

        qc.x(qr[0])
        qc.y(qr[0])
        qc.x(qr[0])
        qc.y(qr[0])

        qc.x(qr[1])
        qc.y(qr[1])
        qc.x(qr[1])
        qc.y(qr[1])

        qc.cx(qr[0], qr[1])

        qc.barrier()

        qc.x(qr[2])
        qc.y(qr[2])
        qc.x(qr[2])
        qc.y(qr[2])

        qc.cx(qr[1], qr[2])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[0], qr[1])
        qc.barrier()
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])

        return qc

    if type == "column_evolution_remake":
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr, name="IdCirc")

        qc.x([qr[0], qr[1]])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.cx(qr[0], qr[1])
        qc.x([qr[0], qr[1]])
        qc.sx([qr[0], qr[1]])
        qc.barrier()
        qc.sx([qr[0], qr[1]])

        qc.cx(qr[0], qr[1])
        qc.x([qr[0], qr[1]])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.barrier()
        qc.x([qr[1], qr[2]])
        qc.sx(qr[1])
        qc.barrier()
        qc.sx(qr[1])
        qc.x(qr[2])
        qc.cx(qr[1], qr[2])
        qc.x([qr[1], qr[2]])
        qc.sx([qr[1], qr[2]])
        qc.barrier()
        qc.sx([qr[1], qr[2]])

        qc.cx(qr[1], qr[2])
        qc.x([qr[1], qr[2]])
        qc.sx(qr[1])
        qc.barrier()
        qc.sx(qr[1])
        qc.x(qr[2])

        return qc

    if type == "column_evolution":
        # circuito calibrazione per 4-cnot
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.x(qr[0])
        qc.y(qr[0])
        qc.sx(qr[0])
        qc.barrier()
        qc.sx(qr[0])
        qc.x(qr[1])
        qc.y([qr[0], qr[1]])
        qc.barrier()
        qc.x(qr[1])
        qc.y(qr[1])
        qc.cx(qr[0], qr[1])
        qc.x(qr[0])
        qc.y(qr[0])
        qc.barrier()
        qc.x(qr[0])
        qc.y(qr[0])
        qc.cx(qr[0], qr[1])
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
        qc.cx(qr[1], qr[2])
        qc.x(qr[1])
        qc.y(qr[1])
        qc.barrier()
        qc.x(qr[1])
        qc.y(qr[1])
        qc.cx(qr[1], qr[2])
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

    # if type=="trotter_steps":

    #    return circ_cal_tot(N)

    return "error"


def build_calibration_circuits_for_all_base_elements(type="", 
                        q_anc=[], 
                        N=0, 
                        time=np.pi, 
                        check="no", 
                        check_type="copy_check"):

    c_qc = get_calibration_circuit(type=type, n_steps=N, time=time)

    qubits = [1, 3, 5] + q_anc
    N_qubits = len(qubits)
    pos_init = bin_list(N_qubits)

    qcs = []
    meas = []

    for i in range(2 ** N_qubits):

        cr = ClassicalRegister(N_qubits)
        qr = QuantumRegister(N_qubits)
        qc = QuantumCircuit(
            qr, cr, name="%scal_%s" % ("", DecimalToBinary(i, N_qubits))
        )

        cr_1 = ClassicalRegister(N_qubits)
        qr_1 = QuantumRegister(N_qubits)
        qc_1 = QuantumCircuit(
            qr_1, cr_1, name="%scal_%s" % ("", DecimalToBinary(i, N_qubits))
        )

        l = 0
        qubits.reverse()
        for k in qubits:
            if pos_init[i][l] == "1":
                qc.x(qr[k])
                qc_1.x(qr_1[k])
            l += 1
        qubits.reverse()

        qc.append(c_qc, [qr[1], qr[3], qr[5]])

        if check == "yes":

            # qc.barrier()
            l = 0
            qubits.reverse()
            for k in qubits:
                if pos_init[i][l] == "1":
                    qc.x(qr[k])
                    # qc_1.x(qr_1[k])
                l += 1
            qubits.reverse()

            if check_type == "copy_check":
                qc = add_symmetry_check(
                    qc, [qr[1], qr[3], qr[5]], [qr[0], qr[2], qr[4]], type=check_type
                )
            if check_type == "4copy_check":
                qc = add_symmetry_check(
                    qc,
                    [qr[1], qr[3], qr[5]],
                    [qr[0], qr[2], qr[4], qr[6]],
                    type=check_type,
                )

            l = 0
            qubits.reverse()
            for k in qubits:
                if pos_init[i][l] == "1":
                    qc.x(qr[k])
                    # qc_1.x(qr_1[k])
                l += 1
            qubits.reverse()

        qc.barrier()
        qc_1.barrier()

        i = 0
        for k in qubits:
            qc.measure(qr[k], cr[i])
            qc_1.measure(qr_1[k], cr_1[i])
            i += 1

        qcs.append(qc)
        meas.append(qc_1)

    return qcs, meas


def mitigate(raw_results, Measure_Mitig="yes", ancillas_conditions=[], meas_fitter=0):

    """
    This function computes the mitigated results of a job.

    Args:

        - raw_results (job result): the results of the job without any mitigation, just runned.
        - Measure_Mitig (string): two options:
                                - "yes": it will do the measure mitigation with the calibration matrix passed "meas_fitter".
                                - "no": will not do the measure mitigation.
        - ancillas_conditions (list of string): contains strings of "0" and "1", that correspond to the states allowed for the ancillas.
                                                for example: if the initial state belong to the subspace with magnetization = 2, wi know
                                                that the state will be in the same subspace for all the evolution. So "ancillas_condition"
                                                should be equal to ["011","110","101"] if we use 3 ancillas.
        - meas_fitter (CompleteMeasFitter): The calibration matrix, for the measurament mitigation.

    """

    N_ancillas = len(ancillas_conditions[0])  # number of ancillas qubits
    N_qubit = N_ancillas + 3  # total number of qubits
    new_result = deepcopy(raw_results)
    new_result_nm = deepcopy(raw_results)

    ################create the list of the total possible outcomes
    r = bin_list(N_qubit=N_qubit)
    r_split = []
    for j in range(2 ** (N_qubit)):
        X_aus = ""
        X_aus += r[j][:N_ancillas]
        X_aus += " "
        X_aus += r[j][N_ancillas:]
        r_split.append(X_aus)

    for i in range(len(raw_results.results)):

        old_counts = raw_results.get_counts(i)
        new_counts = {}
        new_counts_nm = {}

        new_result.results[i].header.creg_sizes = [
            new_result.results[i].header.creg_sizes[0]
        ]
        new_result.results[i].header.clbit_labels = new_result.results[
            i
        ].header.clbit_labels[0:-1]
        new_result.results[i].header.memory_slots = 3

        new_result_nm.results[i].header.creg_sizes = [
            new_result_nm.results[i].header.creg_sizes[0]
        ]
        new_result_nm.results[i].header.clbit_labels = new_result_nm.results[
            i
        ].header.clbit_labels[0:-1]
        new_result_nm.results[i].header.memory_slots = 3

        if N_ancillas > 0:
            for reg_key in old_counts:
                reg_bits = reg_key.split(" ")
                if reg_bits[0] in ancillas_conditions:
                    if reg_bits[1] not in new_counts_nm:
                        new_counts_nm[reg_bits[1]] = 0
                    new_counts_nm[reg_bits[1]] += old_counts[reg_key]
        else:
            for reg_key in old_counts:
                new_counts_nm[reg_key] = old_counts[reg_key]

        new_result_nm.results[i].data.counts = new_counts_nm

        if Measure_Mitig == "yes" and meas_fitter != 0:
            if N_ancillas > 0:
                for j in range(2 ** N_qubit):
                    if r_split[j] in old_counts.keys():
                        old_counts[r[j]] = old_counts.pop(r_split[j])

            old_counts = meas_fitter.filter.apply(
                old_counts, method="least_squares"
            )  # 'least_squares' or 'pseudo_inverse'

            if N_ancillas > 0:
                for j in range(2 ** N_qubit):
                    if r[j] in old_counts.keys():
                        old_counts[r_split[j]] = old_counts.pop(r[j])

            if N_ancillas > 0:
                for reg_key in old_counts:
                    reg_bits = reg_key.split(" ")
                    if reg_bits[0] in ancillas_conditions:
                        if reg_bits[1] not in new_counts:
                            new_counts[reg_bits[1]] = 0
                        new_counts[reg_bits[1]] += old_counts[reg_key]
            else:
                for reg_key in old_counts:
                    new_counts[reg_key] = old_counts[reg_key]

            new_result.results[i].data.counts = new_counts

    if Measure_Mitig == "yes" and meas_fitter != 0:
        return new_result, new_result_nm
    else:
        return new_result_nm


def fidelity_count(result, qcs, target_state):
    tomo_ising = StateTomographyFitter(result, qcs)
    rho_fit_ising = tomo_ising.fit(method="lstsq")
    # fid=state_fidelity(rho_fit_ising, target_state)
    fid = state_fidelity(target_state, rho_fit_ising)
    print("attenzione ho invertito gli arg di state_fidelity")
    return fid


def circuits_without_ancillas_measuraments(job):
    qcs_without_ancillas = []

    for cir in job.circuits():
        circuit = cir.copy()
        circuit.remove_final_measurements()
        c = ClassicalRegister(3, name="c")
        circuit.add_register(c)
        circuit.barrier()
        circuit.measure([1, 3, 5], c)

        qcs_without_ancillas.append(circuit)

    return qcs_without_ancillas


# funzioni da verificare
def matrix_from_circuit(qc, phase=0, type="sympy"):

    backend = Aer.get_backend("unitary_simulator")
    job = execute(qc, backend, shots=32000)
    result = job.result()
    A = result.get_unitary(qc, decimals=10) * np.exp(1j * phase)
    if type == "sympy":
        return Matrix(A)
    else:
        return A


def mitigate2(raw_results, ancillas_conditions=[], meas_fitter=None):

    if meas_fitter is None:
        # Mitigazione senza fitter
        pass

    if "__tens_fitt" in dir(meas_fitter):
        # Mitigazione con fitter singolo
        pass

    if "__iter__" in dir(meas_fitter):
        # Mitigazione con lista di filtri
        pass

    N_ancillas = len(ancillas_conditions[0])

    N_qubit = N_ancillas + 3
    new_result = deepcopy(raw_results)
    new_result_nm = deepcopy(raw_results)

    ################create the list of the total possible outcomes
    r = bin_list(N_qubit=N_qubit)
    r_split = []
    for j in range(2 ** (N_qubit)):
        X_aus = ""
        X_aus += r[j][:N_ancillas]
        X_aus += " "
        X_aus += r[j][N_ancillas:]
        r_split.append(X_aus)

    for i in range(len(raw_results.results)):

        old_counts = raw_results.get_counts(i)
        new_counts = {}
        new_counts_nm = {}

        # cambia il numero di classical bit dei reults

        new_result.results[i].header.creg_sizes = [
            new_result.results[i].header.creg_sizes[0]
        ]
        new_result.results[i].header.clbit_labels = new_result.results[
            i
        ].header.clbit_labels[0:-1]
        new_result.results[i].header.memory_slots = 3

        new_result_nm.results[i].header.creg_sizes = [
            new_result_nm.results[i].header.creg_sizes[0]
        ]
        new_result_nm.results[i].header.clbit_labels = new_result_nm.results[
            i
        ].header.clbit_labels[0:-1]
        new_result_nm.results[i].header.memory_slots = 3

        # if Measure_Mitig=="yes" and meas_fitter != 0:
        if N_ancillas > 0:
            for reg_key in old_counts:
                reg_bits = reg_key.split(" ")
                if reg_bits[0] in ancillas_conditions:
                    if reg_bits[1] not in new_counts_nm:
                        new_counts_nm[reg_bits[1]] = 0
                    new_counts_nm[reg_bits[1]] += old_counts[reg_key]
        else:
            for reg_key in old_counts:
                new_counts_nm[reg_key] = old_counts[reg_key]

        new_result_nm.results[i].data.counts = new_counts_nm

        if meas_fitter is not None:
            if N_ancillas > 0:
                for j in range(2 ** N_qubit):
                    if r_split[j] in old_counts.keys():
                        old_counts[r[j]] = old_counts.pop(r_split[j])

            print(i)

            if "_tens_fitt" in dir(meas_fitter):
                # print(old_counts)
                print("tens_fitt")
                old_counts = meas_fitter.filter.apply(
                    old_counts, method="least_squares"
                )

            if "__iter__" in dir(meas_fitter):
                print(i, np.shape(meas_fitter[i].filter._cal_matrix))
                meas_fitter_aus = deepcopy(meas_fitter[i])
                print("deepcopy")

                old_counts = meas_fitter_aus.filter.apply(
                    old_counts, method="least_squares"
                )

            if N_ancillas > 0:
                for j in range(2 ** N_qubit):
                    if r[j] in old_counts.keys():
                        old_counts[r_split[j]] = old_counts.pop(r[j])

            if N_ancillas > 0:
                for reg_key in old_counts:
                    reg_bits = reg_key.split(" ")
                    if reg_bits[0] in ancillas_conditions:
                        if reg_bits[1] not in new_counts:
                            new_counts[reg_bits[1]] = 0
                        new_counts[reg_bits[1]] += old_counts[reg_key]
            else:
                for reg_key in old_counts:
                    new_counts[reg_key] = old_counts[reg_key]

            new_result.results[i].data.counts = new_counts

    if meas_fitter is not None:
        return new_result, new_result_nm
    else:
        return new_result_nm


def trotter_step_matrix2(parameter):

    """
    Here is computed the matrix of a single trotter step. It can be done numerically or symbolcally.

    Args:

        - parameter: can be a sympy Symbol or a double.

    Returns:

        - trotter_step_matrix: single trotter steps matrix (symbolic or numeric) with parameter=2*time/N_steps.

    """

    m = Matrix(
        [
            [exp(-sympy.I * parameter), 0, 0, 0],
            [
                0,
                exp(sympy.I * parameter) * cos(2 * parameter),
                -sympy.I * sin(2 * parameter) * exp(sympy.I * parameter),
                0,
            ],
            [
                0,
                -sympy.I * sin(2 * parameter) * exp(sympy.I * parameter),
                cos(2 * parameter) * exp(sympy.I * parameter),
                0,
            ],
            [0, 0, 0, exp(-sympy.I * parameter)],
        ]
    )

    trotter_step_matrix = Tp(m, eye(2)) * Tp(eye(2), m)

    return trotter_step_matrix


def Trotter_N_approx(steps=10, tempo=np.pi, precision=10):

    ### Funzione che calcola numericamente la matrice di evoluzione,
    ### con un numero di trotter steps pari a steps e un tempo di evoluzione pari a tempo,
    ### la precision è la cifra meno signivicativa dove si troncano i risultati di tutte le operazioni.

    q = Symbol("q", positive=True)  # q=2*t/N, t=tempo, N=steps
    # e = Symbol("e", positive = True)

    Id = eye(2)

    cx_01 = Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    m = Matrix(
        [
            [exp(-sympy.I * q), 0, 0, 0],
            [0, exp(-sympy.I * q), 0, 0],
            [
                0,
                0,
                cos(2 * q) * exp(sympy.I * q),
                -sympy.I * sin(2 * q) * exp(sympy.I * q),
            ],
            [
                0,
                0,
                -sympy.I * sin(2 * q) * exp(sympy.I * q),
                cos(2 * q) * exp(sympy.I * q),
            ],
        ]
    )

    Trotter_step = Tp(Id, cx_01 * m * cx_01) * Tp(cx_01 * m * cx_01, Id)
    U = Tp(Id, Tp(Id, Id))

    for _ in range(steps):
        # print(steps,i)
        U = U * Trotter_step
        U = U.subs(q, tempo / steps)
        U = U.evalf(precision)

    return U


def solve_equation(initial_state, U):

    a1, r1, f1, a2, r2, f2_ = symbols("a1 r1 f1 a2 r2 f2", real=True)

    H = Matrix([[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]])

    cx_01 = Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    def ry(alpha):  # generic ry gate
        return Matrix(
            [[cos(alpha / 2), -sin(alpha / 2)], [sin(alpha / 2), cos(alpha / 2)]]
        )

    def rz(alpha):  # generic rz gate
        return Matrix(
            [[exp(-1 * sympy.I * (alpha / 2)), 0], [0, exp(sympy.I * (alpha / 2))]]
        )

    Id = eye(2)

    M1 = Matrix(
        simplify(
            Tp(rz(2 * f1), Id)
            * Tp(Id, H)
            * cx_01
            * Tp(ry(a1), ry(a1))
            * cx_01
            * Tp(Id, H)
            * Tp(rz(2 * r1), Id)
        )
    )
    M2 = Matrix(
        simplify(
            Tp(rz(2 * f2_), Id)
            * Tp(Id, H)
            * cx_01
            * Tp(ry(a2), ry(a2))
            * cx_01
            * Tp(Id, H)
            * Tp(rz(2 * r2), Id)
        )
    )

    Ubest = Tp(M2, Id) * Tp(Id, M1)

    lin_equations = []
    non_lin_eqs = []

    column = BinaryToDecimal(initial_state)
    magnetization = Magnetization(initial_state)

    if magnetization == 2:
        rows = [3, 5, 6]
    if magnetization == 1:
        rows = [1, 2, 4]

    for i in rows:
        ar, ab = arg_and_abs(Ubest[i * 8 + column])
        non_lin_eqs.append(ab - Abs(U[i * 8 + column]))
        lin_equations.append(ar - atan2(im(U[i * 8 + column]), re(U[i * 8 + column])))

    linear_solution = linsolve(lin_equations, [f1, f2_, r1, r2])
    linear_solution = linear_solution.subs(r2, 0)

    eqs = [non_lin_eqs[0], non_lin_eqs[2]]
    non_lin_solution = solve(eqs, [a1, a2])

    parameters = {}
    j = 0
    for i in linear_solution.args[0]:
        parameters["phase_%d_%d" % (j // 2, j % 2)] = float(i)
        j += 1

    j = 0
    for i in non_lin_solution[0]:
        parameters["abs_%d" % j] = i
        j += 1

    return parameters


def arg_and_abs(expression):
    absu = 1
    argu = 0
    for b in expression.args:
        if b == -1:
            argu += -pi
        else:
            if im(b) != 0:
                argu += -sympy.I * b.args[0]
            else:
                absu *= b
    return argu, absu
