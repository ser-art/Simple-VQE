from qiskit import *
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

import numpy as np
from scipy.optimize import minimize


def get_hamiltonian_operator(i_coef, z_coef, x_coef, y_coef):
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": i_coef}, "label": "II"},
                   {"coeff": {"imag": 0.0, "real": z_coef}, "label": "ZZ"},
                   {"coeff": {"imag": 0.0, "real": x_coef}, "label": "XX"},
                   {"coeff": {"imag": 0.0, "real": y_coef}, "label": "YY"}
                   ]
    }

    return WeightedPauliOperator.from_dict(pauli_dict)

# Get H from coef, which were found by hands
H = get_hamiltonian_operator(i_coef=0.5, 
                             z_coef=0.5, 
                             x_coef=-0.5, 
                             y_coef=-0.5)


def quantum_state_preparation(circuit, parameters):
    q = circuit.qregs[0] 
    circuit.h(q[0])
    circuit.cx(q[0], q[1])
    circuit.rx(parameters[0], q[0])
    circuit.ry(parameters[1], q[0])

    return circuit


def vqe_circuit(parameters, measure):
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    circuit = quantum_state_preparation(circuit, parameters)

    if measure == 'ZZ':
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])
    elif measure == 'XX':
        circuit.u2(0, np.pi, q[0]) # H gate for right basis
        circuit.u2(0, np.pi, q[1]) # H gate for right basis
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])
    elif measure == 'YY':
        circuit.u2(0, np.pi/2, q[0]) # Y gate for right basis
        circuit.u2(0, np.pi/2, q[1]) # Y gate for right basis
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])

    return circuit


def expectation(parameters, measure):
    circuit = None
    if measure == 'II':
        return 1
    elif measure == 'ZZ':
        circuit = vqe_circuit(parameters, 'ZZ')
    elif measure == 'XX':
        circuit = vqe_circuit(parameters, 'XX')
    elif measure == 'YY':
        circuit = vqe_circuit(parameters, 'YY')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    expectation_value = 0
    for measure_result in counts:
        sign = +1
        if measure_result == '01' or measure_result == '10':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
         
    return expectation_value


def pauli_operator_to_dict(pauli_operator):
    d = pauli_operator.to_dict()
    paulis = d['paulis']
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff

    return paulis_dict

pauli_dict = pauli_operator_to_dict(H)


def energy(parameters):
    energy = 0
    for pauli_name in pauli_dict.keys():
        energy += pauli_dict[pauli_name] * expectation(parameters, pauli_name)
    return energy

params = list(np.pi * (np.random.random(size=2) - 0.5))
tol = 1e-4
opt_vqe = minimize(energy, params, method='Powell', tol=tol)

exact_result = NumPyEigensolver(H).run()
reference_energy = min(np.real(exact_result.eigenvalues))

print(f'Reference minimum energy (eigenvalue) of H: {reference_energy}')
print(f'Minimum energy (eigenvalue) of H: {np.round(opt_vqe.fun, 5)}')
print(f'Optimized params phi1: {np.round(opt_vqe.x[0], 5)}, phi2 {np.round(opt_vqe.x[1], 5)}')