import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize

dev = qml.device('default.qubit', wires=2)

# Define hamiltonian
H = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

@qml.qnode(dev)
def VQE_X(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)

    qml.U2(0, np.pi, wires=0) # H gate for right basis
    qml.U2(0, np.pi, wires=1) # H gate for right basis

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@qml.qnode(dev)
def VQE_Y(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)

    qml.U2(0, np.pi/2, wires=0) # Y gate for right basis
    qml.U2(0, np.pi/2, wires=1) # Y gate for right basis

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@qml.qnode(dev)
def VQE_Z(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)

    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def VQE_I(params):
    return 1


def E(params):
    return 0.5*(VQE_Z(params) + VQE_I(params) - VQE_X(params) - VQE_Y(params))


# Minimizing cost function (energy)
params = list(np.pi * (np.random.random(size=2) - 0.5))
tol = 1e-4
opt_vqe = minimize(E, params, method='Powell', tol=tol)

print(f'Reference minimum energy (eigenvalue) of H: {np.linalg.eigvals(H).min()}')
print(f'Minimum energy (eigenvalue) of H: {np.round(opt_vqe.fun, 5)}')
print(f'Optimized params phi1: {np.round(opt_vqe.x[0], 5)}, phi2 {np.round(opt_vqe.x[1], 5)}')