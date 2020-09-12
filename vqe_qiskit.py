from qiskit import *
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver

import numpy as np
from scipy.optimize import minimize

# define hamiltonian
H = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

def hamiltonian_operator(i_coef, z_coef, x_coef, y_coef):
    """
    Creates i_coef*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    
    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": i_coef}, "label": "II"},
                   {"coeff": {"imag": 0.0, "real": z_coef}, "label": "ZZ"},
                   {"coeff": {"imag": 0.0, "real": x_coef}, "label": "XX"},
                   {"coeff": {"imag": 0.0, "real": y_coef}, "label": "YY"}
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)



# params = list(np.pi * (np.random.random(size=2) - 0.5))
# tol = 1e-3
# opt_vqe = minimize(vqe, params, method="Powell", tol=tol)

# print(f'True minimum energy (eigenvalue) of H: {np.linalg.eigvals(H).min()}')
# print(f'Minimum energy (eigenvalue) of H: {np.round(opt_vqe.fun, 5)}')