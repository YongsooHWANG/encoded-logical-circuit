from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info.operators import Operator

q =  QuantumRegister(2,"qreg")
qc = QuantumCircuit(q)

customUnitary = Operator([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])
a = qc.unitary(customUnitary, [q[0], q[1]], label='custom')
print(a)
for b in a:
    print(b)
print(qc.draw(output='mpl'))