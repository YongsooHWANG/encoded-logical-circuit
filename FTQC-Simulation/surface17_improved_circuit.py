from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

qc = QuantumCircuit(17, 17)

qc.h(9)
qc.h(11)
qc.h(14)
qc.h(16)

qc.barrier()

qc.cx(2, 12)
qc.cx(0, 10)
qc.cx(4, 13)

qc.cx(11,1)
qc.cx(14,5)
qc.cx(16,7)

qc.barrier()

qc.cx(5,12)
qc.cx(7,13)
qc.cx(3,10)

qc.cx(11,0)
qc.cx(14,4)
qc.cx(16,6)

qc.barrier()

qc.cx(1,12)
qc.cx(5,15)
qc.cx(3,13)

qc.cx(9,2)
qc.cx(11,4)
qc.cx(14,8)

qc.barrier()

qc.cx(4,12)
qc.cx(8,15)
qc.cx(6,13)

qc.cx(9,1)
qc.cx(14,7)
qc.cx(11,3)

qc.barrier()

qc.h(9)
qc.h(11)
qc.h(14)
qc.h(16)

qc.barrier()

qc.measure(range(9,17), range(8))

# circuit_drawer(qc, output="text", style={"backgroundcolor":"#EEEEEE"})
# circuit_drawer(qc, output="text")
print(qc)