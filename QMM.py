from qiskit import *
from qiskit.circuit.library import *
from qiskit.circuit import *
from qiskit.quantum_info import *
from qiskit.circuit.library import UnitaryGate
import numpy as np
import math
from qiskit.quantum_info import *


#################################################################
#       Quantum Subroutine for Matrix Multiplication (QMM)      #
#################################################################
def bin_d(j, d):
    assert(j>=0 and j<2**d)
    s = bin(j)[2:]
    return '0'*(d-len(s))+s

def tree(x):
    n = len(x)
    d = int(np.log2(n))
    Btree = np.zeros((d+1, n))
    s = [-1 if k<0 else 1 for k in x]
    Btree[d, :] = x[:]**2
    for i in range(d-1, -1, -1):
        for j in range(2**i):
            Btree[i, j]= Btree[i+1, 2*j]+ Btree[i+1, 2*j+1]
    return Btree, s, Btree[0, 0]

def encode(x):
    dim = len(x)
    d = int(math.ceil(np.log2(dim)))
    qr_q = QuantumRegister(d)
    qc = QuantumCircuit(qr_q, name = 'Encode')
    Bintree, s, nx = tree(x)
    Bintree = Bintree/nx
    rc= 0
    theta= 2*np.arccos(np.sqrt(Bintree[1, 0]/Bintree[0, 0]))
    if d==1:
        Rcustom = np.array([[s[0]*np.cos(theta/2), -s[0]*np.sin(theta/2)],[s[1]*np.sin(theta/2),s[1]*np.cos(theta/2) ]])
        gate_custom = UnitaryGate(Rcustom)#.control(1, ctrl_state = bin_d(0, 1))
        qc.append(gate_custom, [qr_q[0]])
    else:
        qc.append(RYGate(theta), [qr_q[d-1]])
    
    for i in range(2, d+1):
        for j in range(2**(i-1)):
            if i == d:
                theta = 2*np.arccos(np.sqrt(Bintree[i, 2*j]/Bintree[i-1, j]))
                Rcustom = np.array([[s[2*j]*np.cos(theta/2), -s[2*j]*np.sin(theta/2)],[s[2*j+1]*np.sin(theta/2),s[2*j+1]*np.cos(theta/2) ]])
                #print(np.linalg.norm(Rcustom@Rcustom.T-np.eye(2)))
                gate_custom = UnitaryGate(Rcustom).control(i-1, ctrl_state = bin_d(j, i-1))
                qc.append(gate_custom, qr_q[d-i+1:d+1]+[qr_q[d-i]]) 
            else:
                theta= 2*np.arccos(np.sqrt(Bintree[i, 2*j]/Bintree[i-1, j]))
                qc.append(RYGate(theta).control(i-1, ctrl_state = bin_d(j, i-1)), qr_q[d-i+1:d+1]+[qr_q[d-i]]) 
            
    return qc.to_gate()

def Ai(A, row):
    x = A[row, :]
    dim = len(x)
    d = int(math.ceil(np.log2(dim)))
    qr_q = QuantumRegister(d)
    qc = QuantumCircuit(qr_q, name = 'A'+str(row))
    Bintree, s, nx = tree(x)
    Bintree = Bintree/nx
    rc= 0
    theta= 2*np.arccos(np.sqrt(Bintree[1, 0]/Bintree[0, 0]))
    if d==1:
        Rcustom = np.array([[s[0]*np.cos(theta/2), -s[0]*np.sin(theta/2)],[s[1]*np.sin(theta/2),s[1]*np.cos(theta/2) ]])
        gate_custom = UnitaryGate(Rcustom)#.control(1, ctrl_state = bin_d(0, 1))
        qc.append(gate_custom, [qr_q[0]])
    else:
        qc.append(RYGate(theta), [qr_q[d-1]])
   
    for i in range(2, d+1):
        for j in range(2**(i-1)):
            if i == d:
                theta = 2*np.arccos(np.sqrt(Bintree[i, 2*j]/Bintree[i-1, j]))
                Rcustom = np.array([[s[2*j]*np.cos(theta/2), -s[2*j]*np.sin(theta/2)],[s[2*j+1]*np.sin(theta/2),s[2*j+1]*np.cos(theta/2) ]])
                #print(np.linalg.norm(Rcustom@Rcustom.T-np.eye(2)))
                gate_custom = UnitaryGate(Rcustom).control(i-1, ctrl_state = bin_d(j, i-1))
                qc.append(gate_custom, qr_q[d-i+1:d+1]+[qr_q[d-i]]) 
            else:
                theta= 2*np.arccos(np.sqrt(Bintree[i, 2*j]/Bintree[i-1, j]))
                qc.append(RYGate(theta).control(i-1, ctrl_state = bin_d(j, i-1)), qr_q[d-i+1:d+1]+[qr_q[d-i]]) 
            
    return qc.to_gate()

def Atutta(A):
    mm, nn = A.shape
    ml = int(np.log2(mm))
    nl = int(np.log2(nn))
    qr = QuantumRegister(nl)
    qr_control = QuantumRegister(ml)
    qc = QuantumCircuit(qr, qr_control, name = str(A))
    qc.draw()
    #for i in range(m):
    #    qc.h(qr_control[i])
    for i in range(mm):
        qc.append(Ai(A, i).control(ml, ctrl_state = bin_d(i, ml)), qr_control[:]+qr[:])
    return qc.to_gate()

def QMM(A, B):
    M, K = A.shape
    K, N = B.shape 
    m = int(math.ceil(np.log2(M)))
    k = int(math.ceil(np.log2(K)))
    n = int(math.ceil(np.log2(N)))
    qr = QuantumRegister(k, 'reg k')
    qr_c1 = QuantumRegister(m, 'reg m')
    qr_c2 = QuantumRegister(n, 'reg n')
    qc = QuantumCircuit(qr, qr_c1, qr_c2)
    
    # si recuperano le norme di B.T e di A.T
    nb = np.zeros(N)
    na = np.zeros(M)
    for i in range(N):
        _, _, nb[i] = tree(B[:, i])
    for i in range(M):
        _, _, na[i] = tree(A[i, :])
    qc.append(encode(np.sqrt(nb)), qr_c2[:])
    qc.append(encode(np.sqrt(na)), qr_c1[:])
    #qc.barrier()
    #for i in range(k):
    #    qc.h(qr_c2[i])
    #for i in range(m):
    #    qc.h(qr_c1[i])
    qc.append(Atutta(B.T), qr[:]+qr_c2[:]) # qc.append(Atutta(B.T), qr[:]+qr_c2[:])
    #qc.append(V(n, m).inverse(), qr_c1[:]+qr_c2[:])
    #qc.append(V(n, m), qr_c1[:]+qr_c2[:])

    qc.append(Atutta(A).inverse(), qr[:]+qr_c1[:]) #qc.append(Atutta(A).inverse(), qr[:]+qr_c1[:])
    
    return qc 
