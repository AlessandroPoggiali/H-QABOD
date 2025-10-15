# Hybrid Quantum Angle Based Outlier Detection (H-QABOD) Algorithm

Hybrid Quantum Angle Based Outlier Detection (H-QABOD) algorithm  is a hybrid quantum-classical method for solving the Outlier Detection problem. It is inspired by the classical Angle Based Outlier Detection (ABOD)[[1]](#1) algorithm, which, given a dataset record $p$, computes the angles between each other pair of dataset records $a$ and $b$ ($\hat{apb}$). Then, ABOD computes the variance of these angles. If the variance is low, $p$ is highly likely an outlier. H-QABOD mimics the behavior of ABOD by computing the variance of the angles through the QVAR[[2]](#2) and QMM[[3]](#3) subroutines for variance computation and matrix-matrix multiplication, respectively. 

H-QABOD uses [QVAR](https://github.com/AlessandroPoggiali/QVAR) to compute the variance of as measure to detect outliers.

## Quickstart

To run a simple demostration of the H-QABOD algorithm, follow these steps:
* Make sure you have Qiskit installed on your computer (v1.4)
* Install the QVAR package `pip install QVAR` 
* Clone this repo with `git clone https://github.com/AlessandroPoggiali/H-QABOD.git`
* Navigate to the H-QABOD directory and run the command `python3 H-QABOD.py`

The `test.py` file contains code that will run H-QADBO on a very simple dataset.

## References
<a id="1">[1]</a> 
Kriegel, Hans-Peter, Matthias Schubert, and Arthur Zimek. "Angle-based outlier detection in high-dimensional data." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 2008.

<a id="2">[2]</a> 
Bernasconi, A., Berti, A., Del Corso, G. M., Guidotti, R., & Poggiali, A. (2024). Quantum subroutine for variance estimation: algorithmic design and applications. Quantum Machine Intelligence, 6(2), 78.

<a id="3">[3]</a> 
Bernasconi, A., Berti, A., Del Corso, G. M., & Poggiali, A. (2024). Quantum subroutine for efficient matrix multiplication. IEEE Access.
