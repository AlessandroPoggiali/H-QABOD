from qiskit import *
from qiskit.circuit.library import *
from qiskit.circuit import *
from qiskit.quantum_info import *
from qiskit.circuit.library import UnitaryGate
from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import AmplitudeEstimation, FasterAmplitudeEstimation
from qiskit.providers.fake_provider import *
from qiskit.primitives import Sampler, StatevectorSampler
import numpy as np
import math
import csv
from qiskit.quantum_info import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
from pyod.utils.data import generate_data
from metrics import * 
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_gaussian_quantiles
import os
#import pyterrier_alpha as pta
from pyod.models.abod import ABOD
from QMM import QMM


###########################################################
#                         QVAR                            #
###########################################################

# QVAR subroutine for computing the variance of a set of values encoded into a quantum state
#
#
# U: state preparation
# var_index: list of qubit indices of which we want to compute the variance. If more than
#            var_index qubits are present in U, the first var_index are the target qubits.
# ps_index: list of U's qubits that require a post selection measurement  
# version: method for estimating the variance
#
#    'FAE'    (default) Faster Amplitude Estimation
#    'AE'     Amplitude Estimation
#    'SHOTS'  measurements with multiple circuit execution
#
# delta (optional)          : target accuracy (FAE)
# max_iter (optional)       : maximum number of iterations of the oracle (FAE)
# eval_qbits (optional)     : number of additional qubits (AE)
# shots (optional)          : number of shots (SHOTS)
# n_h_gates (optional)      : normalization constant to multiply the final value
# postprocessing (optional) : if True, return the MLE postprocessed value (only for AE)

def QVAR(U, var_index=None, ps_index=None, version='FAE', delta=0.0001, max_iter=5, eval_qbits=5, shots=8192, n_h_gates=0, postprocessing=True, backend=None):

    if var_index is None:
        var_index = [x for x in range(U.num_qubits)]
    
    i_qbits = len(var_index)
    e_qbits = i_qbits
    u_qbits = U.num_qubits

    a = QuantumRegister(1,'a')
    e = QuantumRegister(e_qbits,'e')
    u = QuantumRegister(u_qbits, 'u')

    if version == 'SHOTS':
        ca = ClassicalRegister(1,'ca')
        ce = ClassicalRegister(e_qbits,'ce')
        if ps_index is not None:
            cps = ClassicalRegister(len(ps_index), 'cps')
            qc = QuantumCircuit(a, e, u, ca, ce, cps)
        else:
            qc = QuantumCircuit(a, e, u, ca, ce)
    
    else:
        qc = QuantumCircuit(a, e, u)
    
    #qc.append(U.to_gate(), list(range(1+e_qbits, qc.num_qubits)))       
    st_ff = Statevector.from_instruction(U)
    qc.append(StatePreparation(st_ff.data), list(range(1+e_qbits, qc.num_qubits)))
    qc.h(a)

    
    for t in range(i_qbits):
        qc.cswap(a,e[t],u[var_index[t]])

    qc.ch(a,e)
    for t in range(i_qbits):
        qc.ch(a,u[var_index[t]])
        
    qc.x(e)    
    qc.h(a)
    
    if ps_index is None:
        objective_qubits = [x for x in range(1+e_qbits)]
    else:
        objective_qubits = [x for x in range(1+e_qbits)]+[qc.num_qubits-u_qbits + x for x in ps_index]
    

    qc.x([qc.num_qubits-u_qbits + x for x in ps_index]) ### X SULLE K

    #print(qc.draw())
    
    if version == 'SHOTS':
        qc.measure(a, ca) 
        qc.measure(e, ce)
        
        if ps_index is not None:
            qc.measure(u[ps_index], cps)
            target_conf = '1'*len(ps_index) + ' ' + '1'*e_qbits + ' 1' 
        else:
            target_conf = '1'*e_qbits + ' 1'
            
        counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()

        try: 
            var = (counts[target_conf])/shots
        except:
            var = 0
            
    elif version == 'AE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        ae = AmplitudeEstimation(
            num_eval_qubits=eval_qbits,  
            sampler=sampler
        )
        
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        ae_result = ae.estimate(problem)
    
        if postprocessing:
            var = ae_result.mle
        else:
            var = ae_result.estimation
        
    elif version == 'FAE':
        sampler = Sampler()
        sampler.set_options(backend=backend)
        fae = FasterAmplitudeEstimation(
            delta=delta, 
            maxiter=max_iter,  
            sampler=sampler
        )

        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )
        fae_result = fae.estimate(problem)
        var = fae_result.estimation

    elif version == 'STATEVECTOR':
        problem = EstimationProblem(
            state_preparation=qc, 
            objective_qubits=objective_qubits,
        )

        transpiled_circuit = transpile(qc, backend, optimization_level=0, coupling_map=backend.coupling_map, seed_transpiler=123)
        transpiled_circuit.save_statevector()
        statevector = np.asarray(backend.run(transpiled_circuit,seed_simulator=123).result().get_statevector())
        #print(statevector.real)
        
        var = 0
        for i, amplitude in enumerate(statevector):
            full_state = bin(i)[2:].zfill(qc.num_qubits)[::-1]
            state = ''.join([full_state[i] for i in objective_qubits])
            if problem.is_good_state(state[::-1]):
                var = var + np.abs(amplitude) ** 2
        
    tot_hadamard = 2 + n_h_gates
    norm_factor = 2**tot_hadamard/2**i_qbits

    return var*norm_factor
        
    

###########################################################
#                         H-QABOD                         #
###########################################################

def classical_abod_all(X):
    variances = []
    for pivot_idx, pivot in enumerate(X):
        new_X = X - pivot  
        new_X = np.delete(new_X, pivot_idx, axis=0)  
        new_X = normalize(new_X)
        #new_X = new_X / (np.linalg.norm(new_X, axis=1, ord=2) ** 2)[:, np.newaxis]  # Stesso tipo di normalizzazione di angle_list
        angle_list = []
        #print("hqabod shape",new_X.shape)
        #print("hqabod normalized",np.linalg.norm(new_X[0]))
        prod = (new_X@new_X.T)
        #print("prod\n"+str(prod))
        variances.append(np.var(prod))
    return variances

def classical_abod_triu(X):
    variances = []
    for pivot_idx, pivot in enumerate(X):
        new_X = X - pivot   
        new_X = np.delete(new_X, pivot_idx, axis=0) 
        new_X = normalize(new_X)
        #new_X = new_X / (np.linalg.norm(new_X, axis=1, ord=2) ** 2)[:, np.newaxis]  # Stesso tipo di normalizzazione di angle_list
        #print("abod shape",new_X.shape)
        prod = (new_X@new_X.T)
        triu = prod[np.triu_indices(len(new_X), k = 1)]
        #print("triu\n"+str(triu))  
        variances.append(np.var(triu))
    return variances

from numpy.linalg import norm

def compute_angle(x, y):
    x = np.array(x)
    y = np.array(y)
    return (x @ y) / (norm(x) * norm(y))  

def classical_abod_no_normalization(X):
    variances = []
    for pivot_idx, pivot in enumerate(X):
        new_X = X - pivot
        new_pivot = pivot - pivot 
        new_X = np.array([x for x in new_X if (x != new_pivot).all()])

        angle_list = []
        for i in range(len(new_X)):
            for j in range(len(new_X)):
                if j>i:
                    angle_list.append(compute_angle(new_X[i], new_X[j])) 
        #print("triu\n"+str(triu))  
        variances.append(np.var(angle_list))
    return variances


def classical_abod(X, contamination):
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    X = normalize(X)
    abod = ABOD(contamination=contamination, method='default')
    abod.fit(X)
    return abod.decision_scores_*-1

def inv_stereo(X):
    n = len(X[0])
    m = len(X)
    new_X = []
    for j in range(m):
        s = sum(X[j]**2)
        for i in range(n):
            X[j][i] = 2*X[j][i]
        new_X.append([x for x in X[j]])
        new_X[j] = np.append(X[j], np.array(s-1))
        new_X[j] = new_X[j]/(s+1)
    return new_X


def compute_difference(x, y):
    return [a-b for a,b in zip(x,y)]

def classical_abod_differences(X):
    variances = []
    for pivot_idx, pivot in enumerate(X):
        new_X = X - pivot  
        new_X = np.delete(new_X, pivot_idx, axis=0) 
        new_X = inv_stereo(new_X) # CON ISP
        #print("qoda shape",np.array(new_X).shape)
        #print("qoda normalized",np.linalg.norm(new_X[0]))
        #new_X = normalize(new_X)
        differences = []
        #print("Pivot " + str(pivot_idx))
        for i in range(len(new_X)):
            for j in range(len(new_X)):
                #if j>i:
                differences.append(compute_difference(new_X[i], new_X[j]))
        var = np.var(np.ravel(differences))
        variances.append(var)
    return variances

def load_datasets(to_load, label=False):
    datasets = []

    for ds in to_load:
        
        if ds == 'creditcard.csv':
            ds_name = "Creditcard"
            data = pd.read_csv('outlierdetection_datasets/creditcard/creditcard.csv')
            data = data.iloc[:, 1:]
            _, x_test, _, y_test = train_test_split(data.iloc[:, :-1], data["Class"],
                                                        test_size=0.01,
                                                        random_state=42,
                                                        stratify=data["Class"])
            data = x_test
            data = data.values
        else:
            ds_name = ds.split('_')[0]
            dataset = arff.loadarff('outlierdetection_datasets/' + ds_name + '/' + ds)
            if label:
                df = pd.DataFrame(dataset[0])
                label = df.iloc[:, -1]
                label = pd.Series(np.array([1 if x==b'yes' else 0 for x in label]))
                data = df.iloc[:, :-2]
                data = data.merge(label.rename('label'),left_index=True, right_index=True)
            else:
                data = pd.DataFrame(dataset[0]).iloc[:, :-2].values
        datasets.append((data, ds_name))

    
    return datasets

def compute_p(N, x_percent):
    """ Calcola p per considerare il primo x% della lista """
    n_x = N * (x_percent / 100)
    return np.exp(np.log(0.01) / n_x)

def classic_vs_quantum_real():

    datasets = load_datasets([
        'Glass_withoutdupl_norm.arff',
        'Ionosphere_withoutdupl_norm.arff',
        'Shuttle_withoutdupl_norm_v01.arff',
        'WDBC_withoutdupl_norm_v01.arff',
        'WPBC_withoutdupl_norm.arff'
        ])
    
    p_list = [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    #p_list = [0.5946, 0.8677,0.9310, 0.9609, 0.9763, 0.9856]
    n_list_perc = [1, 10, 20, 30, 40, 50]
    

    with open('rbo_all.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+['p='+str(p) for p in p_list]) 
    with open('p@n_all.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+[str(n)+'%' for n in n_list_perc]) 
    with open('kendalltau_all.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'KendallTau']) 

    with open('rbo_diff.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+['p='+str(p) for p in p_list]) 
    with open('p@n_diff.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+[str(n)+'%' for n in n_list_perc]) 
    with open('kendalltau_diff.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'KendallTau']) 

    for dataset, dataset_name in datasets:
        print("TEST DATASET " + dataset_name + ' - ' + str(dataset.shape))
        rbo_list_all = []
        precision_at_n_list_all = []
        rbo_list_diff = []
        precision_at_n_list_diff = []

        variances_all = classical_abod_all(dataset)
        variances_triu = classical_abod_triu(dataset)
        variances_differences = classical_abod_differences(dataset)
        
        rank_all = list(np.argsort(variances_all))
        rank_triu = list(np.argsort(variances_triu))
        rank_differences = list(np.argsort(variances_differences))

        for p in p_list:
            rbo_list_all.append(compute_rbo(rank_all, rank_triu, p=p))
            rbo_list_diff.append(compute_rbo(rank_differences, rank_triu, p=p))
            #rbo_list(pta.rbo(pd.DataFrame(variances_all, columns=["va"]), pd.DataFrame(variances_triu, columns=["vu"]), p=p))
        
        n_list = [int(n_list_perc[i]/100*dataset.shape[0]) for i in range(len(n_list_perc))]
        for n in n_list:
            precision_at_n_list_all.append(precision_at_n(variances_all, variances_triu, n))
            precision_at_n_list_diff.append(precision_at_n(variances_differences, variances_triu, n))
        
        k_all = kendalltau(rank_all, rank_triu)   
        k_diff = kendalltau(rank_differences, rank_triu)   

        with open('rbo_all.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+rbo_list_all) 
        with open('p@n_all.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+precision_at_n_list_all) 
        with open('kendalltau_all.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+[k_all])  

        with open('rbo_diff.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+rbo_list_diff) 
        with open('p@n_diff.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+precision_at_n_list_diff) 
        with open('kendalltau_diff.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+[k_diff])  

def test_old_heuristic():

    datasets = load_datasets([
        'Glass_withoutdupl_norm.arff',
        'Ionosphere_withoutdupl_norm.arff',
        'Shuttle_withoutdupl_norm_v01.arff',
        #'Waveform_withoutdupl_norm_v01.arff', 
        'WDBC_withoutdupl_norm_v01.arff',
        'WPBC_withoutdupl_norm.arff'
        ])
    
    p_list = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    n_list_perc = [1, 10, 20, 30, 40, 50]
    

    with open('rbo_old2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+['p='+str(p) for p in p_list]) 
    with open('p@n_old2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset']+[str(n)+'%' for n in n_list_perc]) 
    with open('kendalltau_old2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'KendallTau']) 

    for dataset, dataset_name in datasets:
        print("TEST DATASET " + dataset_name + ' - ' + str(dataset.shape))
        rbo_list = []
        precision_at_n_list = []

        variances_all = classical_abod(dataset)
        variances_triu = classical_abod_differences(dataset)

        for p in p_list:
            rbo_list.append(compute_rbo(variances_all, variances_triu, p=p))
        
        n_list = [int(n_list_perc[i]/100*dataset.shape[0]) for i in range(len(n_list_perc))]
        for n in n_list:
            precision_at_n_list.append(precision_at_n(variances_all, variances_triu, n))
        
        k = kendalltau(variances_all, variances_triu)    

        with open('rbo_old2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+rbo_list) 
        with open('p@n_old2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+precision_at_n_list) 
        
        with open('kendalltau_old2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name]+[k])  
        

def heuristic_evaluation_synth(data_distribution='uniform'):
    trials = 100
    
    #var_triu2 = []
    M_list = [50,100,150,200,250,300,350,400,450,500]
    #N = 5
    N_list = [10,20,30,40,50]
    

    for N in N_list:
        MAE_list = []
        MRE_list = []
        discrepancy_errors = []
        for M in M_list:
            print("N=" + str(N) + " M=" + str(M))
            var_tutta = []
            var_triu = []
            for t in range(trials):
                np.random.seed(M*10+t)
                if data_distribution == 'uniform':
                    x = np.random.randn(M,N)
                elif data_distribution == 'normal':
                    x = np.random.normal(0, 1, size=(M,N))      
                    #x, _ = generate_data(offset=1,n_train=M,n_features=N,train_only=True,contamination=0,random_state=t) 
                else:
                    print("Unvalid data distribution")
                    exit()

                x = normalize(x)
                #x = x / (np.linalg.norm(x, axis=1, ord=2) ** 2)[:, np.newaxis]
                A = (x@x.T) 
                var_tutta.append(np.var(A))
                triu = A[np.triu_indices(len(A), k = 1)]
                var_triu.append(np.var(triu))
            
            MAE_list.append(np.mean([abs(x-y) for x,y in zip(var_tutta, var_triu)]))
            MRE_list.append(np.mean([abs(x-y)/x for x,y in zip(var_tutta, var_triu)]))
            discrepancy_errors.append(compute_discrepancy(trials, var_tutta, var_triu))

        rows = zip(M_list, MAE_list, MRE_list, discrepancy_errors)

        # Write to a CSV file
        with open(data_distribution + '_resultsL2_N=' + str(N)+ '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['M', 'MAE', 'MRE', 'Discrepancy'])  # Write header
            writer.writerows(rows)  # Write rows

def QABOD(X):
    variances = []
    for pivot_idx, pivot in enumerate(X):
        new_X = np.array(X) - pivot   
        new_X = np.delete(new_X, pivot_idx, axis=0) 
        Y = new_X.copy()
        angle_list = []
        print(str(pivot_idx+1)+ "\\" + str(len(X)))

        angle_list = []
        
        k = int(np.log2(new_X.shape[1]))
        #new_X = new_X / (np.linalg.norm(new_X, axis=1, ord=2) ** 2)[:, np.newaxis]
        qc = QMM(new_X, new_X.T)

        backend = GenericBackendV2(qc.num_qubits)
        transpiled_circuit = transpile(qc, backend, optimization_level=0, coupling_map=backend.coupling_map, seed_transpiler=123)
        transpiled_circuit.save_statevector()
        statevector = np.asarray(backend.run(transpiled_circuit,seed_simulator=123).result().get_statevector())
        qvar_list = np.zeros(len(new_X)**2)
        for i in range(0,len(new_X)**2):
            qvar_list[i] = statevector[i*len(new_X[0])]
        qvar_list = qvar_list*np.linalg.norm(new_X, 'fro')**2
        qvar_list = list(qvar_list)
        q_var = QVAR(qc, ps_index=list(range(k)), var_index=list(range(k, qc.num_qubits)), version="STATEVECTOR", backend=GenericBackendV2(qc.num_qubits*2-k+1, noise_info=False))
        
        variances.append(q_var)
        
        for i in range(len(Y)):
            for j in range(len(Y)):
                x = np.array(Y[i])
                x = x/(np.linalg.norm(x, 2) ** 2)
                y = np.array(Y[j])
                y = y/(np.linalg.norm(y, 2) ** 2)
                angle_list.append(np.dot(x,y))
        var = np.var(angle_list)
        
        #print("rapporti",[x/y for x,y in zip(qvar_list, angle_list)])


    return variances

def abod_qoda_hqabod():
    datasets = load_datasets([
        'Glass_withoutdupl_norm.arff',
        'Ionosphere_withoutdupl_norm.arff',
        'Shuttle_withoutdupl_norm_v01.arff',
        'WDBC_withoutdupl_norm_v01.arff',
        'WPBC_withoutdupl_norm.arff'
        #'creditcard.csv'
        ])
    contaminations = [0.04,0.36,0.01,0.03,0.24]
    i = 0
    for dataset, dataset_name in datasets:
        print("TEST DATASET " + dataset_name + ' - ' + str(dataset.shape))

        #variances_abod = classical_abod_no_normalization(dataset)
        data_abod = dataset.copy()
        data_qoda = dataset.copy()
        data_hqabod = dataset.copy()
        variances_abod = classical_abod_triu(data_abod)
        variances_hqabod = classical_abod_all(data_hqabod)
        variances_qoda = classical_abod_differences(data_qoda)
        
        data = zip(variances_abod, variances_qoda, variances_hqabod)
        df = pd.DataFrame(data, columns=['variances_abod', 'variances_qoda', 'variances_h-qabod'])
        df.to_csv(dataset_name+'_variances.csv')
        i = i + 1


def create_synth_datasets():

    # Parametri dati
    M = 256
    #N_values = [2, 4, 8, 16, 32]
    N_values = [2]
    mu_o = 0
    cov_o = 1
    mu_n = 10
    cov_n = 2
    steps = list(range(11))  # [0,1,2,3,4,5,6,7,8,9,10]
    #seeds = [123, 456, 789]
    seeds = [456]
    perc_outliers = 0.05
    M_o = int(perc_outliers*M)
    M_n = M - M_o

    # Directory di salvataggio
    output_dir = "generated_datasets"
    os.makedirs(output_dir, exist_ok=True)

    # Generazione dataset
    for N in N_values:
        print("N=",N)
        for step in steps:
            print(" step=",step)
            abod_a = []
            qoda_a = []
            hqabod_a = []
            abod_p = []
            qoda_p = []
            hqabod_p = []
            abod_r = []
            qoda_r = []
            hqabod_r = []
            abod_f1 = []
            qoda_f1 = []
            hqabod_f1 = []
            for seed in seeds:
                print("  seed=",seed)
                np.random.seed(seed)
                
                mu_o_step = mu_o + step  # Incrementa la media degli outlier
                
                # Media e covarianza per gli outlier e i non-outlier
                mean_outlier = np.full(N, mu_o_step)  # Vettore di media per gli outlier
                mean_non_outlier = np.full(N, mu_n)   # Vettore di media per i non-outlier
                cov_matrix_outlier = np.eye(N) * cov_o  # Matrice di covarianza per gli outlier
                cov_matrix_non_outlier = np.eye(N) * cov_n  # Matrice di covarianza per i non-outlier
                
                '''
                # Genera outlier (label 1)
                X_out, _ = make_gaussian_quantiles(mean=mu_o_step, cov=cov_o, n_samples=M_o, n_features=N, random_state=seed)
                y_out = np.ones(len(X_out))  # Label 1 (outlier)
                
                # Genera non-outlier (label 0)
                X_non_out, _ = make_gaussian_quantiles(mean=mu_n, cov=cov_n, n_samples=M_n, n_features=N, random_state=seed)
                y_non_out = np.zeros(len(X_non_out))  # Label 0 (non-outlier)
                '''
                # Genera outlier (label 1)
                X_out = np.random.multivariate_normal(mean_outlier, cov_matrix_outlier, size=M_o)
                y_out = np.ones(len(X_out))  # Label 1 (outlier)
                
                # Genera non-outlier (label 0)
                X_non_out = np.random.multivariate_normal(mean_non_outlier, cov_matrix_non_outlier, size=M_n)
                y_non_out = np.zeros(len(X_non_out))  # Label 0 (non-outlier)
                

                # Concatenazione dataset
                X = np.vstack([X_out, X_non_out])
                y = np.hstack([y_out, y_non_out])
                
                # Creazione DataFrame
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(N)])
                dataset = df.values
                df['label'] = y  # Aggiunge la colonna delle label
                
                '''
                # Salva il dataset
                filename = f"{output_dir}/dataset_N{N}_seed{seed}_step{step}.csv"
                df.to_csv(filename, index=False)

                plt.scatter(df['feature_0'], df['feature_1'], c=df['label'])
                plt.savefig(f"{output_dir}/dataset_N{N}_seed{seed}_step{step}.png")
                plt.close()            
                print(f"Salvato: {filename}")
                '''
                
                dataset_name = f"{output_dir}/dataset_N{N}_seed{seed}_step{step}"

                df["variances_abod"] = classical_abod(dataset, perc_outliers)
                df["variances_qoda"] = classical_abod_differences(dataset)
                df["variances_h-qabod"] = classical_abod_all(dataset)
                abod_predicted_index = df.sort_values(by=["variances_abod"])[:M_o].index
                qoda_predicted_index = df.sort_values(by=["variances_qoda"])[:M_o].index
                hqabod_predicted_index = df.sort_values(by=["variances_h-qabod"])[:M_o].index
                
                df['abod_predicted'] = 0
                df['qoda_predicted'] = 0
                df['hqabod_predicted'] = 0
                df.loc[abod_predicted_index, 'abod_predicted'] = 1
                df.loc[qoda_predicted_index, 'qoda_predicted'] = 1
                df.loc[hqabod_predicted_index, 'hqabod_predicted'] = 1
                
                abod_a.append(accuracy_score(df["label"], df["abod_predicted"]))
                qoda_a.append(accuracy_score(df["label"], df["qoda_predicted"]))
                hqabod_a.append(accuracy_score(df["label"], df["hqabod_predicted"]))
            
                abod_p.append(precision_score(df["label"], df["abod_predicted"]))
                qoda_p.append(precision_score(df["label"], df["qoda_predicted"]))
                hqabod_p.append(precision_score(df["label"], df["hqabod_predicted"]))
            
                abod_r.append(recall_score(df["label"], df["abod_predicted"]))
                qoda_r.append(recall_score(df["label"], df["qoda_predicted"]))
                hqabod_r.append(recall_score(df["label"], df["hqabod_predicted"]))
            
                abod_f1.append(f1_score(df["label"], df["abod_predicted"]))
                qoda_f1.append(f1_score(df["label"], df["qoda_predicted"]))
                hqabod_f1.append(f1_score(df["label"], df["hqabod_predicted"]))
            
            np.mean(abod_a)
            np.mean(qoda_a)
            np.mean(hqabod_a)
            np.std(abod_a)
            np.std(qoda_a)
            np.std(hqabod_a)
            
            np.mean(abod_p)
            np.mean(qoda_p)
            np.mean(hqabod_p)
            np.std(abod_p)
            np.std(qoda_p)
            np.std(hqabod_p)

            np.mean(abod_r)
            np.mean(qoda_r)
            np.mean(hqabod_r)
            np.std(abod_r)
            np.std(qoda_r)
            np.std(hqabod_r)

            np.mean(abod_f1)
            np.mean(qoda_f1)
            np.mean(hqabod_f1)
            np.std(abod_f1)
            np.std(qoda_f1)
            np.std(hqabod_f1)
        print("")

def test_synth_gaussian():
    # Parametri dati
    M = 128
    N_values = [2]
    mu_o = 0
    cov_o = 1
    mu_n = 10
    cov_n = 2
    steps = list(range(11))  # [0,1,2,3,4,5,6,7,8,9,10]
    seeds = [123, 456, 789]
    perc_outliers = 0.05
    M_o = int(perc_outliers * M)
    M_n = M - M_o

    # Directory di salvataggio
    output_dir = "generated_datasets"
    metrics_dir = "metrics"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Generazione dataset e metriche
    for N in N_values:
        print(f"N = {N}")

        # DataFrame per salvare le metriche finali
        results = []

        for step in steps:
            print(f"  Step = {step}")

            metrics = {
                "step": step,
                "abod_accuracy": [], "qoda_accuracy": [], "hqabod_accuracy": [],
                "abod_precision": [], "qoda_precision": [], "hqabod_precision": [],
                "abod_recall": [], "qoda_recall": [], "hqabod_recall": [],
                "abod_f1": [], "qoda_f1": [], "hqabod_f1": [],
                "abod_f1w": [], "qoda_f1w": [], "hqabod_f1w": [],
                "abod_auc": [], "qoda_auc": [], "hqabod_auc": []  
            }

            for seed in seeds:
                print(f"    Seed = {seed}")
                np.random.seed(seed)

                mu_o_step = mu_o + step  # Incrementa la media degli outlier
                
                # Media e covarianza
                mean_outlier = np.full(N, mu_o_step)
                mean_non_outlier = np.full(N, mu_n)
                cov_matrix_outlier = np.eye(N) * cov_o
                cov_matrix_non_outlier = np.eye(N) * cov_n

                # Genera outlier (label 1)
                X_out = np.random.multivariate_normal(mean_outlier, cov_matrix_outlier, size=M_o)
                y_out = np.ones(len(X_out))

                # Genera non-outlier (label 0)
                X_non_out = np.random.multivariate_normal(mean_non_outlier, cov_matrix_non_outlier, size=M_n)
                y_non_out = np.zeros(len(X_non_out))

                # Concatenazione dataset
                X = np.vstack([X_out, X_non_out])
                y = np.hstack([y_out, y_non_out])

                # Creazione DataFrame
                df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(N)])   

                # Calcolo varianze per i modelli
                df["variances_abod"] = classical_abod_triu(df.values)
                df["variances_qoda"] = classical_abod_differences(df.values)
                df["variances_hqabod"] = classical_abod_all(df.values)

                # Predizioni dei modelli
                abod_predicted_index = df.sort_values(by=["variances_abod"])[:M_o].index
                qoda_predicted_index = df.sort_values(by=["variances_qoda"])[:M_o].index
                hqabod_predicted_index = df.sort_values(by=["variances_hqabod"])[:M_o].index

                df['abod_predicted'] = 0
                df['qoda_predicted'] = 0
                df['hqabod_predicted'] = 0
                df.loc[abod_predicted_index, 'abod_predicted'] = 1
                df.loc[qoda_predicted_index, 'qoda_predicted'] = 1
                df.loc[hqabod_predicted_index, 'hqabod_predicted'] = 1
                
                df['label'] = y
                # Calcolo metriche
                for model in ["abod", "qoda", "hqabod"]:
                    metrics[f"{model}_accuracy"].append(accuracy_score(df["label"], df[f"{model}_predicted"]))
                    metrics[f"{model}_precision"].append(precision_score(df["label"], df[f"{model}_predicted"]))
                    metrics[f"{model}_recall"].append(recall_score(df["label"], df[f"{model}_predicted"]))
                    metrics[f"{model}_f1"].append(f1_score(df["label"], df[f"{model}_predicted"]))
                    metrics[f"{model}_f1w"].append(f1_score(df["label"], df[f"{model}_predicted"], average='weighted'))
                    metrics[f"{model}_auc"].append(roc_auc_score(df["label"], df[f"{model}_predicted"]))

            # Calcolo medie e std
            row = {"step": step}
            for metric in ["accuracy", "precision", "recall", "f1", "f1w", "auc"]:
                for model in ["abod", "qoda", "hqabod"]:
                    key = f"{model}_{metric}"
                    row[f"{key}_mean"] = np.mean(metrics[key])
                    row[f"{key}_std"] = np.std(metrics[key])

            results.append(row)

        # Salvataggio delle metriche
        results_df = pd.DataFrame(results)
        metrics_filename = f"{metrics_dir}/metrics_N{N}.csv"
        results_df.to_csv(metrics_filename, index=False)
        print(f"Salvate metriche in: {metrics_filename}\n")


if __name__ == "__main__":
    
    #test_synth_gaussian()
    #abod_qoda_hqabod()
    #test_old_heuristic()
    #classic_vs_quantum_real()
    #heuristic_evaluation_synth(data_distribution='uniform')
    #heuristic_evaluation_synth(data_distribution='normal')
    
    M = 5
    N = 2
    
    X_train, Y_train = generate_data(   
                                        n_train=M, 
                                        n_features=N,
                                        train_only=True,
                                        contamination=0.2,
                                        random_state=42
                                    )
                                    
    x1, x2 = X_train[:,0], X_train[:,1]
    
    
    q_vars = QABOD(X_train)
    print("\nQuantum variances\n")
    print(q_vars)
    
    c_vars = classical_abod_all(X_train)
    print("\nClassical variances\n")
    print(c_vars)

    
    