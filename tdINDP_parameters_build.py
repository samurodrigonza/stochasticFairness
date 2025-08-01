"""
tdINDP_parameters_build Python script

Author: Samuel Rodriguez-Gonzalez, PhD Candidate, The University of Oklahoma
Last modified: 2023-03-30

"""
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
import openpyxl as opxl
import networkx as nx
import pandas as pd
from math import log
import winsound
import time
import ast

def load_tdINDP_parameters():
	'''
	Load parameters and .csv files

	'''
	print_check = False
	csv_folder_path = os.path.join(os.path.dirname(__file__), 'csv_files_folder')
	'''
	Sets
	'''
	N = [] # N Set of nodes before a destructive event
	A = [] # A Set of arcs before a destructive event
	# T = [0,1,2,3] # Set of periods for the recovery process (time horizon)
	T = [0]
	# T = [t for t in range(0,2)] # Set of periods for the recovery process (time horizon)
	# T = [t for t in range(0,6)] # Set of periods for the recovery process (time horizon)
	# T = [t for t in range(0,10)] # Set of periods for the recovery process (time horizon)
	# T = [0] # Set of periods for the recovery process (time horizon)
	S = [] # Set of geographical spaces (spatial distribution of the area that contains the infrastructure networks)
	L = ['Gas', 'Power', 'Water']#, 'Interdependency'] #Set of commodities flowing in the system
	R = [1] # R Set of limited resources to be used in the reconstruction process
	K = ['Gas', 'Power', 'Water'] #Set of infrastructure networks
	N_star_k = {k:[] for k in K} # N_star_k Set of nodes in network k ∈ K that require their demands to be fully satisfied to be functional
	N_k = {k:[] for k in K} #Set of nodes in network k ∈ K before a destructive event
	N_prime_k = {k:[] for k in K} # N_prime_k Set of destroyed nodes in network k ∈ K after the event
	A_k = {k:[] for k in K} #Set of arcs in network k ∈ K before a destructive event
	A_prime_k = {k:[] for k in K} # A_prime_k Set of destroyed arcs in network k ∈ K after the event
	L_k = {k:[] for k in K} # L_k Set of commodities flowing in network k ∈ K

	# N_star_k['Power'].append('Power_node_9')
	# N_prime_k['Power'].append('Power_node_3')

	if print_check:
		print(f'N: {N}')
		print(f'A: {A}')
		print(f'T: {T}')
		print(f'L: {S}')
		print(f'L: {L}')
		print(f'R: {R}')
		print(f'K: {K}')
		print(f'N_star_k: {N_star_k}')
		print(f'N_k: {N_k}')
		print(f'A_k: {A_k}')

	'''
	Parameters
	'''
	v_value = [3, 6, 9, 12]
	v = {} # v_{rt} Availability of resource r at time t 3, 6, 9, 12
	h = {} # h_{ijkrt} Usage of resource r related to recovering arc (i, j) in network k at time t 1
	p = {} # p_{ikrt} Usage of resource r related to recovering node i in network k at time t 1
	M_plus = {} # M_plus_{iklt} Costs of excess of supply of commodity l in node i in network k at time t
	M_minus = {} # M_minus_{iklt} Costs of unsatisfied demand of commodity l in node i in network k at time t
	alpha = {} # alpha_{ijkst} Indicates if repairing arc (i, j) in network k at time t requires preparing space s
	beta = {} # beta_{ikst} Indicates if repairing node i in network k at time t requires preparing space s
	gamma = {} # gamma_{ijkk_tildet} Indicates if at time t node i in network k depends on node j in network k_tilde ∈ K
	g = {} # g_{st} Cost of preparing geographical space s at time t
	f = {} # f_{ijkt} Cost of recovering arc (i, j) in network k at time t
	q = {} # q_{ikt} Cost of recovering node i in network k at time t
	c = {} # c_{ijklt} Commodity l unitary flow cost through arc (i, j) in network k at time t
	u = {} # u_{ijkt} Total flow capacity of arc (i, j) in network k at time t
	b = {} # b_{iklt} Demand/supply of commodity l in node i in network k at time t

	if print_check:
		print(f'v: {v}')
		print(f'h: {h}')
		print(f'p: {p}')
		print(f'M_plus: {M_plus}')
		print(f'M_minus: {M_minus}')
		print(f'g: {g}')
		print(f'f: {f}')
		print(f'q: {q}')
		print(f'c: {c}')
		print(f'u: {u}')
		print(f'b: {b}')

	'''
	Geographical spaces
	'''
	prefix = 'Geographical_space_'
	csvGeoSpace = os.path.join(csv_folder_path, 'g.csv')
	dataGeoSpace = pd.read_csv(csvGeoSpace)
	dataGeoSpace['Subspace_ID'] = prefix + dataGeoSpace['Subspace_ID'].astype(str)
	# Fill in S
	S = dataGeoSpace['Subspace_ID'].tolist()

	# Fill in g
	for t in T:
		for row in dataGeoSpace.index:
			g[(dataGeoSpace['Subspace_ID'][row],t)] = dataGeoSpace['g'][row]

	if print_check:
		print(f'S: {S}')
		print(f'g: {g}')


	'''
	Gas
	'''
	# Nodes
	prefix = 'Gas_node_'
	csvGasNodes = os.path.join(csv_folder_path, 'GasNodes.csv')
	dataGasNodes = pd.read_csv(csvGasNodes)
	dataGasNodes['ID'] = prefix + dataGasNodes['ID'].astype(str)
	# Fill in N_k['Gas']
	N_k['Gas'] = dataGasNodes['ID'].tolist()

	# Fill in b, q, M_plus and M_minus
	for t in T:
		for row in dataGasNodes.index:
			b[(dataGasNodes['ID'][row],'Gas','Gas',t)] = dataGasNodes['Demand'][row]
			q[(dataGasNodes['ID'][row],'Gas',t)] = dataGasNodes['q (complete DS)'][row]
			M_plus[(dataGasNodes['ID'][row],'Gas','Gas',t)] = dataGasNodes['Mp'][row]
			M_minus[(dataGasNodes['ID'][row],'Gas','Gas',t)] = dataGasNodes['Mm'][row]


	if print_check:
		print(f'N_k: {N_k}')
		print(f'b: {b}')
		print(f'q: {q}')
		print(f'M_plus: {M_plus}')
		print(f'M_minus: {M_minus}')

	# Arcs
	csvGasArcs = os.path.join(csv_folder_path, 'GasArcs.csv')
	dataGasArcs = pd.read_csv(csvGasArcs)
	dataGasArcs['tuple_col'] = dataGasArcs.apply(lambda x: (prefix + str(x['Start Node']), prefix + str(x['End Node'])), axis=1)
	# Fill in A_k['Gas']
	A_k['Gas'] = dataGasArcs['tuple_col'].tolist()

	# Fill in c
	for t in T:
		for row in dataGasArcs.index:
			c[('Gas_node_'+str(dataGasArcs['Start Node'][row]),'Gas_node_'+str(dataGasArcs['End Node'][row]),'Gas','Gas',t)] = dataGasArcs['c'][row]

	# Fill in f
	for t in T:
		for row in dataGasArcs.index:
			f[('Gas_node_'+str(dataGasArcs['Start Node'][row]),'Gas_node_'+str(dataGasArcs['End Node'][row]),'Gas',t)] = dataGasArcs['f'][row]

	# Fill in u
	for t in T:
		for row in dataGasArcs.index:
			u[('Gas_node_'+str(dataGasArcs['Start Node'][row]),'Gas_node_'+str(dataGasArcs['End Node'][row]),'Gas',t)] = dataGasArcs['u'][row]

	if print_check:
		print(f'A_k: {A_k}')
		print(f'c: {c}')
		print(f'f: {f}')
		print(f'u: {u}')


	'''
	Power
	'''
	# Nodes
	prefix = 'Power_node_'
	csvPowerNodes = os.path.join(csv_folder_path, 'PowerNodes.csv')
	dataPowerNodes = pd.read_csv(csvPowerNodes)
	dataPowerNodes['ID'] = prefix + dataPowerNodes['ID'].astype(str)
	# Fill in N_k['Power']
	N_k['Power'] = dataPowerNodes['ID'].tolist()

	# Fill in b, q, M_plus and M_minus
	for t in T:
		for row in dataPowerNodes.index:
			b[(dataPowerNodes['ID'][row],'Power','Power',t)] = dataPowerNodes['Demand'][row]
			q[(dataPowerNodes['ID'][row],'Power',t)] = dataPowerNodes['q (complete DS)'][row]
			M_plus[(dataPowerNodes['ID'][row],'Power','Power',t)] = dataPowerNodes['Mp'][row]
			M_minus[(dataPowerNodes['ID'][row],'Power','Power',t)] = dataPowerNodes['Mm'][row]


	if print_check:
		print(f'N_k: {N_k}')
		print(f'b: {b}')
		print(f'q: {q}')
		print(f'M_plus: {M_plus}')
		print(f'M_minus: {M_minus}')

	# Arcs
	csvPowerArcs = os.path.join(csv_folder_path, 'PowerArcs.csv')
	dataPowerArcs = pd.read_csv(csvPowerArcs)
	dataPowerArcs['tuple_col'] = dataPowerArcs.apply(lambda x: (prefix + str(x['Start Node']), prefix + str(x['End Node'])), axis=1)
	# Fill in A_k['Power']
	A_k['Power'] = dataPowerArcs['tuple_col'].tolist()

	# Fill in c
	for t in T:
		for row in dataPowerArcs.index:
			c[('Power_node_'+str(dataPowerArcs['Start Node'][row]),'Power_node_'+str(dataPowerArcs['End Node'][row]),'Power','Power',t)] = dataPowerArcs['c'][row]

	# Fill in f
	for t in T:
		for row in dataPowerArcs.index:
			f[('Power_node_'+str(dataPowerArcs['Start Node'][row]),'Power_node_'+str(dataPowerArcs['End Node'][row]),'Power',t)] = dataPowerArcs['f'][row]

	# Fill in u
	for t in T:
		for row in dataPowerArcs.index:
			u[('Power_node_'+str(dataPowerArcs['Start Node'][row]),'Power_node_'+str(dataPowerArcs['End Node'][row]),'Power',t)] = dataPowerArcs['u'][row]
	if print_check:
		print(f'A_k: {A_k}')
		print(f'c: {c}')
		print(f'f: {f}')
		print(f'u: {u}')

	'''
	Telecommunication
	'''
	# # Nodes
	# prefix = 'Telecommunication_node_'
	# dataTelecommunicationNodes = pd.read_csv("TelecommunicationNodes.csv")
	# dataTelecommunicationNodes['ID'] = prefix + dataTelecommunicationNodes['ID'].astype(str)
	# # Fill in N_k['Telecommunication']
	# N_k['Telecommunication'] = dataTelecommunicationNodes['ID'].tolist()

	# # Fill in b, q, M_plus and M_minus
	# for t in T:
	# 	for row in dataTelecommunicationNodes.index:
	# 		b[(dataTelecommunicationNodes['ID'][row],'Telecommunication','Telecommunication',t)] = dataTelecommunicationNodes['Demand'][row]
	# 		q[(dataTelecommunicationNodes['ID'][row],'Telecommunication',t)] = dataTelecommunicationNodes['q (complete DS)'][row]
	# 		M_plus[(dataTelecommunicationNodes['ID'][row],'Telecommunication','Telecommunication',t)] = dataTelecommunicationNodes['Mp'][row]
	# 		M_minus[(dataTelecommunicationNodes['ID'][row],'Telecommunication','Telecommunication',t)] = dataTelecommunicationNodes['Mm'][row]


	# if print_check:
	# 	print(f'N_k: {N_k}')
	# 	print(f'b: {b}')
	# 	print(f'q: {q}')
	# 	print(f'M_plus: {M_plus}')
	# 	print(f'M_minus: {M_minus}')

	# # Arcs
	# dataTelecommunicationArcs = pd.read_csv("TelecommunicationArcs.csv")
	# dataTelecommunicationArcs['tuple_col'] = dataTelecommunicationArcs.apply(lambda x: (prefix + str(x['Start Node']), prefix + str(x['End Node'])), axis=1)
	# # Fill in A_k['Telecommunication']
	# A_k['Telecommunication'] = dataTelecommunicationArcs['tuple_col'].tolist()

	# # Fill in c
	# for t in T:
	# 	for row in dataTelecommunicationArcs.index:
	# 		c[('Telecommunication_node_'+str(dataTelecommunicationArcs['Start Node'][row]),'Telecommunication_node_'+str(dataTelecommunicationArcs['End Node'][row]),'Telecommunication','Telecommunication',t)] = dataTelecommunicationArcs['c'][row]

	# # Fill in f
	# for t in T:
	# 	for row in dataTelecommunicationArcs.index:
	# 		f[('Telecommunication_node_'+str(dataTelecommunicationArcs['Start Node'][row]),'Telecommunication_node_'+str(dataTelecommunicationArcs['End Node'][row]),'Telecommunication',t)] = dataTelecommunicationArcs['f'][row]

	# # Fill in u
	# for t in T:
	# 	for row in dataTelecommunicationArcs.index:
	# 		u[('Telecommunication_node_'+str(dataTelecommunicationArcs['Start Node'][row]),'Telecommunication_node_'+str(dataTelecommunicationArcs['End Node'][row]),'Telecommunication',t)] = dataTelecommunicationArcs['u'][row]
	# if print_check:
	# 	print(f'A_k: {A_k}')
	# 	print(f'c: {c}')
	# 	print(f'f: {f}')
	# 	print(f'u: {u}')



	'''
	Water
	'''
	# Nodes
	prefix = 'Water_node_'
	csvWaterNodes = os.path.join(csv_folder_path, 'WaterNodes.csv')
	dataWaterNodes = pd.read_csv(csvWaterNodes)
	dataWaterNodes['ID'] = prefix + dataWaterNodes['ID'].astype(str)
	# Fill in N_k['Water']
	N_k['Water'] = dataWaterNodes['ID'].tolist()

	# Fill in b, q, M_plus and M_minus
	for t in T:
		for row in dataWaterNodes.index:
			b[(dataWaterNodes['ID'][row],'Water','Water',t)] = dataWaterNodes['Demand'][row]
			q[(dataWaterNodes['ID'][row],'Water',t)] = dataWaterNodes['q (complete DS)'][row]
			M_plus[(dataWaterNodes['ID'][row],'Water','Water',t)] = dataWaterNodes['Mp'][row]
			M_minus[(dataWaterNodes['ID'][row],'Water','Water',t)] = dataWaterNodes['Mm'][row]


	if print_check:
		print(f'N_k: {N_k}')
		print(f'b: {b}')
		print(f'q: {q}')
		print(f'M_plus: {M_plus}')
		print(f'M_minus: {M_minus}')

	# Arcs
	csvWaterArcs = os.path.join(csv_folder_path, 'WaterArcs.csv')
	dataWaterArcs = pd.read_csv(csvWaterArcs)
	dataWaterArcs['tuple_col'] = dataWaterArcs.apply(lambda x: (prefix + str(x['Start Node']), prefix + str(x['End Node'])), axis=1)
	# Fill in A_k['Water']
	A_k['Water'] = dataWaterArcs['tuple_col'].tolist()

	# Fill in c
	for t in T:
		for row in dataWaterArcs.index:
			c[('Water_node_'+str(dataWaterArcs['Start Node'][row]),'Water_node_'+str(dataWaterArcs['End Node'][row]),'Water','Water',t)] = dataWaterArcs['c'][row]

	# Fill in f
	for t in T:
		for row in dataWaterArcs.index:
			f[('Water_node_'+str(dataWaterArcs['Start Node'][row]),'Water_node_'+str(dataWaterArcs['End Node'][row]),'Water',t)] = dataWaterArcs['f'][row]

	# Fill in u
	for t in T:
		for row in dataWaterArcs.index:
			u[('Water_node_'+str(dataWaterArcs['Start Node'][row]),'Water_node_'+str(dataWaterArcs['End Node'][row]),'Water',t)] = dataWaterArcs['u'][row]

	if print_check:
		print(f'A_k: {A_k}')
		print(f'c: {c}')
		print(f'f: {f}')
		print(f'u: {u}')



	'''
	Interdependencies
	'''

	'''
	Fill in gamma'''
	csvInterdependencies = os.path.join(csv_folder_path, "Interdep.csv")
	dataInterdependencies = pd.read_csv(csvInterdependencies)

	for t in T:
		for row in dataInterdependencies.index:
			dependee_node = dataInterdependencies['Dependee Node'][row]
			dependee_network = dataInterdependencies['Dependee Network'][row]
			depender_node = dataInterdependencies['Depender Node'][row]
			depender_network = dataInterdependencies['Depender Network'][row]
			if dependee_network in K and depender_network in K:
				gamma[(str(dependee_network)+'_node_'+str(dependee_node),
					   str(depender_network)+'_node_'+str(depender_node),
					   dependee_network,
					   depender_network,
					   t)] = 1
				#b_{iklt} Demand/supply of commodity l in node i in network k at time t

	# gamma_{ijkk_tildet} Indicates if at time t node i in network k depends on node j in network k_tilde ∈ K
	if print_check:
		print(f'gamma: {gamma}')


	'''
	Fill in N, A, and L_k
	'''

	for k in K:
		N = N + N_k[k]
		A = A + A_k[k]
		L_k[k] = [k]#, 'Interdependency']

	# for interdependent_arc in gamma.keys():
	# 	A = A + [(interdependent_arc[1],interdependent_arc[0])]


	# # Bidirectionality
	for k in K:
		for (i,j) in A_k[k]:
			if (j,i) not in A_k[k]:
				 A_k[k].append((j,i))
	for k in K:
		for (i,j) in A_k[k]:
			for t in T:
				if (i,j,k,t) in f.keys() and (j,i,k,t) not in f.keys():
					f[(j,i,k,t)] = f[(i,j,k,t)]
				if (i,j,k,t) in u.keys() and (j,i,k,t) not in u.keys():
					u[(j,i,k,t)] = u[(i,j,k,t)]
				for l in L_k[k]:
					if (i,j,k,l,t) in c.keys() and (j,i,k,l,t) not in c.keys():
						c[(j,i,k,l,t)] = c[(i,j,k,l,t)]

	if print_check:
		print(f'N: {N}')
		print(f'A: {A}')
		print(f'L_k: {L_k}')

	'''
	Fill in alpha and beta
	'''

	# alpha_{ijkst} Indicates if repairing arc (i, j) in network k at time t requires preparing space s
	# beta_{ikst} Indicates if repairing node i in network k at time t requires preparing space s

	csvAlphaBeta = os.path.join(csv_folder_path, "beta.csv")
	dataAlphaBeta = pd.read_csv(csvAlphaBeta)
	for t in T:
		for row in dataAlphaBeta.index:
			network = dataAlphaBeta['Network'][row]
			if network in K:
				start_node = dataAlphaBeta['Start Node'][row]
				end_node = dataAlphaBeta['End Node'][row]
				sub_space = dataAlphaBeta['Subspace'][row]
				alpha[(str(network)+'_node_'+str(start_node),
					   str(network)+'_node_'+str(end_node),
					   network,
					   'Geographical_space_'+str(sub_space),
					   t)] = 1
				beta[(str(network)+'_node_'+str(start_node),
					   network,
					   'Geographical_space_'+str(sub_space),
					   t)] = 1
				beta[(str(network)+'_node_'+str(end_node),
					   network,
					   'Geographical_space_'+str(sub_space),
					   t)] = 1
	
	alpha_keys = alpha.keys()
	for k in K:
		for s in S:
			for t in T:
				for (i,j) in A_k[k]:
					if (i,j,k,s,t) in alpha_keys and (j,i,k,s,t) not in alpha.keys():
						alpha[(j,i,k,s,t)] = alpha[(i,j,k,s,t)] 
	if print_check:
		print(alpha)
		print(beta)

	# Fill in v, h, and p
	# v_{rt} Availability of resource r at time t 3, 6, 9, 12
	for r in R:
		for t in T:
			v[(r,t)] = v_value[-1]
			# v[(r,t)] = 6 #v_value[0]

	# h_{ijkrt} Usage of resource r related to recovering arc (i, j) in network k at time t 1
	# p_{ikrt} Usage of resource r related to recovering node i in network k at time t 1
	for t in T:
		for k in K:
			for (i,j) in A_k[k]:
				h[(i,j,k,r,t)] = 1/2
			for i in N_k[k]:
				p[(i,k,r,t)] = 1

	full_disruption = False

	if full_disruption:
		for k in K:
			for i in N_k[k]:
				N_prime_k[k].append(i)
			for (i,j) in A_k[k]:
				A_prime_k[k].append((i,j))


	return N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b

N, A, T, S, L, R, K, N_star_k, N_k, N_prime_k, A_k, A_prime_k, L_k, v, h, p, M_plus, M_minus, alpha, beta, gamma, g, f, q, c, u, b = load_tdINDP_parameters()

def disrput(disruption_size, K = K, N_prime_k = N_prime_k, A_prime_k = A_prime_k):
	
	csv_folder_paths = {0:'none',
						3:'03_dmg_set_00_sc_20',
						7:'07_dmg_set_49_sc_54',
						12:'12_dmg_set_04_sc_78',
						19:'19_dmg_set_47_sc_89',
						26:'26_dmg_set_30_sc_03',
						33:'33_dmg_set_01_sc_13',
						40:'40_dmg_set_44_sc_60',
						48:'48_dmg_set_30_sc_92',
						52:'52_dmg_set_04_sc_46',
						56:'56_dmg_set_15_sc_39'
							}
	
	if disruption_size == 0:
		return N_prime_k, A_prime_k 
	else:
		
		if disruption_size in csv_folder_paths: 
			
			csv_folder_path = os.path.join(os.path.dirname(__file__), csv_folder_paths[disruption_size])

			disrupted_nodes_file_names = ['Net_Gas_Damaged_Nodes',
		              					  'Net_Power_Damaged_Nodes',
		              					  #'Net_Telecommunication_Damaged_Nodes',
		              					  'Net_Water_Damaged_Nodes']

			disrupted_arcs_file_names = ['Net_Gas_Damaged_Arcs',
		    							 'Net_Power_Damaged_Arcs',
		    							 #'Net_Telecommunication_Damaged_Arcs',
		    							 'Net_Water_Damaged_Arcs']
			prefix = ['Gas_node_',
		    		  'Power_node_',
		    		  #'Telecommunication_node_',
		    		  'Water_node_']

			for k in range(len(K)):

				disrupted_nodes_file = os.path.join(csv_folder_path, str(disrupted_nodes_file_names[k])+'.csv')
				disrupted_nodes_k_df = pd.read_csv(disrupted_nodes_file)
				if disrupted_nodes_k_df.shape[0] > 0:
					disrupted_nodes_k_df['node_i'] = prefix[k] + disrupted_nodes_k_df['node_i'].astype(str)
					N_prime_k[K[k]] = disrupted_nodes_k_df['node_i'].tolist()

				disrupted_arcs_file = os.path.join(csv_folder_path, str(disrupted_arcs_file_names[k])+'.csv')
				disrupted_arcs_k_df = pd.read_csv(disrupted_arcs_file)
				if disrupted_arcs_k_df.shape[0] > 0:
					disrupted_arcs_k_df['tuple_col'] = disrupted_arcs_k_df.apply(lambda x: (prefix[k] + str(x['node_i']), prefix[k] + str(x['node_j'])), axis=1)
					A_prime_k[K[k]] = disrupted_arcs_k_df['tuple_col'].tolist()

			#Bidirectionality
			for k in K:
				for (i,j) in A_prime_k[k]:
					if (j,i) not in A_prime_k[k]:
						 A_prime_k[k].append((j,i))
			
			return N_prime_k, A_prime_k 

		else:
			print('No such disruption size in directory')
			sys.exit()

def find_capacities(N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k, N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p, M_plus=M_plus, M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g, f=f, q=q, c=c, u=u, b=b):
	
	build = False
	
	if build:

		def tdINDP_free_cap(N=N, A=A, T=T, S=S, L=L, R=R, K=K, N_star_k=N_star_k, N_k=N_k, N_prime_k=N_prime_k, A_k=A_k, A_prime_k=A_prime_k, L_k=L_k, v=v, h=h, p=p, M_plus=M_plus, M_minus=M_minus, alpha=alpha, beta=beta, gamma=gamma, g=g, f=f, q=q, c=c, b=b):	
		
			
			model = Model('tdINDP')
			model.setParam(GRB.Param.OutputFlag, 0)
			model.setParam('InfUnbdInfo',1)
			
			'''
			Fairness constarints just in case
			'''
			N_k_d = {k:[i for i in N_k[k] if b[(i,k,k,0)] < 0]  for k in K}
			# print(f'N_k_l_d: {N_k_l_d}\n')
			N_k_s = {k:[i for i in N_k[k] if b[(i,k,k,0)] > 0] for k in K}
			# print(f'N_k_l_s: {N_k_l_s}\n')
			N_k_nod = {k:[i for i in N_k[k] if b[(i,k,k,0)] == 0] for k in K}

			total_cost = model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name = 'total_cost')
			delta_plus = {(i,k,l,t): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_plus_'+str((i,k,l,t))) for k in K for i in N_k_s[k] for l in L_k[k] for t in T}
			delta_minus = {(i,k,l,t): model.addVar(vtype = GRB.CONTINUOUS, name = 'delta_minus_'+str((i,k,l,t))) for k in K for i in N_k_d[k] for l in L_k[k] for t in T}
			x = {(i,j,k,l,t): model.addVar(vtype = GRB.CONTINUOUS, name = 'x_'+str((i,j,k,l,t))) for k in K for (i,j) in A_k[k] for l in L_k[k] for t in T}
			u = {(i,j,k,t): model.addVar(vtype = GRB.CONTINUOUS, name = 'u_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T}
			log_u = {(i,j,k,t): model.addVar(vtype = GRB.CONTINUOUS, name = 'log_u_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T}
			w = {(i,k,t): model.addVar(vtype = GRB.BINARY, name = 'w_'+str((i,k,t))) for k in K for i in N_k[k] for t in T}
			y = {(i,j,k,t): model.addVar(vtype = GRB.BINARY, name = 'y_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T}
			DELTA_w = {(i,k,t): model.addVar(vtype = GRB.BINARY, name = 'DELTA_w_'+str((i,k,t))) for k in K for i in N_k[k] for t in T if t > 0}
			DELTA_y = {(i,j,k,t): model.addVar(vtype = GRB.BINARY, name = 'DELTA_y_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T if t > 0}
			DELTA_z = {(s,t): model.addVar(vtype = GRB.BINARY, name = 'DELTA_z_'+str((s,t))) for s in S for t in T if t > 0}

			cost_1 = quicksum(g[(s,t)]*DELTA_z[(s,t)] for s in S for t in T if t > 0)
			cost_2 = quicksum(f[(i,j,k,t)]*DELTA_y[(i,j,k,t)] for k in K for (i,j) in A_prime_k[k] for t in T if A_prime_k[k] and t > 0)
			cost_3 = quicksum(q[(i,k,t)]*DELTA_w[(i,k,t)] for k in K for i in N_prime_k[k] for t in T if N_prime_k[k] and t > 0)
			cost_4 = quicksum(M_plus[(i,k,l,t)]*delta_plus[(i,k,l,t)]  for k in K for l in L_k[k] for i in N_k_s[k] for t in T)  + quicksum(M_minus[(i,k,l,t)]*delta_minus[(i,k,l,t)] for k in K for l in L_k[k] for i in N_k_d[k] for t in T)
			cost_5 = quicksum(c[(i,j,k,l,t)]*x[(i,j,k,l,t)] for k in K for l in L_k[k] for (i,j) in A_k[k] for t in T)	

			model.addConstr(total_cost == cost_1+cost_2+cost_3+cost_4+cost_5, name = 'total_cost_constraint')
			# model.setObjective(total_cost, GRB.MINIMIZE)
			# model.update()
			# f1 = {j.varName:j.obj for j in model.getVars()} 

			
			for k in K:
				for i in N_k[k]:
					for l in L_k[k]:
						for t in T:
							if i in N_k_d[k]:
								model.addConstr(quicksum(x[(i,j,k,l,t)] for j in N_k[k] if (i,j) in A_k[k]) - 
												quicksum(x[(j,i,k,l,t)] for j in N_k[k] if (j,i) in A_k[k]) == 
												b[(i,k,l,t)] + delta_minus[(i,k,l,t)], name = 'A_2_'+str((i,k,l,t)))
							elif i in N_k_s[k]:
								model.addConstr(quicksum(x[(i,j,k,l,t)] for j in N_k[k] if (i,j) in A_k[k]) - 
												quicksum(x[(j,i,k,l,t)] for j in N_k[k] if (j,i) in A_k[k]) == 
												b[(i,k,l,t)] - delta_plus[(i,k,l,t)], name = 'A_2_'+str((i,k,l,t)))
							else:
								model.addConstr(quicksum(x[(i,j,k,l,t)] for j in N_k[k] if (i,j) in A_k[k]) - 
												quicksum(x[(j,i,k,l,t)] for j in N_k[k] if (j,i) in A_k[k]) == 
												b[(i,k,l,t)], name = 'A_2_'+str((i,k,l,t)))

			for k in K:
				for (i,j) in A_k[k]:
					for t in T:
						model.addConstr(quicksum(x[(i,j,k,l,t)] for l in L_k[k]) <=
										u[(i,j,k,t)]*y[(i,j,k,t)], name = 'A_3_'+str((k,i,j,t)))

						model.addConstr(quicksum(x[(i,j,k,l,t)] for l in L_k[k]) <=
										u[(i,j,k,t)]*w[(i,k,t)], name = 'A_4_'+str((k,i,j,t)))
						
						model.addConstr(quicksum(x[(i,j,k,l,t)] for l in L_k[k]) <=
										u[(i,j,k,t)]*w[(j,k,t)], name = 'A_5_'+str((k,i,j,t)))

			for k in K:
				if N_star_k[k]:
					for i in N_star_k[k]:
						for l in L_k[k]:
							for t in T:
								model.addConstr(w[(i,k,t)]*abs(b[(i,k,l,t)]) <=
												abs(b[(i,k,l,t)])-delta_minus[(i,k,l,t)], name = 'A_6_'+str((k,i,l,t)))
			for k in K:
				for k_tilde in K:
					for j in N_k[k_tilde]:
						for t in T:
							if (i,j,k,k_tilde,t) in gamma.keys():
								model.addConstr(quicksum(w[(i,k,t)]*gamma[(i,j,k,k_tilde,t)] for i in N_k[k]) >= 
												w[(j,k_tilde,t)], name = 'A_7_'+str((k,k_tilde,j,t)))

			for k in K:
				if N_prime_k[k]:
					for i in N_prime_k[k]:
						model.addConstr(w[(i,k,0)] ==  
										0, name = 'A_8_'+str((k,i)))

			for k in K:
				if A_prime_k[k]:
					for (i,j) in A_prime_k[k]:
						model.addConstr(y[(i,j,k,0)] ==  
										0, name = 'A_9_'+str((k,i,j)))

			for k in K:
				if N_prime_k[k]:
					for i in N_prime_k[k]:
						for t in T:
							if t > 0:
								model.addConstr(w[(i,k,t)] <= 
												quicksum(DELTA_w[(i,k,t_tilde)] for t_tilde in range(1,t+1)), name = 'A_10_'+str((k,i,t)))

			for k in K:
				if A_prime_k[k]:
					for (i,j) in A_prime_k[k]:
						for t in T:
							if t > 0:
								model.addConstr(y[(i,j,k,t)] <= 
												quicksum(DELTA_y[(i,j,k,t_tilde)] for t_tilde in range(1,t+1)), name = 'A_11_'+str((k,i,j,t)))

			for r in R:
				for t in T:
					if t > 0:
						model.addConstr(quicksum(h[(i,j,k,r,t)]*DELTA_y[(i,j,k,t)] for k in K for (i,j) in A_prime_k[k]) + 
										quicksum(p[(i,k,r,t)]*DELTA_w[(i,k,t)] for k in K for i in N_prime_k[k]) <= 
										v[(r,t)], name = 'A_12_'+str((r,t)))

			for k in K:
				if N_prime_k[k]:
					for i in N_prime_k[k]:
						for s in S:
							for t in T:
								if t > 0 and (i,k,s,t) in beta.keys():
									model.addConstr(DELTA_w[(i,k,t)]*beta[(i,k,s,t)] <=
													DELTA_z[(s,t)], name = 'A_13_'+str((k,i,s,t)))

			for k in K:
				if A_prime_k[k]:
					for (i,j) in A_prime_k[k]:
						for s in S:
							for t in T:
								if t > 0 and (i,j,k,s,t) in alpha.keys():
									model.addConstr(DELTA_y[(i,j,k,t)]*alpha[(i,j,k,s,t)] <=
													DELTA_z[(s,t)], name = 'A_13_'+str((k,i,j,s,t)))

			model.update()
			
			for k in K:
				for i in N_k_d[k]:
					for l in L_k[k]:
						for t in T:
							model.addConstr(delta_minus[(i,k,l,t)] == 0, name = 'meet_demand_'+str((i,k,l,t)))

			for k in K:
					for t in T:
						for (i,j) in A_k[k]:
							model.addGenConstrLog(u[(i,j,k,t)], log_u[(i,j,k,t)], name = 'log_u_constr_'+str((i,j,k,t)))
							if (j,i) in A_k[k]:
								model.addConstr(u[(i,j,k,t)] == u[(j,i,k,t)], name = 'bidir_cap_'+str((i,j,k,t)))
							for t_tilde in T:
								if t != t_tilde:
									model.addConstr(u[(i,j,k,t)] == u[(i,j,k,t_tilde)], name = 'homogen_time_cap_'+str((i,j,k,t)))

			model.setObjective(quicksum(u[(i,j,k,t)] for k in K for (i,j) in A_k[k] for t in T), GRB.MINIMIZE)
			# model.setObjective(quicksum(delta_minus[(i,k,l,t)] for k in K for l in L_k[k] for i in N_k_d[k] for t in T), GRB.MINIMIZE)
			model.update()
			
			return model

		model = tdINDP_free_cap()
		model.optimize()
		if model.status == 2:

			optimal_agg_cap_value = model.objVal
			print(f'optimal_agg_cap_value: {optimal_agg_cap_value}') 
			u_agg_cap = {}
			for v in model.getVars():
				if v.varName[0] == 'u':
					u_agg_cap[ast.literal_eval(v.varName[2:])] = v.x 
					# print(f'{v.varName}: {v.x}')
					# print(u_agg_cap)
			# print()
			# for v in model.getVars():
			# 	if v.varName[0:7] == 'delta_m' and v.x > 1e-4:
			# 		print(f'{v.varName}: {v.x}')

		elif model.status == 3:
			model.computeIIS()
			for c in model.getConstrs():
				if c.IISConstr == 1:
					print(f'{c.constrName}: {c.IISConstr}')

		del model
		
		print()

		model = tdINDP_free_cap()
		model.setObjective(quicksum(model.getVarByName('log_u_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T), GRB.MAXIMIZE)
		model.update()
		u_k = {k:[u_agg_cap[(i,j,k,0)] for (i,j) in A_k[k]] for k in K}
		for k in K:
			# print(math.log(max(u_k[k])))
			for (i,j) in A_k[k]:
				for t in T:
					model.addConstr(model.getVarByName('log_u_'+str((i,j,k,t))) <= math.log(max(u_k[k])), name = 'max_cap_'+str((i,j,k,t)))
					model.update()
		model.optimize()
		if model.status == 2:
			optimal_agg_log_cap_value = model.objVal
			print(f'optimal_agg_log_cap_value: {optimal_agg_log_cap_value}') 
			u_agg_log_cap = {}
			for v in model.getVars():
				if v.varName[0] == 'u':
					u_agg_log_cap[ast.literal_eval(v.varName[2:])] = v.x 
					# print(f'{v.varName}: {v.x}')
					# print(u_agg_cap)
			# print()
			# for v in model.getVars():
			# 	if v.varName[0:7] == 'delta_m' and v.x > 1e-4:
			# 		print(f'{v.varName}: {v.x}')

		elif model.status == 3:
			model.computeIIS()
			for c in model.getConstrs():
				if c.IISConstr == 1:
					print(f'{c.constrName}: {c.IISConstr}')
		
		print()
		# for key in u_agg_cap.keys():
		# 	# print(f'u_{key}: u_agg_cap = {u_agg_cap[key]} vs u_agg_log_cap = {u_agg_log_cap[key]}')
		# 	print(f'u_{key} abs diff: {round(abs(u_agg_cap[key]-u_agg_log_cap[key]),2)}')

		worst_u_agg_log_cap = sum(math.log(u_agg_cap[key]) for key in u_agg_cap.keys())
		actual_agg_cap_value = sum(u_agg_log_cap[key] for key in u_agg_log_cap.keys())
		print(f'worst_u_agg_log_cap: {worst_u_agg_log_cap}')
		pareto_solutions = []
		epsilon = 1
		u_agg_cap['sum_log_u'] = worst_u_agg_log_cap
		u_agg_cap['sum_u'] = optimal_agg_cap_value
		pareto_solutions.append(u_agg_cap)
		while actual_agg_cap_value > optimal_agg_cap_value and epsilon > 1e-1:
			print(f'epsilon: {epsilon}, actual_agg_cap_value: {actual_agg_cap_value}, actual_agg_log_cap_value: {optimal_agg_log_cap_value*epsilon}')
			model = tdINDP_free_cap()
			model.setObjective(quicksum(model.getVarByName('log_u_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T), GRB.MAXIMIZE)
			model.update()
			for k in K:
				for (i,j) in A_k[k]:
					for t in T:
						model.addConstr(model.getVarByName('log_u_'+str((i,j,k,t))) <= math.log(max(u_k[k])), name = 'max_cap_'+str((i,j,k,t)))
						model.update()
			model.addConstr(quicksum(model.getVarByName('u_'+str((i,j,k,t))) for k in K for (i,j) in A_k[k] for t in T) <= actual_agg_cap_value*epsilon)
			model.update()
			model.optimize()
			if model.status == 2:
				actual_agg_cap_value
				actual_u = {ast.literal_eval(v.varName[2:]): v.x for v in model.getVars() if v.varName[0] == 'u'}
				actual_agg_cap_value = sum(v.x for v in model.getVars() if v.varName[0] == 'u')
				actual_u['sum_log_u'] = sum(math.log(v.x+1e-6) for v in model.getVars() if v.varName[0] == 'u')
				actual_u['sum_u'] = actual_agg_cap_value
				if actual_u not in pareto_solutions:
					pareto_solutions.append(actual_u)
			epsilon*=0.99
		data = {var_name:[] for var_name in pareto_solutions[0].keys()}
		for thy_solution in pareto_solutions:
			for var_name in thy_solution.keys():
				data[var_name].append(float(thy_solution[var_name]))
				
					
		data = pd.DataFrame(data, columns = data.keys())
		data.to_csv('pareto_of_capacities.csv')
	else:
		
		# Read the CSV into a DataFrame
		df = pd.read_csv("pareto_of_capacities.csv")
		df = df.drop(df.columns[0], axis = 1)
		# Define the row number you want to access
		row_number = 18  # Change this to the desired row number (0-based)

		# Get the row as a Series
		row = df.iloc[row_number]

		# Define the list of column names to exclude from the dictionary keys
		exclude_columns = ["sum_log_u","sum_u"]

		# Create the dictionary
		u = {ast.literal_eval(column): value for column, value in row.items() if column not in exclude_columns}

		if len(T) > 1:
			for k in K:
				for (i,j) in A_k[k]:
					for t in T:
						if t > 0:
							u[(i,j,k,t)] = u[(i,j,k,0)] 

		return u

