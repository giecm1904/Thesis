import datetime
import itertools
from typing import List, Tuple

import numpy as np
from ortools.linear_solver import pywraplp

import pandas as pd
from math import radians, cos, sin, asin, sqrt, isnan
import random
from ortools.sat.python import cp_model

class Data():
    sources: List[str] = []
    nodes: List[str] = []
    functions: List[str] = []   ## ¿¿¿??? (does this represents carts/catalog/shipping/order/etc...?) ## YES

    node_memory_matrix: np.array = np.array([])
    function_memory_matrix: np.array = np.array([])
    node_delay_matrix: np.array = np.array([])
    workload_matrix: np.array = np.array([])
    max_delay_matrix: np.array = np.array([]) 
    # response_time_matrix: np.array = np.array([])   # We don't need it because is only use for GPU, right? ## DON'T USE IT
    node_cores_matrix: np.array = np.array([])
    # cores_matrix: np.array = np.array([])           # Where is it used? ## DON'T USE IT
    # old_allocations_matrix: np.array = np.array([]) # Where is it used? In our case it would be mat_mul? # IGNORE IT
    core_per_req_matrix: np.array = np.array([])

    ### gpu_function_memory_matrix: np.array = np.array([])
    ### gpu_node_memory_matrix: np.array = np.array([])

    # prev_x = np.array([])  ## Use by neptune for GPU ....


    def __init__(self, sources: List[str], nodes: List[str], functions: List[str]):
        self.sources = sources
        self.nodes = nodes
        self.functions = functions


class Input:
    nodes: List[str] = []
    functions: List[str] = []

    node_memory_matrix: np.array = np.array([])
    function_memory_matrix: np.array = np.array([])
    node_delay_matrix: np.array = np.array([])
    workload_matrix: np.array = np.array([])
    max_delay_matrix: np.array = np.array([])
    response_time_matrix: np.array = np.array([])
    node_memory: np.array = np.array([])
    function_memory: np.array = np.array([])
    node_cores: np.array = np.array([])
    cores_matrix: np.array = np.array([])

    cpu_nodes: List[str] = []
    cpu_functions: List[str] = []

    cpu_node_memory_matrix: np.array = np.array([])
    cpu_function_memory_matrix: np.array = np.array([])
    cpu_node_delay_matrix: np.array = np.array([])
    cpu_workload_matrix: np.array = np.array([])
    cpu_max_delay_matrix: np.array = np.array([])
    cpu_response_time_matrix: np.array = np.array([])
    cpu_node_memory: np.array = np.array([])
    cpu_function_memory: np.array = np.array([])

    cpu_node_cores: np.array = np.array([])
    cpu_cores_matrix: np.array = np.array([])
    cpu_actual_allocation: np.array = np.array([])
    cpu_core_per_req: np.array = np.array([])

    # x_cpu = np.array([])
    # c_cpu = np.array([])

    # gpu_nodes: List[str] = []
    # gpu_functions: List[str] = []

    # gpu_node_memory_matrix: np.array = np.array([])
    # gpu_function_memory_matrix: np.array = np.array([])
    # gpu_node_delay_matrix: np.array = np.array([])
    # gpu_workload_matrix: np.array = np.array([])
    # gpu_max_delay_matrix: np.array = np.array([])
    # gpu_response_time_matrix: np.array = np.array([])
    # gpu_node_memory: np.array = np.array([])
    # gpu_function_memory: np.array = np.array([])

    # x_gpu = np.array([])
    # c_gpu = np.array([])

    # cpu_function_gpu_map = {}

    def __init__(self,
                 cpu_nodes: List[str], cpu_functions: List[str],
                 gpu_nodes: List[str], gpu_functions: List[str]):
        # Initialize attributes
        self.cpu_nodes = cpu_nodes
        self.cpu_functions = cpu_functions
        self.gpu_nodes = gpu_nodes
        self.gpu_functions = gpu_functions
        self.functions = cpu_functions + gpu_functions
        self.nodes = cpu_nodes + gpu_nodes
        # for i, cpu_f in enumerate(cpu_functions):
        #     cpu_function = cpu_f[:4]
        #     for j, gpu_f in enumerate(gpu_functions):
        #         gpu_function = gpu_f[:4]
        #         if cpu_function == gpu_function:
        #             self.cpu_function_gpu_map[i] = j
        #             break

    def load_node_memory_matrix(self, matrix: List[int]):
        self.node_memory_matrix = np.array(matrix, dtype=int)
        self.cpu_node_memory_matrix = self.node_memory_matrix[:len(self.cpu_nodes)]
        self.gpu_node_memory_matrix = self.node_memory_matrix[len(self.cpu_nodes):]

        # Check input correctness
        assert len(self.node_memory_matrix) == len(self.nodes), \
            f"Actual {len(self.node_memory_matrix.shape)}, Expected {len(self.nodes)}"

    def load_function_memory_matrix(self, matrix: List[int]):
        self.function_memory_matrix = np.array(matrix, dtype=int)
        self.cpu_function_memory_matrix = self.function_memory_matrix[:len(self.cpu_functions)]
        self.gpu_function_memory_matrix = self.function_memory_matrix[len(self.cpu_functions):]

        # Check input correctness
        assert len(self.function_memory_matrix) == len(self.functions), \
            f"Actual {len(self.function_memory_matrix)}, Expected {len(self.functions)}"

    def load_node_delay_matrix(self, matrix: List[List[int]]):
        self.node_delay_matrix = np.matrix(matrix, dtype=int)
        self.cpu_node_delay_matrix = self.node_delay_matrix[:, :len(self.cpu_nodes)]
        self.gpu_node_delay_matrix = self.node_delay_matrix[:, len(self.cpu_nodes):]

        # Check input correctness
        assert self.node_delay_matrix.shape == (len(self.nodes), len(self.nodes)), \
            f"Actual {self.node_delay_matrix.shape}, Expected {(len(self.nodes), len(self.nodes))}"

    def load_workload_matrix(self, matrix: List[List[int]]):
        self.workload_matrix = np.matrix(matrix, dtype=int)
        self.cpu_workload_matrix = self.workload_matrix[:, :len(self.cpu_functions)]
        self.gpu_workload_matrix = self.workload_matrix[:, len(self.cpu_functions):]

        # Check input correctness
        assert self.workload_matrix.shape == (len(self.nodes), len(self.functions)), \
            f"Actual {self.workload_matrix.shape}, Expected {(len(self.nodes), len(self.functions))}"

    def load_max_delay_matrix(self, matrix: List[int]):
        self.max_delay_matrix = np.array(matrix, dtype=int)
        self.cpu_max_delay_matrix = self.max_delay_matrix[:len(self.cpu_functions)]
        self.gpu_max_delay_matrix = self.max_delay_matrix[len(self.cpu_functions):]

        # Check input correctness
        assert len(self.max_delay_matrix) == len(self.functions), \
            f"Actual {len(self.max_delay_matrix)}, Expected {len(self.functions)}"

    def load_response_time_matrix(self, matrix: List[List[int]]):
        self.response_time_matrix = np.matrix(matrix, dtype=int)
        self.cpu_response_time_matrix = self.response_time_matrix[:len(self.cpu_nodes), :len(self.cpu_functions)]
        self.gpu_response_time_matrix = self.response_time_matrix[len(self.cpu_nodes):, len(self.cpu_functions):]

        # Check input correctness
        assert self.response_time_matrix.shape == (len(self.nodes), len(self.functions)), \
            f"Actual {self.response_time_matrix.shape}, Expected {(len(self.nodes), len(self.functions))}"


class Solver:
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    x = {}
    c = {}
    y = {}
    x_jr=[] # Result matrix for allocation of request 'r' in node 'j'
    requests_index=[]
    requests_received=0
    req_distribution=[]
    # moved_from = {}
    # moved_to = {}
    data: Data = None
    

    def haversine(self, lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if verbose:
            self.solver.EnableOutput()
        pass

    def load_input(self, data: Data):
        self.data = data

        ################## MISSING INPUTS ##################

        # users_location: coordinates of users
        users_location = pd.read_csv('users-test.csv') 
        N_src = pd.read_csv('serverstest.csv')

        data.sources = N_src
        data.nodes = N_src
        
        # Amount of request received in time-slot
        for f in range(len(data.functions)):
            for i in range(len(data.sources)):
                data.workload_matrix[f][i]=round(data.workload_matrix[f][i])
        self.requests_received = int(np.sum(data.workload_matrix))

        # Set of requests within coverage of node i
        req_node_coverage = []  
        
        # Identifies which user sent the request [users_location x requests_received]
        req_by_user=[
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                    [0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
                    [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
                    [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]]
        
        # 1 if request r arrives to node i [N x R]
        loc_arrival_r=np.zeros([int(len(data.sources)),int(self.requests_received)])

        # Show which requests are assigned to each function [F x requests_received]
        self.req_distribution = np.zeros([int(len(data.functions)),int(self.requests_received)])

        print("--------NODES_LEN [N]--------------",len(data.sources))
        print("--------REQUESTS [R]--------------",self.requests_received)
        print("--------M_F_LEN [F]--------------",len(data.function_memory_matrix))

        r = 0
        while r<self.requests_received:
            for i in range(len(data.sources)):
                for f in range(len(data.functions)):
                    dif = data.workload_matrix[f][i]
                    while dif >0:
                        self.req_distribution[f][r]=1
                        loc_arrival_r[i][r]=1
                        r=r+1
                        dif = dif-1

        # COVERAGE REQUEST-NODE
        #radius = np.round(np.random.uniform(0.1,0.15,len(S)),3) # in km
        radius = np.full(len(data.sources), 0.03)

        for i in range(len(data.sources)):
            node_latitude = data.sources.iloc[i]['LATITUDE']
            node_longitude = data.sources.iloc[i]['LONGITUDE']
            temp = []
            for r in range(self.requests_received):
                for u in range(len(users_location)):
                    if req_by_user[u][r]==1:
                        request_latitude = users_location.iloc[u]['Latitude']
                        request_longitude = users_location.iloc[u]['Longitude']
                        dist_geo = self.haversine(node_longitude, node_latitude, request_longitude, request_latitude)
                        if dist_geo <= radius[i]:
                            temp.append(1)
                        else:
                            temp.append(0)
            
            req_node_coverage.append(temp)

        # Initialize variable
        self.log("Initializing variables...")
        for j in range(len(data.nodes)):
            for r in range(self.requests_received):
                    self.x[j, r] = self.model.NewBoolVar(f'c[{j}][{r}]')
        for f in range(len(data.functions)):
            for j in range(len(data.nodes)):
                self.c[f, j] = self.model.NewBoolVar(f'c[{f}][{j}]')
        for j in range(len(data.nodes)): 
            self.y[j] = self.model.NewBoolVar(f'c[{j}]')

        # Initialize constraints
        self.log("Initializing constraints...")
        #Controls if request r can be managed by node j
        for r in range(self.requests_received):
            for j in range(len(data.nodes)):
                if req_node_coverage[j][r]==0:
                    self.model.Add(self.x[j, r]==0) 

        #Proximity constraint (node i-node j) 
        for i in range(len(data.sources)):
            for r in range(self.requests_received):
                for f in range(len(data.functions)):
                    for j in range(len(data.nodes)):
                        if data.node_delay_matrix[i][j]> data.max_delay_matrix[f] and loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1:
                            self.model.Add(
                                self.x[j, r]==0
                            )         

        # The sum of the memory of functions deployed on a node `n` is less than its capacity
        for j in range(len(data.nodes)):
            self.model.Add(
                sum([
                    self.c[f, j] * data.function_memory_matrix[f] for f in range(len(data.functions))
                ]) <= data.node_memory_matrix[j]*self.y[j])
        
        # Consider the amount of cores available on a node
        # Do not overload a node
        for j in range(len(data.nodes)):
             self.model.Add(
                 sum([
                     self.x[j, r] * data.core_per_req_matrix[f,j]*self.req_distribution[f][r] for r in range(self.requests_received) for f in range(len(data.functions))
                 ]) <= data.node_cores_matrix[j]*self.y[j])

        # Contraint family (each request can be allocated just once)
        for r in range(self.requests_received):
            self.model.Add(sum([self.x[j, r] for j in range(len(data.nodes))]) <= 1)

        # If a function `f` is deployed on node `n` then c[f,n] is True
        for f in range(len(data.functions)):
            for j in range(len(data.nodes)):
                self.model.Add(
                    sum([
                        self.x[j, r]* self.req_distribution[f][r] for r in range(self.requests_received)
                    ]) <= self.c[f, j] * 1000)
        
        # If request 'r' is allocated to node j then y[j] is 1
        for j in range(len(data.nodes)):
            self.model.Add(
                sum([
                    self.x[j, r] for r in range(self.requests_received)
                ]) <= self.y[j] * 1000) 
        
        # Allocates at least one instance for each function (even with no incoming requests)
        for f in range(len(self.data.functions)):
            self.model.Add(
                sum([self.c[f,j] for j in range(len(self.data.nodes))])>= 1
            )

    def log(self, msg: str):
        if self.verbose:
            print(f"{datetime.datetime.now()}: {msg}")

    def solve(self):
        # Starting to solve
        self.log("Starting solving problem...")

        objective_max = []

        # Objective function
        for j in range(len(self.data.nodes)):
            for r in range(self.requests_received):
                objective_max.append(self.x[j,r])
        self.model.Maximize(sum(objective_max))

        self.solver.Solve(self.model)
        max_requests = self.solver.ObjectiveValue()

        # Hint (speed up solving)
        for j in range(len(self.data.nodes)):
            for r in range(self.requests_received):
                self.model.AddHint(self.x[j,r], self.solver.Value(self.x[j,r]))
            
        # Constraint previous objective
        self.model.Add(
            sum([
                self.x[j, r] for j in range(len(self.data.nodes)) for r in range(self.requests_received)
            ]) == round(self.solver.ObjectiveValue())
        ) 

        # Minimize the number of nodes used
        objective_min = []

        for j in range(len(self.data.nodes)):
            objective_min.append(self.y[j])
        self.model.Minimize(sum(objective_min))

        # Solve problem

        self.x_jr = np.zeros([int(len(self.data.nodes)),int(self.requests_received)])
        status = self.solver.Solve(self.model)
        self.log(f"Problem solved with status {status}")
        # DISPLAY THE SOLUTION-----------------------------------
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print('SOLUTION:')
            print(f'Objective value: {max_requests} requests have been allocated to {self.solver.ObjectiveValue()} nodes\n')
            for r in range(self.requests_received):
                for j in range(len(self.data.nodes)):
                    if int(self.solver.Value(self.x[j,r])) == 1:
                        print(f'x[{j},{r}]: Request {r} has been allocated to node {j}')
                        self.x_jr[j][r]=1
          
            print('----------------------------------------------------------------------')
            for f in range(len(self.data.functions)):
                for j in range(len(self.data.nodes)):
                    if int(self.solver.Value(self.c[f,j])) == 1:
                        print(f'c[{f},{j}]: Function {f} has been deployed on node {j}')

            print('----------------------------------------------------------------------')
            for j in range(len(self.data.functions)):
                if int(self.solver.Value(self.y[j])) == 1:
                        print(f'y[{j}]: Node {j} is used') 

    def results(self) -> Tuple[np.array, np.array]:
        # Fill c matrix
        c_matrix = np.empty(shape=(len(self.data.functions), len(self.data.nodes)))
        for j in range(len(self.data.nodes)):
            for f in range(len(self.data.functions)):
                c_matrix[f][j] = self.solver.Value(self.c[f, j])
        
        print("---------------C_MATRIX [F,N]----------------")
        print(c_matrix)
        
        # Fill x matrix
        mat_mul = np.dot(self.req_distribution,np.transpose(self.x_jr))
        x_matrix = np.empty(shape=(len(self.data.sources),len(self.data.functions),len(self.data.nodes)))
        for i in range(len(self.data.sources)):
            for j in range(len(self.data.nodes)):
                for f in range(len(self.data.functions)):
                    if sum(mat_mul[f])==0:
                        x_matrix[i][f][j] = 0
                    else:
                        x_matrix[i][f][j] = mat_mul[f][j]/sum(mat_mul[f])
                    if self.req_distribution.sum(axis=1)[f]==0:
                        
                        x_matrix[i][f][j]=c_matrix[f][j]/c_matrix.sum(axis=1)[f]

        print("---------------X_MATRIX [FxN]----------------")
        print(x_matrix)

        return x_matrix, c_matrix

    def score(self) -> float:
        return self.solver.ObjectiveValue()