import datetime
import itertools
from typing import List, Tuple

import numpy as np
from ortools.linear_solver import pywraplp

import pandas as pd
from math import radians, cos, sin, asin, sqrt, isnan
import random
from ortools.sat.python import cp_model

from sklearn import manifold
from matplotlib import pyplot as plt
from pyproj import Transformer
import geopy.distance
import math

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
    #Decision variables:
    x_jr = [] # 1 when request r allocated to node j
    c_fj = [] # 1 when function instance f deployed in node j
    y_j = []  # 1 if node j is being used
    
    #Other variables
    S_active = [] #1 when function instance f is active in node j
    requests_index=[]
    requests_received=0
    req_distribution=[]
    req_node_coverage = [] # Set of requests within coverage of node i
    loc_arrival_r=[]
    data: Data = None
    # moved_from = {}
    # moved_to = {}

    transformer = Transformer.from_crs('epsg:3857', 'epsg:4326')

    def cartesian_to_geo(self, x, y):
        return self.transformer.transform(x, y)

    def to_km_distance(self, coords):
        size = len(coords) # No of nodes
        distances = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                distances[i, j] = geopy.distance.geodesic(coords[i], coords[j]).km # ??
        return distances
    
    def km_to_deg(self, d):
        earth_radius = 6371
        return 180 * d / (np.pi * earth_radius)

    def delay_to_geo(self, delay_matrix): # Establish the coordinates of each server according to the delay matrix
        size = len(delay_matrix) # No of nodes
        mds_model = manifold.MDS(n_components=2, random_state=0, dissimilarity='precomputed', normalized_stress="auto") # MDS(dissimilarity='precomputed', normalized_stress='auto', random_state=0)
        mds_model.fit(delay_matrix)
        coords = mds_model.fit_transform(delay_matrix) # [[-711.64291427 -379.86399722][..][..]]
        for i in range(size):
            coords[i] = self.cartesian_to_geo(*coords[i]) # [[-0.0034123763 -0.0063927971][..][..]]
        return coords
    
    def get_radius(self, coords, scale_factor = 0.9): # The greater the scale_factor the higher the intersection between the nodes
        size = len(coords) # No of nodes
        node_rad_scale_factor = math.sqrt(math.sqrt(size)) / scale_factor 
        distances = self.to_km_distance(coords)
        distances = np.tril(distances, k=0) # distance in km between the nodes
        average_distance = np.sum(distances)/((size*size)/2-size/2)
        km_radius = average_distance/node_rad_scale_factor
        deg_radius = self.km_to_deg(km_radius)
        return km_radius, deg_radius

    def place_users_close_to_nodes(self, users: int, node_coords):
        size = len(node_coords)
        user_coords = []
        radius = self.get_radius(node_coords)[1]
        for i in range(users):
            node_coord = node_coords[random.choice(range(size))]
            user_coord = []
            alpha = 2 * math.pi * random.random()
            r = radius * random.random()
            user_coord.append(node_coord[0] + r*math.cos(alpha))
            user_coord.append(node_coord[1] + r*math.sin(alpha))
            user_coords.append(user_coord)
        return np.array(user_coords)

    def plot(self, node_coords, user_coords):
        size = len(node_coords)
        colors = ['r'] * size
        sizes = [100] * size
        ax = plt.gca()

        ax.axis("equal")
        radius = self.get_radius(node_coords)[1]
        plt.scatter(user_coords[:,0], user_coords[:,1])

        for coord in node_coords:
            cir = plt.Circle(coord, radius, color='r', fill=False, linewidth=2)
            ax.add_patch(cir)
        
        plt.scatter(node_coords[:,0], node_coords[:,1], c=colors, s=sizes, edgecolor='black')

        plt.show()
      
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
        #users_location = pd.read_csv('users-test.csv') 
        #N_src = pd.read_csv('serverstest.csv')

        ################## MISSING INPUTS ##################
        # EXAMPLE USAGE
        max_delay = 2000 # millis
        num_nodes = data.sources # random.randint(3, 20) # gen some nodes
        num_users = 8 # random.randint(num_nodes*2, num_nodes*50) # gen some users

        # creates a diagonal matrix of delays (in Neptune this is given)
        b = np.random.randint(0, max_delay, size=(len(num_nodes),len(num_nodes)))
        delay_matrix = (b + b.T)/2
        np.fill_diagonal(delay_matrix, 0)

        # print("####### DELAY_MATRIX ######")
        # print(delay_matrix)
        node_coords = self.delay_to_geo(delay_matrix)
        radius = self.get_radius(node_coords)
        #print("Radius used in KM: ", radius[0], "equivalent to degrees:",  radius[1])
        user_coords = self.place_users_close_to_nodes(num_users, node_coords)
        self.plot(node_coords, user_coords)

        #data.sources = num_nodes
        #data.nodes = num_nodes

        # Amount of request received in time-slot
        for f in range(len(data.functions)):
            for i in range(len(data.sources)):
                data.workload_matrix[f][i]=round(data.workload_matrix[f][i])
        self.requests_received = int(np.sum(data.workload_matrix)) 
        
        print("--------SOURCES_LEN [N]--------------",len(data.sources))
        print("--------NODES_LEN [N]--------------",len(data.nodes))
        print("--------REQUESTS [R]---------------",self.requests_received)
        print("--------M_F_LEN [F]---------------",len(data.function_memory_matrix))
        #print("--------U_rj---------------------", data.function_memory_matrix)
        # Identifies which user sent the request [users_location x requests_received]
        matrix_size = (num_users, self.requests_received)
        req_by_user = np.zeros(matrix_size)
        row_indices = np.random.randint(0, matrix_size[0], matrix_size[1])
        req_by_user[row_indices, np.arange(matrix_size[1])] = 1

        # 1 if request r arrives to node i [N x R]
        self.loc_arrival_r=np.zeros([int(len(data.sources)),int(self.requests_received)])

        # Show which requests are assigned to each function [F x requests_received]
        self.req_distribution = np.zeros([int(len(data.functions)),int(self.requests_received)])

        r = 0
        while r<self.requests_received:
            for i in range(len(data.sources)):
                for f in range(len(data.functions)):
                    dif = data.workload_matrix[f][i]
                    while dif >0:
                        self.req_distribution[f][r]=1
                        self.loc_arrival_r[i][r]=1
                        r=r+1
                        dif = dif-1

        # Matrix that assignes a function memory to each request [F x N]
        m_request = np.empty((len(data.functions),int(self.requests_received)))
        for f in range(len(data.functions)):
            for r in range (self.requests_received):
                m_request[f][r] = data.function_memory_matrix[f]*self.req_distribution[f][r]

        # Sort the requests by their memory requirement --- returns position of the [] where request is found
        m_index = []

        for r in range (self.requests_received):
            for f in range (len(data.functions)):
                if m_request[f][r]!=0:
                    m_index.append(m_request[f][r])
       
        #if all(element == m_index[0] for element in m_index)==True:
         #   self.requests_index=[item for item in range(len(m_index))]
        #else:
        self.requests_index=np.argsort(m_index,kind='stable')

        #print("------------INDEX---------------",self.requests_index)
        # COVERAGE REQUEST-NODE
        #radius = np.round(np.random.uniform(0.1,0.15,len(S)),3) # in km
        #radius = np.full(len(data.sources), 0.03)
        for i in range(len(data.sources)):
            node_latitude = node_coords[i,0]
            node_longitude = node_coords[i,1]
            temp = []
            for r in range(self.requests_received):
                for u in range(num_users):
                    if req_by_user[u][r]==1:
                        request_latitude = user_coords[u,0]
                        request_longitude = user_coords[u,1]
                        dist_geo = self.haversine(node_longitude, node_latitude, request_longitude, request_latitude)
                        if dist_geo <= radius[0]:
                            temp.append(1)
                        else:
                            temp.append(0)
            
            self.req_node_coverage.append(temp)
        
        # Initialize variable
        self.x_jr = np.zeros(shape=(int(len(data.nodes)),int(self.requests_received)))
        self.c_fj = np.zeros(shape=(len(data.functions),int(len(data.nodes))))
        self.y_j = np.zeros(int(len(data.nodes)))
        self.S_active = np.zeros(shape=(len(data.functions),int(len(data.nodes))))
        

    def log(self, msg: str):
        if self.verbose:
            print(f"{datetime.datetime.now()}: {msg}")

    def solve(self):
        # Starting to solve
        self.log("Starting solving problem...")
        temp_req_index=0
        index_distribution = np.zeros(len(self.data.nodes))
        for j in range(len(self.data.node_cores_matrix)):
            index_distribution[j] = self.data.node_cores_matrix[j]
        
        while temp_req_index in self.requests_index:
            r = self.requests_index[temp_req_index]
            index_j = np.argsort(-index_distribution)#[::-1]
            loc=0
            print("---------------------REQUEST:  ",r," ----------------------------------")
            print("ORDER CHECK NODES:  ", index_j)
            print("CORE NODES:", self.data.node_cores_matrix)
            for j in index_j:
                for f in range(len(self.data.functions)):
                    if all(self.S_active[f,:]==0) and loc==0 and self.req_distribution[f][r]:
                        #print(f" **** NO CONTAINERS FOR FUNCTION {f} **** ") 
                        #Option 1 there is no container for function f and there are active nodes we can check
                        active_j=np.where(self.y_j==1)[0][:]  
                        ordered_active_j = [x for x in index_j if x  in active_j]
                        #ordered_active_j= np.where(self.y_j==1)[0][:]  
                        #print("Active nodes:  ", ordered_active_j)
                        for j_temp in ordered_active_j:
                            if self.req_node_coverage[j_temp][r]==1 and loc==0:
                                #print("Active nodes:  ", ordered_active_j)
                                #print("NODE:  ",j_temp," FUNCTION: ", f)
                                #print("✓ proximity constraint ")
                                if sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f]<=self.data.node_memory_matrix[j_temp]: #memory constraint
                                    #print("✓ memory constraint ")
                                    #print("SUM MEMORY: ", sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f])
                                    if sum(self.x_jr[j_temp,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp]*self.req_distribution[f][r]<= self.data.node_cores_matrix[j_temp]: #core constraint
                                        #print("✓ core constraint ")
                                        #print("CORE REQ: ",sum(self.x_jr[j_temp,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp]*self.req_distribution[f][r] )
                                        for i in range(len(self.data.sources)):
                                            if self.data.node_delay_matrix[i,j_temp]<self.data.max_delay_matrix[f] and self.loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1: #delay constraint
                                                #print("✓ delay constraint, arrived to node: ", i)
                                                loc=1
                                                self.x_jr[j_temp][r]=1
                                                self.S_active[f][j_temp]=1
                                                self.y_j[j_temp]=1
                                                self.c_fj[f][j_temp]=1
                                                index_distribution[j_temp]=index_distribution[j_temp]-self.data.core_per_req_matrix[f,j_temp]*self.req_distribution[f,r]
                                                #print("DEPLOY CONTAINER IN NODE: ",j_temp," function: ", f)
                                                #print("||||||||||||||||||||| OPTION 1 |||||||||||||||||||||")
                                                break
                        #Option 2: there is no container for function f and no active nodes (all y_j==0)
                        if j not in ordered_active_j and loc==0:
                            #print("NODE:  ",j," FUNCTION: ", f)
                            if self.req_node_coverage[j][r]==1:
                                #print("✓ proximity constraint ")
                                if (sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f])<=(self.data.node_memory_matrix[j]): #memory constraint
                                    #print("✓ memory constraint ")
                                    #print("SUM MEMORY: ", (sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j] for f_temp in range(len(self.data.functions))))+self.data.function_memory_matrix[f])
                                    if (sum(self.x_jr[j,r_temp]*self.data.core_per_req_matrix[f_temp,j]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j]*self.req_distribution[f][r])<=(self.data.node_cores_matrix[j]) : #core constraint
                                        #print("✓ core constraint ")
                                        #print("CORE REQ: ",sum(self.x_jr[j,r_temp]*self.data.core_per_req_matrix[f_temp,j]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j]*self.req_distribution[f][r])
                                        for i in range(len(self.data.sources)):
                                            if self.data.node_delay_matrix[i,j]<self.data.max_delay_matrix[f] and self.loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1: #delay constraint
                                                #print("✓ delay constraint, arrived to node: ", i)
                                                loc=1
                                                self.x_jr[j][r]=1
                                                self.S_active[f][j]=1
                                                self.y_j[j]=1
                                                index_distribution[j]=index_distribution[j]-self.data.core_per_req_matrix[f,j]*self.req_distribution[f,r]
                                                self.c_fj[f][j]=1
                                                #print("ACTIVATE NODE: ",j," for function: ", f)
                                                #print("||||||||||||||||||||| OPTION 2 |||||||||||||||||||||")
                                                break
                    if any(self.S_active[f,:]==1) and loc==0 and self.req_distribution[f][r]==1: 
                        #Option 3: there is already a container for function f in a node, so it checks if request can be allocated to this node
                        active_loc_f=np.where(self.S_active[f,:]==1)[0][:]
                        ordered_active_loc_f = [x for x in index_j if x  in active_loc_f]
                        #ordered_active_loc_f=np.where(self.S_active[f,:]==1)[0][:]
                        #print(f"Active nodes with container {f}:  ", ordered_active_loc_f)
                        for j_temp_active in ordered_active_loc_f:
                            if self.req_node_coverage[j_temp_active][r]==1 and loc==0: #Proximity constraint:
                                #print(f"THERE IS AN ACTIVE CONTAINER OF TYPE {f} IN NODE {j_temp_active} THAT CAN BE USED")
                                #print("NODE:  ",j_temp_active," FUNCTION: ", f)
                                #print("✓ proximity constraint ")
                                if sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp_active] for f_temp in range(len(self.data.functions)))<=self.data.node_memory_matrix[j_temp_active]: #memory constraint
                                    #print("✓ memory constraint ")
                                    #print("SUM MEMORY: ", sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp_active] for f_temp in range(len(self.data.functions))))
                                    if sum(self.x_jr[j_temp_active,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp_active]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp_active]*self.req_distribution[f][r]<= self.data.node_cores_matrix[j_temp_active]: #core constraint
                                        #print("✓ core constraint ")
                                        #print("CORE REQ: ",sum(self.x_jr[j_temp_active,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp_active]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp_active]*self.req_distribution[f][r])
                                        for i in range(len(self.data.sources)):
                                            if self.data.node_delay_matrix[i,j_temp_active]<self.data.max_delay_matrix[f] and self.loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1: #delay constraint
                                                #print("✓ delay constraint, arrived to node: ", i)
                                                loc=1
                                                self.x_jr[j_temp_active][r]=1
                                                index_distribution[j_temp_active]=index_distribution[j_temp_active]-self.data.core_per_req_matrix[f,j_temp_active]*self.req_distribution[f,r]
                                                #print("USES NODE: ",j_temp_active," function: ", f)
                                                #print("||||||||||||||||||||| OPTION 3 |||||||||||||||||||||")
                                                break
                        #Option 4: Needs to deploy a container for function f in one of the active nodes
                        if self.req_node_coverage[j][r]==1 and (j not in ordered_active_loc_f) and loc==0 and any(self.y_j==1): #Proximity constraint:
                            active_j_f=np.where(self.y_j==1)[0][:]  
                            ordered_active_j_f = [x for x in index_j if x  in active_j_f]
                            #ordered_active_j_f= np.where(self.y_j==1)[0][:]  
                            #print("Active nodes:  ", ordered_active_j_f)
                            for j_temp_f in ordered_active_j_f:
                                if self.req_node_coverage[j_temp_f][r]==1 and loc==0:
                                    #print("Node ",j_temp_f, "is active and can deploy funcion ", f)
                                    #print("NODE:  ",j_temp_f," FUNCTION: ", f)
                                    #print("✓ proximity constraint ")
                                    if sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp_f] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f]<=self.data.node_memory_matrix[j_temp_f]: #memory constraint
                                        #print("✓ memory constraint ")
                                        #print("SUM MEMORY: ", sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j_temp_f] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f])
                                        if sum(self.x_jr[j_temp_f,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp_f]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp_f]*self.req_distribution[f][r]<= self.data.node_cores_matrix[j_temp_f]: #core constraint
                                            #print("✓ core constraint ")
                                            #print("CORE REQ: ",sum(self.x_jr[j_temp_f,r_temp]*self.data.core_per_req_matrix[f_temp,j_temp_f]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j_temp_f]*self.req_distribution[f][r] )
                                            for i in range(len(self.data.sources)):
                                                if self.data.node_delay_matrix[i,j_temp_f]<self.data.max_delay_matrix[f] and self.loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1: #delay constraint
                                                    #print("✓ delay constraint, arrived to node: ", i)
                                                    loc=1
                                                    self.x_jr[j_temp_f][r]=1
                                                    self.S_active[f][j_temp_f]=1
                                                    self.y_j[j_temp_f]=1
                                                    self.c_fj[f][j_temp_f]=1
                                                    index_distribution[j_temp_f]=index_distribution[j_temp_f]-self.data.core_per_req_matrix[f,j_temp_f]*self.req_distribution[f,r]
                                                    #print("DEPLOY CONTAINER IN NODE: ",j_temp_f," function: ", f)
                                                    #print("||||||||||||||||||||| OPTION 4 |||||||||||||||||||||")
                                                    break
                            #Option 5: when need to turn on a new node for a given function f
                            if j not in ordered_active_j_f and loc==0:
                                if self.req_node_coverage[j][r]==1:
                                    #print("No active nodes that can serve function ", f)
                                    #print("NODE:  ",j," FUNCTION: ", f)
                                    #print("✓ proximity constraint ")
                                    if sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f]<=(self.data.node_memory_matrix[j]): #memory constraint
                                        #print("✓ memory constraint ")
                                        #print("SUM MEMORY: ", sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,j] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f])
                                        if sum(self.x_jr[j,r_temp]*self.data.core_per_req_matrix[f_temp,j]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j]*self.req_distribution[f][r]<= (self.data.node_cores_matrix[j]): #core constraint
                                            #print("✓ core constraint ")
                                            #print("CORE REQ: ",sum(self.x_jr[j,r_temp]*self.data.core_per_req_matrix[f_temp,j]*self.req_distribution[f_temp,r_temp] for f_temp in range(len(self.data.functions)) for r_temp in self.requests_index)+self.data.core_per_req_matrix[f,j]*self.req_distribution[f][r] )
                                            for i in range(len(self.data.sources)):
                                                if self.data.node_delay_matrix[i,j]<self.data.max_delay_matrix[f] and self.loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1: #delay constraint
                                                    #print("✓ delay constraint, arrived to node: ", i)
                                                    loc=1
                                                    self.x_jr[j][r]=1
                                                    self.S_active[f][j]=1
                                                    self.y_j[j]=1
                                                    index_distribution[j]=index_distribution[j]-self.data.core_per_req_matrix[f,j]*self.req_distribution[f,r]
                                                    self.c_fj[f][j]=1
                                                    #print("ENTRO EXISTING NODE j:",j," and function: ",f, "for request: ",r)
                                                    #print("ACTIVATE CONTAINER NODE: ",j," function: ", f)
                                                    #print("||||||||||||||||||||| OPTION 5 |||||||||||||||||||||")
                                                    break

            print("Core capacity: ", index_distribution)
            print("Active nodes: " ,self.y_j)
            #print("Coontainer allocation: ")
            #print(self.c_fj)
            temp_req_index=temp_req_index+1
        
        for f in range (len(self.data.functions)):
            if all(self.c_fj[f,:]==0) and any(self.y_j==1): #use nodes that are already being used
                #print(" **ENTRO CONDICION VACIO")
                rand_node= np.where(self.y_j==1)[0][:]  
                for t in rand_node:
                    if sum(self.data.function_memory_matrix[f_temp]*self.c_fj[f_temp,t] for f_temp in range(len(self.data.functions)))+self.data.function_memory_matrix[f]<=(self.data.node_memory_matrix[t]*self.y_j[t]):
                        self.c_fj[f][rand_node]=1
                        self.S_active[f][rand_node]=1
                        self.y_j[rand_node]=1
    
            if all(self.c_fj[f,:]==0): #when there are no requests
                #print(" **ENTRO CONDICION VACIO 2")
                rand_node= np.random.choice(range(len(self.data.sources)))
                self.c_fj[f][rand_node]=1
                self.S_active[f][rand_node]=1
                self.y_j[rand_node]=1

    def results(self) -> Tuple[np.array, np.array]:
        # Fill c matrix
        c_matrix = np.empty(shape=(len(self.data.functions), len(self.data.nodes)))
        for j in range(len(self.data.nodes)):
            for f in range(len(self.data.functions)):
                c_matrix[f][j] = self.c_fj[f][j]
        
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
        print(x_matrix[0])

        return x_matrix, c_matrix

    def score(self) -> float:
        #return self.solver.ObjectiveValue()
        return self.req_distribution