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
from scipy.stats import truncnorm
import rvo2
import matplotlib.cm as cm


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
    num_users=0
    req_by_user=[]
    req_distribution=[]
    # moved_from = {}
    # moved_to = {}
    data: Data = None
    
    transformer = Transformer.from_crs('epsg:3857', 'epsg:4326')

    #Criticality inputs
    CR_matrix = []
    #D =(-37.81952,144.95714099999998)  # Danger source position
    D = (round(random.uniform(-0.010, 0.010),3),round(random.uniform(-0.010, 0.010),3)) 
    D_rad = (0.5) # Influence range of danger source D (radius) in km
    U_per = 0  # Perception range of individual uj in km
    T_1 =0 # Starting point of period
    T_2 = 1 # Ending point of period
    T = T_2-T_1 # Total interval time
    lambd = 0.5 # Severity of the stimulus event
    nej = 0 # Emotional fluctuation of uj ---> nej ∈ (0, 1)
    se_j =0 # Individual sensitivity uj 

    

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

    def rvoAlgorithm(self,user_coords):
        pref_vel = 5/(111.1*3600)
        sim = rvo2.PyRVOSimulator(1,    #float timeStep
                                1.5,    # float MAX neighborDist
                                self.num_users, # size_t maxNeighbors
                                1,    # float timeHorizon --->  tiempo de respuesta frente la presencia de otros agentes
                                2,      # float timeHorizonObst ---> tiempo de respuesta frente a presencia de obstaculos
                                1,      # float radius ---> of agents
                                pref_vel)      # float maxSpeed ---> of agents
                                    # tuple velocity=(0, 0)
    
        agents =[]
        for u in range(self.num_users):
            user_latitude = user_coords[u,0]
            user_longitude = user_coords[u,1]
            agents.append(sim.addAgent((user_latitude,user_longitude)))

    
        circle_area = [self.D[0],self.D[1],self.D_rad/111.139]
        circle=[self.D[0],self.D[1],0.05/111.1]
        xc = circle[0] #x-co of circle (center) 
        xc_area=circle_area[0]
        yc = circle[1] #y-co of circle (center) 
        yc_area= circle_area[1]
        r=circle[2] 
        r_area=circle_area[2]
        t= np.arange(-np.pi,np.pi, 0.1) 
        x_circle= xc + r*np.sin(t) 
        x_circle_area = xc_area+r_area*np.sin(t)
        y_circle= yc + r*np.cos(t) 
        y_circle_area= yc_area + r_area*np.cos(t) 

        # Obstacles are also supported.
        o1 = sim.addObstacle([(xc+r,yc+r), (xc+r,yc-r), (xc-r,yc-r),(xc-r,yc+r)])
        sim.processObstacles()

    
        for a in range(len(agents)):
            #sim.setAgentPrefVelocity(agents[a], (-2,2))
            #sim.setAgentPrefVelocity(agents[a], (random.randint(-10, 10),random.randint(-10, 10)))
            sim.setAgentPrefVelocity(agents[a], (random.uniform(-pref_vel, pref_vel),random.uniform(-pref_vel, pref_vel)))

        #print('Simulation has %i agents and %i obstacle vertices in it.' %
                #(sim.getNumAgents(), sim.getNumObstacleVertices()))
        #print('Running simulation')
    
        positions_T=[]
        fig, ax = plt.subplots(figsize = (10,5))

        for step in range(self.T):
            sim.doStep()

            positions = ['(%5.8f, %5.8f)' % sim.getAgentPosition(agent_no) for agent_no in (agents)]
            velocities = ['(%3.8f, %3.8f)' % sim.getAgentVelocity(agent_no)for agent_no in (agents)]
        
            #print('%s' % ( '  '.join(velocities)))
            #print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))
        
            x_p=[]
            y_p=[]
            temp_T=[]

            colors = cm.rainbow(np.linspace(0, 1, len(positions)))
        
            for p in range(len(positions)):
                temp=positions[p]
                temp= temp.replace('(','')
                temp=temp.replace(')','')
                temp=temp.split(',')
                x_p.append(float(temp[0]))
                y_p.append(float(temp[1]))
                temp_T.append([float(temp[0]),float(temp[1])])

                if step==0:
                    ax.scatter(float(temp[0]),float(temp[1]),color=colors[p],edgecolor='green', linewidth=2)

                if step==self.T-1:
                    ax.scatter(float(temp[0]),float(temp[1]),color=colors[p],edgecolor='red', linewidth=2)
        
                else:
                    ax.scatter(float(temp[0]),float(temp[1]),edgecolors=colors[p],color='white',marker='o')

            positions_T.append(temp_T)
        
        ax.plot(x_circle,y_circle,color='red') 
        ax.plot(x_circle_area,y_circle_area,color='red',alpha=0.3) 
        ax.plot([xc+r,xc+r,xc-r,xc-r,xc+r],[yc+r,yc-r,yc-r,yc+r,yc+r],color='black',linewidth=2)
        ax.plot()
        ax.axis('equal')
        ax.set_title('Location')
        ax.ticklabel_format(useOffset=False)

        return positions_T
    
    def du_dt_function(self,time,positions):
        du_Dt=[] # Distance between danger and user
        for j in range(self.num_users):
            user_latitude=positions[time][j][0]
            user_longitude=positions[time][j][1]
            user_coordinates = (user_latitude,user_longitude)
            dist_geoDanger = geopy.distance.geodesic(user_coordinates, self.D).km
            du_Dt.append(dist_geoDanger)
        
        # for r in range(self.requests_received):
        #     for u in range(self.num_users):
        #         if self.req_by_user[u][r]==1:
        #             du_Dt_requests[r]=du_Dt[u]
        return du_Dt

    def get_truncated_normal(self,mean, sd, low, upp):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    
    def criticality (self,time,du_Dt):
        CR = []
        CR_requests = np.zeros(self.requests_received)
        
        # emj: emotional value
        em_t = np.zeros(self.num_users)
        for j in range(self.num_users):
            if du_Dt[j] < (self.D_rad+self.U_per[j]):
                eq = ((time-self.T_1)/self.T) * (1-(du_Dt[j]/(self.D_rad+self.U_per[j])))*self.lambd* self.nej[j]
                em_t[j]=eq
            else:
                em_t[j]=0

        # scrj(t) = sej + emj (t) (subjective criticality equation)
        scr=np.zeros(self.num_users)
        for j in range(self.num_users):
            scr[j]= (self.se_j[j]+em_t[j])

        ocr_j=np.zeros(self.num_users) # ocr objective criticality of individual uj at time step t
        nDiv = 5 # Number of partitions of the coverage area
        circles = self.D_rad/nDiv # Radius of each circular partition
        covCircles = [] # Distance from the center
        ri = [0.5,0.4,0.3,0.2,0.1]

        for p in range(1,nDiv+1):
            covCircles.append(circles*p)

        for j in range(self.num_users):
                if du_Dt[j]<covCircles[0]:
                    ocr_j[j] = ri[0]

                elif du_Dt[j]<covCircles[1] and du_Dt[j]>=covCircles[0]:
                    ocr_j[j]  =ri[1]

                elif du_Dt[j]<covCircles[2] and du_Dt[j]>=covCircles[1]:
                    ocr_j[j]  = ri[2]

                elif du_Dt[j]<covCircles[3] and du_Dt[j]>=covCircles[2]:
                    ocr_j[j] = ri[3]
                elif du_Dt[j]<covCircles[4] and du_Dt[j]>=covCircles[3]:
                    ocr_j[j]  = ri[4]

                else:
                    ocr_j[j]  = 0


                # Criticality equation
                # CRj = (L1 * scrj(t)) + (L2 * ocrj)
                mu=1
                sigma=0.05
                low=0.95
                up =1.05

                X = self.get_truncated_normal(mu, sigma, low, up)

                #L1 =X.rvs() # weight lambda 1
                L1 = 0.95
                L2 = 1 # weight lambda 2

                # Criticality            
                cr = (L1*scr[j])+(L2*ocr_j[j])
                CR.append(cr)

        for r in range(self.requests_received):
            for u in range(self.num_users):
                if self.req_by_user[u][r]==1:
                    CR_requests[r]=CR[u]
        
        #index_CR=np.argsort(CR_requests, kind='stable')

        return CR_requests

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if verbose:
            self.solver.EnableOutput()
        pass

    def load_input(self, data: Data):
        self.data = data

        ################## MISSING INPUTS ##################

        # # users_location: coordinates of users
        # users_location = pd.read_csv('users-test.csv') 
        # N_src = pd.read_csv('serverstest_3.csv')

        ################## MISSING INPUTS ##################
        # EXAMPLE USAGE

        max_delay = 2000 # millis
        num_nodes = data.sources # random.randint(3, 20) # gen some nodes
        self.num_users = 8 # random.randint(num_nodes*2, num_nodes*50) # gen some users

        # creates a diagonal matrix of delays (in Neptune this is given)
        b = np.random.randint(0, max_delay, size=(len(num_nodes),len(num_nodes)))
        delay_matrix = (b + b.T)/2
        np.fill_diagonal(delay_matrix, 0)

        node_coords = self.delay_to_geo(delay_matrix)
        radius = self.get_radius(node_coords)
        # print("Radius used in KM: ", radius[0])
        user_coords = self.place_users_close_to_nodes(self.num_users, node_coords)
        self.plot(node_coords, user_coords)

        # data.sources = num_nodes
        # data.nodes = num_nodes
        
        # Amount of request received in time-slot
        for f in range(len(data.functions)):
            for i in range(len(data.sources)):
                data.workload_matrix[f][i]=round(data.workload_matrix[f][i])
        self.requests_received = int(np.sum(data.workload_matrix))

        print("--------SOURCES_LEN [N]--------------",data.sources)
        print("--------NODES_LEN [N]--------------",data.nodes)
        print("--------REQUESTS [R]---------------",self.requests_received)
        print("--------M_F_LEN [F]---------------",len(data.function_memory_matrix))
        #print("--------WORKLOAD [R]---------------",data.workload_matrix)
        
        # Set of requests within coverage of node i
        req_node_coverage = []  
        
        # Identifies which user sent the request [users_location x requests_received]
        matrix_size = (self.num_users, self.requests_received)
        self.req_by_user = np.zeros(matrix_size)
        row_indices = np.random.randint(0, matrix_size[0], matrix_size[1])
        self.req_by_user[row_indices, np.arange(matrix_size[1])] = 1
        
        # 1 if request r arrives to node i [N x R]
        loc_arrival_r=np.zeros([int(len(data.sources)),int(self.requests_received)])

        # Show which requests are assigned to each function [F x requests_received]
        self.req_distribution = np.zeros([int(len(data.functions)),int(self.requests_received)])


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

        #Obtain criticality
        self.U_per = np.full(self.num_users,0.2)  # Perception range of individual uj in km
        self.nej = np.random.uniform(0,1,self.num_users) # Emotional fluctuation of uj ---> nej ∈ (0, 1)
        self.se_j =np.random.uniform(0.05,0.1,self.num_users) # Individual sensitivity uj 

        self.D = (node_coords[0,0],node_coords[0,1])
        live_positions = self.rvoAlgorithm(user_coords)
        du_dt_temp=self.du_dt_function(0,live_positions)
        self.CR_matrix = self.criticality(0,du_dt_temp)
        print("-------Criticality---------",self.CR_matrix)
        
        live_positions_requests =[]
        for r in range(self.requests_received):
            for u in range(self.num_users):
                if self.req_by_user[u][r]==1:
                    live_positions_requests.append(live_positions[0][u])
        # COVERAGE REQUEST-NODE
        #radius = np.round(np.random.uniform(0.1,0.15,len(S)),3) # in km
        # radius = np.full(data.sources, 0.03)

        for i in range(len(data.sources)):
            node_latitude = node_coords[i,0]
            node_longitude = node_coords[i,1]
            temp = []
            for r in range(self.requests_received):
                for u in range(self.num_users):
                    if self.req_by_user[u][r]==1:
                        request_latitude = live_positions_requests[r][0]
                        request_longitude = live_positions_requests[r][1]
                        dist_geo = self.haversine(node_longitude, node_latitude, request_longitude, request_latitude)
                        if dist_geo <= radius[0]:
                            temp.append(1)
                        else:
                            temp.append(0)
            
            req_node_coverage.append(temp)
        print("-----------position------------------",req_node_coverage)
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
                        if int(data.node_delay_matrix[i][j])> int(data.max_delay_matrix[f]) and loc_arrival_r[i][r]==1 and self.req_distribution[f][r]==1:
                            self.model.Add(
                                self.x[j, r]==0
                            )         

        # The sum of the memory of functions deployed on a node `n` is less than its capacity
        for j in range(len(data.nodes)):
            suma_constraint = sum([self.c[f, j] * int(data.function_memory_matrix[f]) for f in range(len(data.functions))])
            self.model.Add(suma_constraint <= int(data.node_memory_matrix[j])*self.y[j])

        # Consider the amount of cores available on a node
        # Do not overload a node
        for j in range(len(data.nodes)):
             self.model.Add(
                 sum([
                     self.x[j, r] * int(data.core_per_req_matrix[f,j])*int(self.req_distribution[f][r]) for r in range(self.requests_received) for f in range(len(data.functions))
                 ]) <= int(data.node_cores_matrix[j])*self.y[j])

        # Contraint family (each request can be allocated just once)
        for r in range(self.requests_received):
            self.model.Add(sum([self.x[j, r] for j in range(len(data.nodes))]) <= 1)

        # If a function `f` is deployed on node `n` then c[f,n] is True
        for f in range(len(data.functions)):
            for j in range(len(data.nodes)):
                self.model.Add(
                    sum([
                        self.x[j, r]* int(self.req_distribution[f][r]) for r in range(self.requests_received)
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
                objective_max.append(self.x[j,r]*self.CR_matrix[r])
        self.model.Maximize(sum(objective_max))

        self.solver.Solve(self.model)
        max_requests = self.solver.ObjectiveValue()

        # Hint (speed up solving)
        for j in range(len(self.data.nodes)):
            for r in range(self.requests_received):
                self.model.AddHint(self.x[j,r], self.solver.Value(self.x[j,r]))
        
        for r in range(self.requests_received):
            self.model.Add(sum([self.x[j, r] for j in range(len(self.data.nodes))]) == sum([self.solver.Value(self.x[j,r]) for j in range(len(self.data.nodes))]))

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
            for j in range(len(self.data.nodes)):
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
        print(x_matrix[0])

        return x_matrix, c_matrix

    def score(self) -> float:
        return self.solver.ObjectiveValue()