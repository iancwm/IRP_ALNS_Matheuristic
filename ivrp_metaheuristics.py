'''
Description: IVRP State
Author: Group 5
Date: 2022-03-15
LastEditors: Ian Chong
LastEditTime: 2022-03-20 23:45
'''

import copy
import numpy as np
import random
import os

from scipy.spatial.distance import cdist
from pathlib import Path
from typing import List
import sys
from docplex.mp.model import Model


sys.path.append('./ALNS')
from alns import ALNS, State

### Parser to parse instance xml file ###
class Parser(object):
    
    def __init__(self, dat_file):
        '''initialize the parser
        Args:
            dat_file::str
                the path to the .dat file
        '''
        self.name = os.path.split(dat_file)[1]
        
        self.datContent = [i.strip().split() for i in open(dat_file).readlines()]
        
        self.nNodes = int(self.datContent[0][0])
        self.nPeriods = int(self.datContent[0][1])
        self.nVehicles = int(self.datContent[0][3])
        self.customers=[]
        self.vehicles=[]
        self.depot=None
        self.set_depot()
        self.set_customers()        
        self.set_vehicles()
    
    def set_depot(self):
        # Initialize depot
        self.depot = Depot(int(self.datContent[1][0]), 0, float(self.datContent[1][1]), float(self.datContent[1][2]), 
                           int(self.datContent[1][3]), float(self.datContent[1][5]), int(self.datContent[1][4]))
    
    def set_customers(self):
        # Initialize customers    
        for i in range(len(self.datContent)):
            if i > 1:                
                self.customers.append(Customer(int(self.datContent[i][0]), 1, float(self.datContent[i][1]), float(self.datContent[i][2]), 
                                               int(self.datContent[i][3]), float(self.datContent[i][7]), int(self.datContent[i][6]),
                                               int(self.datContent[i][4]), int(self.datContent[i][5])))

    def set_vehicles(self):
        # Initialize Vehicle
        for t in range(self.nPeriods):
            vehicles_temp = []
            for i in range(int(self.datContent[0][3])):
                vehicles_temp.append(Vehicle(i, self.depot, self.depot, int(self.datContent[0][2])))
            self.vehicles.append(vehicles_temp)
        

### Node class ###
class Node(object):

    def __init__(self, id:int, type:int, x:float, y:float, i:float, h:float, r:float):
        '''Initialize a node
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
            i::float
                starting inventory level
            h::float
                inventory cost
            r::float
                Daily production/consumption
        '''
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.i = i
        self.h = h
        self.r = r

    def get_nearest_node(self, nodes):
        '''Find the nearest node in the list of nodes
        Args:
            nodes::[Node]
                a list of nodes
        Returns:
            node::Node
                the nearest node found
        '''
        dis = [cdist([[self.x, self.y]], [[node.x, node.y]], 'euclidean')
               for node in nodes]
        idx = np.argmin(dis)
        return nodes[idx]

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id and self.type == other.type and self.x == other.x and self.y == other.y
        return False

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}'.format(self.id, self.type, self.x, self.y)

    
### Depot class ###
class Depot(Node):

    def __init__(self, id:int, type:int, x:float, y:float, i:int, h:float, r:int):
        '''Initialize a depot
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
            i::int
                starting inventory level
            h::float
                inventory cost
            r::int
                Daily production
        '''
        super(Depot, self).__init__(id, type, x, y, i, h, r)
        
        # Initialize current inventory level
        self.current_inventory=i
        
        # Keep track of inventory costs
        self.inventory_cost=0
    
    def load(self,vehicle,load_policy:str='max',exact_load:float=None):
        '''Load a vehicle
        Args:
            vehicle::Vehicle
                Vehicle to load inventory
            load_policy::str ('max','exact','none')
                Determines how much to load
            exact_load::float
                If load_policy is 'exact', load the vechicle with this amount.            
        '''
        
        if load_policy=='max':
            inventory_loaded=min(self.current_inventory,vehicle.Q)
        elif load_policy=='exact':
            inventory_loaded=exact_load
        elif load_policy=='none':
            inventory_loaded=None
        
        vehicle.current_inventory+=inventory_loaded
        self.current_inventory-=inventory_loaded
        
    
    def accrue_cost(self):
        ''' Calculate cost for the period
        '''
        self.inventory_cost+=self.current_inventory*self.h
    
    def produce(self):
        ''' Produce inventory for the period
        '''
        self.current_inventory+=self.r

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, i: {}, h: {}, r: {}.format(self.id, self.type, self.x, self.y, self.i, self.h, self.r)'
        
### Customer class ###
class Customer(Node):

    def __init__(self, id:int, type:int, x:float, y:float, i:int, h:float, r:int, u:int, l:int):
        '''Initialize a customer
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
            i::int
                Starting inventory level
            h::float
                Inventory cost
            r::int
                Daily #production/(cost)
            u::int
                Maximum inventory level
            l::int
                Minimum inventory level
        '''
        super(Customer, self).__init__(id, type, x, y, i, h, r)        
        self.u=u
        self.l=l
        
        # Inventory cost used in calculating objective function
        self.inventory_cost=0
        self.current_inventory=self.i
        self.no_delivery=0

    def accrue_cost(self):
        self.inventory_cost+=self.current_inventory*self.h
        
    def consume(self):
        assert self.current_inventory - self.r >=0, f"Customer {self.id} stock out!"
        self.current_inventory-=self.r

    def check_capacity(self,inventory_delivered:float):
        return self.current_inventory + inventory_delivered <= self.u

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, i: {}, h: {}, r: {}, u: {}, l: {}'.format(self.id, self.type, self.x, self.y, self.i, self.h,self.r,self.u,self.l)
    

### Vehicle class ###
class Vehicle(object):

    def __init__(self, id:int, start_node:Node, end_node:Node, Q:float):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            end_node::Node
                ending node of the vehicle
            Q::float
                Vehicle capacity            
        '''
        self.id = id
        self.start_node = start_node
        self.end_node = end_node        
        self.Q = Q
        
        # To keep track of current node
        self.current_node= start_node

        # To keep track of inventory
        self.current_inventory=0
        
        # travel time of the vehicle
        self.travel_cost = 0
        
        # all the nodes including depot & customers
        self.node_visited = [self.start_node]  # start from depot

    def check_return(self):
        ''' Check whether the vehicle's return to the depot
        Return True if returned, False otherwise
        '''
        if len(self.node_visited) > 1:
            return self.node_visited[-1] == self.end_node
    
    def move(self,node_start:Node,node_end:Node):
        ''' Move the vehicle to next node, and deliver if customer
        Args:            
            node_start::Node
                Node to move from
            node_end::Node
                Node to move to
            
        '''
        
        self.travel_cost+=cdist([(node_start.x,node_start.y)],[(node_end.x,node_end.y)])
        self.current_node=node_end
        self.node_visited.append(node_end)
            
    def unload(self,node:Customer,unload_policy:str,exact_unload:float=None):        
        '''Unloads inventory from vehicle to node
        Args:
            node::Customer
                The node to unload at
            unload_policy::str ('max','exact')
                Determines how to unload inventory.
            exact_unload::float
                Inventory to unload under 'exact' policy
        '''
        if unload_policy=='max':
            inventory_unloaded = min(node.u - node.current_inventory,self.current_inventory)
                        
        elif unload_policy=='exact':
            inventory_unloaded=exact_unload
                            
        elif unload_policy=='none':
            inventory_unloaded=0                
        
        assert node.check_capacity(inventory_unloaded), f"Customer {node.id} over capacity!"
        assert node.no_delivery==0, f"Customer {node.id} already received delivery this cycle!"

        node.no_delivery+=1
        node.current_inventory+=inventory_unloaded
        self.current_inventory-=inventory_unloaded
        
    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, Q:{}, current_inventory: {}, travel_cost: {}'\
            .format(self.id, self.start_node, self.end_node, self.Q, self.current_inventory, self.travel_cost)

### IVRP state class ###
class IVRP(State):

    def __init__(self, name, depot: Depot, customers: List[Customer], vehicles: List[Vehicle], nPeriods:int, nVehicles:int, destruction: float = 0.25):
        '''Initialize the EVRP state
        Args:
            name::str
                name of the instance
            depot::Depot
                depot of the instance
            customers::[Customer]
                customers of the instance            
            vehicles::[Vehicle]   
                Vehicles of the instance
            nPeriods::int
                Number of periods to simulate
            destruction::Float
                Degree of destruction to be passed onto destroy operators where appropriate
        '''
        self.name = name
        self.depot = depot
        self.customers = customers        
        # record the vehicle used
        self.vehicles = vehicles
        # record numer of periods
        self.nPeriods = nPeriods
        # record numer of vehicles
        self.nVehicles = nVehicles
        # record the visited customers, eg. [Customer1, Customer2]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = []
        # the route visited by each vehicle, eg. [vehicle1.node_visited, vehicle2.node_visited, ..., vehicleN.node_visited]
        self.route = []
        # Degree of destruction for destroy operators where appropriate
        self.destruction = destruction
        # total travelled distance
        self.travel_distance = 0
        # total inventory cost
        self.inventory_cost = 0
        # saving of quantities solution
        self.DeliverQuantities_solution = None

    def random_initialize(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random_tour = copy.deepcopy(self.customers)
        random.shuffle(random_tour)
        self.random_generate_routes(random_tour)
        x, y = self.get_edges()
        self.get_DeliverQuantities(x, y)
        return self.objective()

    def random_generate_routes(self, tour):
        '''Randomly generate routes by spliting tour into n_veh sub-tours and visiting each customer in the same
        order for all time periods.
        Args:
            tour::array
        '''
        split_no = len(tour) // self.nVehicles
        for t in range(self.nPeriods):
            self.customer_visited.append([[self.depot] + tour[int(split_no*(i-1)):int(split_no*i)] + [self.depot] for i in range(1, self.nVehicles)] + [[self.depot]+tour[int(split_no*(self.nVehicles - 1)):]+[self.depot]])

    def get_edges(self):
        # list of customer-customer-vehicle-period (i,j,k,t) excl. t=0 initial period, incl depot
        x = {}
        for t in range(1, self.nPeriods + 1): 
            for k in range(self.nVehicles):
                for i in range(len(self.customers) + 1):
                    for j in range(len(self.customers) + 1):
                        if j != i:
                            x[(i, j, k, t)] = 0
                            
        # list of cust-vehicle-period (i,k,t) excl depot excl t=0 initial period 
        y = {}
        for t in range(1, self.nPeriods + 1):
            for k in range(self.nVehicles):
                for i in range(1, len(self.customers) + 1):
                    y[(i, k, t)] = 0

        customer_ids = [self.depot.id] + [i.id for i in self.customers]
        travelled_distance = 0
        for t_period in range(1, self.nPeriods + 1):
            for t_veh in range(self.nVehicles):
                for node in range(len(self.customer_visited[t_period-1][t_veh]) - 1):
                    x[(customer_ids.index(self.customer_visited[t_period-1][t_veh][node].id), 
                       customer_ids.index(self.customer_visited[t_period-1][t_veh][node+1].id), 
                       t_veh, t_period)] = 1
                    travelled_distance += round(cdist([[self.customer_visited[t_period-1][t_veh][node].x, 
                                                        self.customer_visited[t_period-1][t_veh][node].y]], 
                                                    [[self.customer_visited[t_period-1][t_veh][node+1].x, 
                                                      self.customer_visited[t_period-1][t_veh][node+1].y]], 
                                                    'euclidean')[0][0])
                    if customer_ids.index(self.customer_visited[t_period-1][t_veh][node].id) != 0:
                        y[(customer_ids.index(self.customer_visited[t_period-1][t_veh][node].id), 
                           t_veh, t_period)] = 1
        self.travel_distance = travelled_distance
        return x, y

    def copy(self):
        return copy.deepcopy(self)

    def get_DeliverQuantities(self, x, y):
        ### MODEL INITIALIZATION 
        ivrp = self
        m = Model('ivrp')

        ### DECISION VARIABLES FORUMATION

        #no. of customers
        n_cust = len(ivrp.customers)
        #no. of nodes (customers+depot)
        n_node = n_cust + 1
        #no. of vehicles
        n_veh = ivrp.nVehicles
        #no. of periods (incl. t=0 for initial state)
        t_period = ivrp.nPeriods + 1

        # list of node-period pair (i, t) excl. t=0 initial period, incl. depot
        I = []
        for t in range(t_period):
            for i in range(n_node):
                I.append((i, t))

        # List of customer pairs (i,j), no depot
        A = []
        for i in range(n_cust):
            for j in range(n_cust):
                if i != j:
                    A.append((ivrp.customers[i].id, ivrp.customers[j].id))

        # list of cust-vehicle-period (i,k,t) excl depot excl t=0 initial period 
        Q = []
        for t in range(1, t_period):
            for k in range(n_veh):
                for i in range(1, n_node):
                    Q.append((i, k, t))

        # i(i,t) = inventory level of customer i in period t
        # q(i,k,t) = qunatity delivered to customer i via vehicle k in period t
        # y(i,k,t) = 1 if customer i is visited by vehicle k in period t
        # w(i,k,t) = cumulative qty delivered by k up to and incl. cust i in t
        # x(i,j,k,t) = 1 if customer j is visited after customer i via k in t
        inv = m.continuous_var_dict(I, name = 'inv')
        q = m.continuous_var_dict(Q, lb = 0, name = 'qty')

        ### PARAMETERS DEFINITION                    

        # distance between customers 
        c = {(i,j) : round(cdist([[ivrp.customers[int(i-1)].x, 
                            ivrp.customers[int(i-1)].y]], 
                        [[ivrp.customers[int(j-1)].x, 
                            ivrp.customers[int(j-1)].y]], 
                        'euclidean')[0][0]) for (i,j) in A}

        # depot-customer pairs and their distances
        for i in range(1,n_node):
            A.append((0,i))
            c[0,i] = round(cdist([[ivrp.depot.x, 
                        ivrp.depot.y]], 
                        [[ivrp.customers[int(i-1)].x, 
                        ivrp.customers[int(i-1)].y]], 
                        'euclidean')[0][0])
            c[i,0] = c[0,i]

        # holding cost at depot and customers
        h = {}
        h[0] = ivrp.depot.h
        for i in ivrp.customers:
            h[i.id] = i.h

        # daily production at depot
        r = ivrp.depot.r

        # daily demand|consumption for each customer
        d = {}
        for i in ivrp.customers:
            d[i.id] = i.r

        # min inventory level
        l = {}
        for i in ivrp.customers:
            l[i.id]=i.l

        # max inventory level
        u = {}
        for i in ivrp.customers:
            u[i.id]=i.u

        # initial inventory (at t=0) 
        m.add_constraint(inv[(0,0)] == ivrp.depot.i)
        for i in ivrp.customers:
            m.add_constraint(inv[(i.id, 0)] == i.i)

        # vehicle capacity
        cap = {}
        for i in ivrp.vehicles[0]:
            cap[i.id] = i.Q


        #(31) Objective function
        m.minimize(m.sum((h[0] * inv[(0, t)])
                        for t in range(1, t_period)) + \
                m.sum((h[i] * inv[(i, t)])
                        for i in range(1, n_node)
                        for t in range(1, t_period)))

        #(32) Inv at Depot this period = last period's + production - delivered
        m.add_constraints((inv[(0,t)] == inv[(0, t - 1)] + r - 
                        (m.sum(q[(i,k,t)] 
                                for i in range(1, n_node)
                                for k in range(n_veh))))
                        for t in range(1, t_period))

        #(33) Inv at Cust = last period's inv - consumed + delivered
        m.add_constraints((inv[(i,t)] == inv[(i,t - 1)] - d[i]
                        + m.sum(q[(i,k,t)] 
                                for k in range(n_veh)))
                        for i in range(1, n_node)
                        for t in range(1, t_period))

        #(34) Inv at Depot >=0
        m.add_constraints(inv[(0, t)] >= 0 for t in range(1, t_period))

        #(35) Inv at Cust >= lower bound
        m.add_constraints((inv[(i,t)] >= l[i])
                        for i in range(1, n_node)
                        for t in range(1, t_period))

        #(36) Inv at Cust <= upper bound
        m.add_constraints((inv[i,t] <= u[i])
                        for i in range(1, n_node)
                        for t in range(1, t_period))

        #(37) Qty delivered cannot exceed space in Cust warehouse (i.e. upper - existing inv)
        m.add_constraints((m.sum(q[(i,k,t)] for k in range(n_veh)) <= u[i] - inv[(i,t-1)])
                        for i in range(1, n_node)
                        for t in range(1, t_period))

        #(38) If x is 1, qty delivered<capacity, if x is 0, qty delivered=0. Capacity is the large-M
        m.add_constraints((m.sum(q[(i,k,t)] for k in range(n_veh)) <= 
                        u[i] * m.sum(x[(i, j, k, t)]
                                for j in range(n_node) if j != i
                                for k in range(n_veh)))
                        for i in range(1, n_node)
                        for t in range(1, t_period))

        #(39) Qty to be delivered by each vehicle within vehicle's capacity
        m.add_constraints((m.sum(q[(i,k,t)] for i in range(1,n_node)) <= 
                        cap[k])
                        for k in range(n_veh) 
                        for t in range(1, t_period))

        #(40) qty delivered to each cust is below capacity if visited, and 0 if not visited. Capacity is the large-M
        m.add_constraints((q[(i,k,t)]<=(y[(i,k,t)] * u[i]))
                        for i in range(1,n_node)
                        for k in range(n_veh)
                        for t in range(1,t_period))

        solution = m.solve(log_output = False)
        if solution != None:
            self.inventory_cost = solution.get_objective_value()
            self.DeliverQuantities_solution = solution
        else:
            self.inventory_cost = 10000000
            self.DeliverQuantities_solution = None
        return solution

    def advance_time(self):        
        # Consume and then accrue costs
        for customer in self.customers:
            customer.consume()
            customer.accrue_cost()
            customer.no_delivery=0
        # Produce inventory at depot
        self.depot.produce()
        
    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        return self.inventory_cost + self.travel_distance