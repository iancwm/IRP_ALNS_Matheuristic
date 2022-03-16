'''
Description: IVRP State
Author: Group 5
Date: 2022-03-15
LastEditors: Ian Chong
LastEditTime: 2022-03-15
'''

import copy
import numpy as np
import random
import os

from scipy.spatial.distance import cdist
from pathlib import Path

from typing import List

import sys


sys.path.append('./ALNS')
from alns import ALNS, State

### Parser to parse instance xml file ###
# You should not change this class!


class Parser(object):
    
    def __init__(self, dat_file):
        '''initialize the parser
        Args:
            dat_file::str
                the path to the .dat file
        '''
        self.name = os.path.splitext(dat_file)[0]
        
        self.datContent = [i.strip().split() for i in open(dat_file).readlines()]
        
        self.nNodes = int(self.datContent[0][0])
        self.nPeriods = int(self.datContent[0][1])
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
        for i in range(int(self.datContent[0][3])):
            self.vehicles.append(Vehicle(i, self.depot, self.depot, int(self.datContent[0][2])))
        

### Node class ###
# You should not change this class!


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
                Daily production/(consumption)
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
                Daily production/(cost)
        '''
        super(Depot, self).__init__(id, type, x, y, i, h, r)
        
        # Initialize current inventory level
        self.current_inventory=i
        
        # Keep track of inventory costs
        self.inventory_cost=0
        
    def load(self,vehicle:Vehicle,load_policy:str='max',exact_load:float=None):
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
        
        
### Customer class ###
# You should not change this class!


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
                Daily production/(cost)
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

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, service_time: {}'.format(self.id, self.type, self.x, self.y, self.service_time)
    
    def accrue_cost(self):
        self.inventory_cost+=self.current_inventory*self.h
        
    def consume(self):
        self.current_inventory-=self.r

### Vehicle class ###
# Vehicle class. You could add your own helper functions freely to the class, and not required to use the functions defined
# But please keep the rest untouched!


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
        
        # all the nodes including depot, customers, or charging stations (if any) visited by the vehicle
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
        
        self.travel_cost+=cdist([(node_start.x,node_start.y)],[(node_end.x,node_end.y]))
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
        
        node.current_inventory+=inventory_unloaded
        self.current_inventory-=inventory_unloaded
        
    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, max_travel_time: {}, speed_factor: {}, consumption_rate: {}, battery_capacity: {}'\
            .format(self.id, self.start_node, self.end_node, self.max_travel_time, self.speed_factor, self.consumption_rate, self.battery_capacity)

### EVRP state class ###
# EVRP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!


class IVRP(State):

    def __init__(self, name, depot: Depot, customers: List[Customer], vehicles: List[Vehicle], nPeriods:int, destruction: float = 0.25):
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
        # record the visited customers, eg. [Customer1, Customer2]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = []
        # the route visited by each vehicle, eg. [vehicle1.node_visited, vehicle2.node_visited, ..., vehicleN.node_visited]
        self.route = []
        # Degree of destruction for destroy operators where appropriate
        self.destruction = destruction

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
        self.split_route(random_tour)
        return self.objective()

    def copy(self):
        return copy.deepcopy(self)

    def split_route(self, tour):
        '''Generate the route given a tour visiting all the customers
        Args:
            tour::[Customer]
                a tour (list of Nodes) visiting all the customers        

        # You should update the following variables for the EVRP
        EVRP.vehicles
        EVRP.travel_time
        EVRP.customer_visited
        EVRP.customer_unvisited
        EVRP.route

        # You should update the following variables for each vehicle used
        Vehicle.travel_cost
        Vehicle.node_visited
        
        '''
        # You should implement your own method to construct the route of EVRP from any tour visiting all the customers
    
    def advance_time(self):        
        # Consume and then accrue costs
        for customer in self.customers:
            customer.consume()
            customer.accrue_cost()
        # Produce inventory at depot
        self.depot.produce()
        
    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''        
        return sum([v.travel_cost for v in self.vehicles]) + sum([c.inventory_cost for c in self.customers] + self.depot.inventory_cost)