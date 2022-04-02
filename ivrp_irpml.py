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
from docplex.mp.model import Model

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
        self.name = os.path.split(dat_file)[1]

        self.datContent = [i.strip().split() for i in open(dat_file).readlines()]

        self.nNodes = int(self.datContent[0][0])
        self.nPeriods = int(self.datContent[0][1])
        self.customers = []
        self.vehicles = []
        self.depot = None
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
                self.customers.append(
                    Customer(int(self.datContent[i][0]), 1, float(self.datContent[i][1]), float(self.datContent[i][2]),
                             int(self.datContent[i][3]), float(self.datContent[i][7]), int(self.datContent[i][6]),
                             int(self.datContent[i][4]), int(self.datContent[i][5])))

    def set_vehicles(self):
        # Initialize Vehicle
        for i in range(int(self.datContent[0][3])):
            self.vehicles.append(Vehicle(i, self.depot, self.depot, int(self.datContent[0][2])))


### Node class ###
# You should not change this class!


class Node(object):

    def __init__(self, id: int, type: int, x: float, y: float, i: float, h: float, r: float):
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

    def __init__(self, id: int, type: int, x: float, y: float, i: int, h: float, r: int):
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
        self.current_inventory = i

        # Keep track of inventory costs
        self.inventory_cost = 0

    def load(self, vehicle, load_policy: str = 'max', exact_load: float = None):
        '''Load a vehicle
        Args:
            vehicle::Vehicle
                Vehicle to load inventory
            load_policy::str ('max','exact','none')
                Determines how much to load
            exact_load::float
                If load_policy is 'exact', load the vechicle with this amount.            
        '''

        if load_policy == 'max':
            inventory_loaded = min(self.current_inventory, vehicle.Q)
        elif load_policy == 'exact':
            inventory_loaded = exact_load
        elif load_policy == 'none':
            inventory_loaded = None

        vehicle.current_inventory += inventory_loaded
        self.current_inventory -= inventory_loaded

    def accrue_cost(self):
        ''' Calculate cost for the period
        '''
        self.inventory_cost += self.current_inventory * self.h

    def produce(self):
        ''' Produce inventory for the period
        '''
        self.current_inventory += self.r


### Customer class ###
# You should not change this class!


class Customer(Node):

    def __init__(self, id: int, type: int, x: float, y: float, i: int, h: float, r: int, u: int, l: int):
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
        self.u = u
        self.l = l

        # Inventory cost used in calculating objective function
        self.inventory_cost = 0
        self.current_inventory = self.i
        self.no_delivery = 0

    def accrue_cost(self):
        self.inventory_cost += self.current_inventory * self.h

    def consume(self):
        assert self.current_inventory - self.r >= 0, f"Customer {self.id} stock out!"
        self.current_inventory -= self.r

    def check_capacity(self, inventory_delivered: float):
        return self.current_inventory + inventory_delivered <= self.u

    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, i: {}, h: {}, r: {}, u: {}, l: {}'.format(self.id, self.type,
                                                                                               self.x, self.y, self.i,
                                                                                               self.h, self.r, self.u,
                                                                                               self.l)


### Vehicle class ###
# Vehicle class. You could add your own helper functions freely to the class, and not required to use the functions defined
# But please keep the rest untouched!


class Vehicle(object):

    def __init__(self, id: int, start_node: Node, end_node: Node, Q: float):
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
        self.current_node = start_node

        # To keep track of inventory
        self.current_inventory = 0

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

    def move(self, node_start: Node, node_end: Node):
        ''' Move the vehicle to next node, and deliver if customer
        Args:            
            node_start::Node
                Node to move from
            node_end::Node
                Node to move to
            
        '''

        self.travel_cost += cdist([(node_start.x, node_start.y)], [(node_end.x, node_end.y)])
        self.current_node = node_end
        self.node_visited.append(node_end)

    def unload(self, node: Customer, unload_policy: str, exact_unload: float = None):
        '''Unloads inventory from vehicle to node
        Args:
            node::Customer
                The node to unload at
            unload_policy::str ('max','exact')
                Determines how to unload inventory.
            exact_unload::float
                Inventory to unload under 'exact' policy
        '''
        if unload_policy == 'max':
            inventory_unloaded = min(node.u - node.current_inventory, self.current_inventory)

        elif unload_policy == 'exact':
            inventory_unloaded = exact_unload

        elif unload_policy == 'none':
            inventory_unloaded = 0

        assert node.check_capacity(inventory_unloaded), f"Customer {node.id} over capacity!"
        assert node.no_delivery == 0, f"Customer {node.id} already received delivery this cycle!"

        node.no_delivery += 1
        node.current_inventory += inventory_unloaded
        self.current_inventory -= inventory_unloaded

    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, Q:{}, current_inventory: {}, travel_cost: {}' \
            .format(self.id, self.start_node, self.end_node, self.Q, self.current_inventory, self.travel_cost)


### IVRP state class ###
# IVRP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!


class IVRP(State):

    def __init__(self, name, depot: Depot, customers: List[Customer], vehicles: List[Vehicle], sol, cust_assignment,
                 nPeriods: int,
                 destruction: float = 0.25):
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
            IVRP.sol
                solution class for docplex model
            IVRP.cust_assignment::[[customer]]
                nested list of customers to periods
            destruction::Float
                Degree of destruction to be passed onto destroy operators where appropriate
        '''
        self.name = name
        self.depot = depot
        self.customers = customers
        # record the vehicle used
        self.vehicles = vehicles
        # record number of periods
        self.nPeriods = nPeriods
        # solution generated by docplex MP
        self.sol = sol
        # customer assignment generated across periods
        self.cust_assignment = cust_assignment
        # record travelling time
        self.travelling_time = 0
        # record inventory_cost
        self.inventory_cost = 0
        # record all nodes
        self.nodes = [self.depot]
        self.nodes.extend[self.customers]
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
        # The initial solution is generated by randomly selecting 75% of the customers
        # and randomly assigning them to some periods of the planning horizon. Their
        # insertion in the routes follows the cheapest insertion rule. Our computational
        # experiments have shown that the initial solution does not have a significant
        # impact on the overall solution cost or running time.
        percentage = float(0.75)
        k = int(len(random_tour) * percentage)
        indices = random.sample(range(len(random_tour)), k)
        initialize_customers = [random_tour[i] for i in indices]
        # cust1 cust4 cust6 cust7
        random.shuffle(initialize_customers)
        # cust4 cust6 cust1 cust7
        random_period_assignment = [initialize_customers[i::self.nPeriods] for i in range(self.nPeriods)]
        # nPeriods = 3
        # random_period_assignment = [[cust4, cust6],[cust1], [cust7]]
        self.split_route(random_period_assignment)

        return self.objective()

    def copy(self):
        return copy.deepcopy(self)

    def split_route(self, tour):
        '''Generate the route given a tour visiting all the customers
        Args:
            tour::[Customer]
                a tour (list of Nodes) visiting all the customers        
        # You should update the following variables for the IVRP
        IVRP.vehicles
        IVRP.customers
        IVRP.travel_time
        IVRP.customer_visited
        IVRP.customer_unvisited
        IVRP.route

        IVRP.sol
        IVRP.cust_assignment

        # You should update the following variables for each vehicle used
        Vehicle.travel_cost
        Vehicle.node_visited
        
        '''
        # You should implement your own method to construct the route of IVRP from any tour visiting all the customers

        # We have applied our algorithm to the IRP-OU without any structural change.
        # To avoid passing an infeasible problem to the network
        # flow algorithm (for instance
        # when vehicle capacity would be exceeded or when a stockout would occur
        # at a customer due to it not being served as often as required), we have kept all
        # transshipment arcs from the supplier and from every customer to every other
        # customer, with large artificial costs; this means that feasible solutions can always
        # be reached, but at very high cost if transshipments are used. These costs
        # act as penalties in the objective function when the vehicle capacity is exceeded
        # or when the master level heuristic does not add all customers to the current
        # solution.
        # The remaining problem is then similar to the IRPT-OU: decisions regarding
        # routings are fixed by the ALNS algorithm and modeled as a network
        # flow problem
        # with one vertex representing the vehicle for each period, and arcs leaving
        # the vehicle vertex and arriving at each selected customer. The vehicle vertex
        # receives an arc from the supplier with up to Q units of
        # ow. The OU policy is
        # modeled by fixing the
        # floow on the arcs connecting customers in successive time
        # periods: once customer i is visited in period t, the arc linking to it in the next
        # period has a
        # flow equal to Ui - dti
        # .

        # Modeling the IRP-ML as a network
        # flow problem is similar to the IRP-OU,
        # except that arcs connecting the customers in successive time periods have a
        # minimum
        # ow equal to 0. The vehicle vertex is fed from the supplier with up to
        # Q units and the minimum-cost network
        # ow algorithm decides on how much to
        # deliver to each of the customers selected from the master level heuristic. Dummy
        # arcs are again inserted to penalize unvisited customers and solutions that would
        # require exceeded vehicle capacity. It is easy to see that the IRP-OU yields an
        # upper bound on the IRP-ML optimum as we just relaxed one constraint of the
        # former problem.

        self.cust_assignment = copy.deepcopy(tour)

        # create model
        m = Model('ivrp')

        # create decision variable
        depot_inventory = m.continuous_var_dict([(0, t) for t in range(self.nPeriods)],
                                                name='depot_inventory')
        customer_inventory = m.continuous_var_dict([(c.id, t) for c in self.customers for t in range(self.nPeriods)],
                                                   lb=[c.l for c in self.customers for t in range(self.nPeriods)],
                                                   ub=[c.u for c in self.customers for t in range(self.nPeriods)],
                                                   name='customer_inventory')
        holding_cost_nodes = dict(zip([c.id for c in self.nodes], [c.h for c in self.nodes]))
        distances = [[cdist([[i.x, i.y]], [[j.x, j.y]], 'euclidean')[0][0] for i in self.nodes] for j in self.nodes]
        leg = m.binary_var_dict(
            [(c1.id, c2.id, t) for c1 in self.nodes for c2 in self.nodes for t in range(self.nPeriods)],
            name='leg')
        total_shipped_cust = m.continuous_var_dict([(c.id, t) for c in self.customers for t in range(self.nPeriods)],
                                                   lb=[0 for c in self.customers for t in range(self.nPeriods)],
                                                   ub=[c.u for c in self.customers for t in range(self.nPeriods)],
                                                   name='customer_shipped')
        total_shipped = []
        for t in range(self.nPeriods):
            total_shipped[t] = m.sum((total_shipped_cust[c.id, t] for c in self.customers))
        v_cust_load_initial = m.continuous_var_dict([(v.id, c.id, t)
                                                     for v in self.vehicles
                                                     for c in self.customers
                                                     for t in range(self.Periods)],
                                                    name="v_cust_load_initial")

        # Objective Function
        # (1)
        m.minimize(m.sum(holding_cost_nodes[0] * depot_inventory[(0, t)]
                         for t in range(self.nPeriods)) +
                   m.sum(holding_cost_nodes[c.id] * customer_inventory[(c.id, t)]
                         for c in self.customers
                         for t in range(self.nPeriods)) +
                   m.sum(leg[(n1.id, n2.id, t)] * distances[n1.id][n2.id]
                         for n1 in self.nodes
                         for n2 in self.nodes
                         for t in range(self.nPeriods))
                   )

        # Constraints

        # The inventory level at the supplier in period t is dened at the beginning
        # of the period and is given by its previous inventory level (period t - 1),
        # plus the quantity rt-1 made available in period t - 1, minus the total
        # quantity shipped to the customers using the supplier's vehicle in period
        # t - 1

        # (2)
        m.add_constraints(
            depot_inventory[0, t] == m.sum((depot_inventory[0, t - 1] + self.depot.r) - total_shipped[t - 1])
            for t in range(self.nPeriods))

        # These constraints impose that the supplier's inventory cannot be less than
        # the total amount of product delivered in period t

        # (3)
        m.add_constraints(depot_inventory[0, t] >= total_shipped[t]
                          for t in range(self.nPeriods))

        # Likewise, the inventory level at each retailer in period t is given by its
        # previous inventory level in period t-1, plus the quantity yt-1
        # delivered by the supplier's vehicle in period t-1, minus its demand in period t - 1, that is:

        # (4)
        m.add_constraints(
            customer_inventory[c.id, t] == customer_inventory[c.id, t - 1] + total_shipped_cust[c.id, t - 1] - c.r
            for c in self.customers
            for t in range(self.nPeriods))

        # These constraints guarantee that for each customer, the inventory
        # level remains non-negative at all time:

        # (5)
        """(addressed by customer_inventory lb)"""

        # These constraints guarantee that for each customer, the inventory level
        # remains below the maximum level at all time:

        # (6)
        """(addressed by customer_inventory ub)"""

        # These sets of constraints ensure that the quantity delivered by the supplier's
        # vehicle to each customer in each period t will fill the
        # customer's inventory capacity if the customer is served, and will be zero otherwise:

        # (7) currently OU, HAVE TO REVISIT to tweak ML
        # m.add_constraints(
        #     total_shipped_cust[c1.id, t] >= (c1.u * m.sum(leg[c1.id, c2.id, t] for c2 in self.customers) - customer_inventory[c1.id, t])
        #                                     for c1 in self.customers for t in range(self.nPeriods)
        # )

        # (8)
        m.add_constraints(
            total_shipped_cust[c1.id, t] <= (c1.u - customer_inventory[c1.id, t])
            for c1 in self.customers for t in range(self.nPeriods))

        # (9)
        m.add_constraints(
            total_shipped_cust[c1.id, t] <= (c1.u * m.sum(leg[c1.id, c2.id, t]
                                                          for c2 in self.customers)
                                             ) for c1 in self.customers for t in range(self.nPeriods))

        # If customer i is not visited in period t, then constraints (9) mean that the
        # quantity delivered to it will be zero (while constraints (7) and (8) are still
        # respected). If, otherwise, customer i is visited in period t, constraints (9)
        # limit the quantity delivered to the customer's inventory holding capacity,
        # and this bound is tightened by constraints (8), making it impossible to
        # deliver more than what would exceed this capacity. Constraints (7) model
        # the OU replenishment policy, ensuring that the quantity delivered will be
        # exactly the bound provided by constraints (8).

        # These constraints guarantee that the vehicle's capacity is not exceeded:

        # (10) Used self.vehicles[0] since all vehicles are the same
        m.add_constraints(m.sum(total_shipped_cust[c.id, t] for c in self.customers) <= self.vehicles[0].Q for t in
                          range(self.nPeriods))

        # These constraints guarantee that a feasible route is determined to visit all
        # customers served in period t:

        # (11) Flow conservation constraints: these constraints impose that the
        # number of arcs entering and leaving a vertex should be the same:
        m.add_constraints(
            m.sum(leg[c1.id, c2.id, t] for c1 in self.customers) == m.sum(leg[c2.id, c1.id, t] for c1 in self.customers)
            for c2 in self.customers for t in range(self.nPeriods)
        )

        # (12) A single vehicle is available:
        m.add_constraints(m.sum(leg[c1.id, 0, t] for c1 in self.customers) <= 1
                          for t in range(self.nPeriods)
                          )

        # (13) Subtour elimination constraints:
        m.add_constraints(v_cust_load_initial(v.id, c1.id, t) - v_cust_load_initial(v.id, c2.id, t)
                          + (self.vehicles[0].Q * leg[c1.id, c2.id, t])
                          <= self.vehicles[0].Q - total_shipped_cust[c2.id, t]
                          for c1 in self.customers
                          for c2 in self.customers
                          for t in range(self.nPeriods)
                          for v in self.vehicles
        )
        # (14) Subtour elimination constraints cont..:
        m.add_constraints(total_shipped_cust[c.id, t] <= v_cust_load_initial(v.id, c.id, t)
                          for c in self.customers
                          for v in self.vehicles
                          for t in range(self.nPeriods))
        m.add_constraints(v_cust_load_initial(v.id, c.id, t) <= self.vehicles[0].Q
                          for c in self.customers
                          for v in self.vehicles
                          for t in range(self.nPeriods))

        # (15) Integrality and nonnegativity constraints
        m.add_constraints(v_cust_load_initial(v.id, c.id, t) >= 0
                          for c in self.customers
                          for v in self.vehicles
                          for t in range(self.nPeriods))
        m.add_constraints(total_shipped_cust[c.id, t] >= 0
                          for c in self.customers
                          for t in range(self.nPeriods))

        # (16) Integrality and nonnegativity constraints for leg

        """(addressed by leg binary_var_dict)"""

        self.sol = m.solve(log_output = True)


    def advance_time(self):
        # Consume and then accrue costs
        for customer in self.customers:
            customer.consume()
            customer.accrue_cost()
            customer.no_delivery = 0
        # Produce inventory at depot
        self.depot.produce()

    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        return sum([v.travel_cost for v in self.vehicles]) + sum(
            [c.inventory_cost for c in self.customers]) + self.depot.inventory_cost
