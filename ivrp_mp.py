### LIBRARY
import os
import sys
from ivrp import *
from docplex.mp.model import Model
import argparse

def run_cplex_vrp(path:str,filename:str):
    '''Runs the MP solution for IVRP
    Args:
        path::str:: Folder where .dat files are stored
        filename::str:: Name of file to load
    
    Writes results to file if optimal solutions are found
    '''
    parsed = Parser(os.path.join(path, filename))

    ivrp = IVRP(parsed.name, parsed.depot, parsed.customers, parsed.vehicles, parsed.nPeriods)

    ### MODEL INITIALIZATION 

    m = Model('ivrp')

    ### DECISION VARIABLES FORUMATION

    #no. of customers
    n_cust = len(ivrp.customers)
    #no. of nodes (customers+depot)
    n_node = n_cust + 1
    #no. of vehicles
    n_veh = len(ivrp.vehicles)
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

    # list of customer-customer-vehicle-period (i,j,k,t) excl. t=0 initial period, incl depot
    X = []
    for t in range(1, t_period): 
        for k in range(n_veh):
            for i in range(n_node):
                for j in range(n_node):
                    if j != i:
                        X.append((i, j, k, t))

    # i(i,t) = inventory level of customer i in period t
    # q(i,k,t) = qunatity delivered to customer i via vehicle k in period t
    # y(i,k,t) = 1 if customer i is visited by vehicle k in period t
    # w(i,k,t) = cumulative qty delivered by k up to and incl. cust i in t
    # x(i,j,k,t) = 1 if customer j is visited after customer i via k in t
    inv = m.continuous_var_dict(I, name = 'inv')
    q = m.continuous_var_dict(Q, lb = 0, name = 'qty')
    y = m.binary_var_dict(Q, name = 'y')
    w = m.continuous_var_dict(Q, lb = 0, name = 'w')
    x = m.binary_var_dict(X, name = 'x')


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
    for i in ivrp.vehicles:
        cap[i.id] = i.Q


    ### MODEL FORUMATION

    ### OBJECTIVE FUNTION

    #(1) Objective function
    m.minimize(m.sum((h[i] * inv[(i, t)])
                        for i in range(n_node) 
                        for t in range(1, t_period)) + \
                m.sum((c[(i, j)] * x[(i, j, k, t)])
                        for i in range(n_node)
                        for j in range(n_node) if j != i
                        for k in range(n_veh) 
                        for t in range(1, t_period)))

    ### CONSTRAINTS

    #(2) Inv at Depot this period = last period's + production - delivered
    m.add_constraints((inv[(0,t)] == inv[(0,t-1)] + r - 
                        (m.sum(q[(i,k,t)] 
                                for i in range(1,n_node)
                                for k in range(n_veh))))
                    for t in range(1,t_period))

    #(3) Inv at Depot >=0
    m.add_constraints(inv[(0,t)] >= 0 for t in range(1,t_period))

    #(4) Inv at Cust = last period's inv - consumed + delivered
    m.add_constraints((inv[(i,t)] == inv[(i,t-1)] - d[i]
                        + m.sum(q[(i,k,t)] 
                                for k in range(n_veh)))
                        for i in range(1,n_node)
                        for t in range(1,t_period))

    #(5) Inv at Cust >= lower bound
    m.add_constraints((inv[(i,t)] >= l[i])
                        for i in range(1,n_node)
                        for t in range(1,t_period))

    #(6) Inv at Cust <= upper bound
    m.add_constraints((inv[i,t]<=u[i])
                        for i in range(1,n_node)
                        for t in range(1,t_period))

    #(7) Qty delivered cannot exceed space in Cust warehouse (i.e. upper - existing inv)
    m.add_constraints((m.sum(q[(i,k,t)] for k in range(n_veh)) <= u[i] - inv[(i,t-1)])
                        for i in range(1,n_node)
                        for t in range(1,t_period))

    #(8) If x is 1, qty delivered<capacity, if x is 0, qty delivered=0. Capacity is the large-M
    m.add_constraints((m.sum(q[(i,k,t)] for k in range(n_veh)) <= 
                        u[i] * m.sum((x[(i,j,k,t)] 
                            for j in range(n_node) if j != i
                            for k in range(n_veh))))
                        for i in range(1,n_node)
                        for t in range(1,t_period))

    #(9) Qty to be delivered by each vehicle within vehicle's capacity
    m.add_constraints((m.sum(q[(i,k,t)] for i in range(1,n_node)) <= 
                        cap[k])
                        for k in range(n_veh) 
                        for t in range(1,t_period))

    #(10) qty delivered to each cust is below capacity if visited, and 0 if not visited. Capacity is the large-M
    m.add_constraints((q[(i,k,t)]<=(y[(i,k,t)]*u[i]))
                        for i in range(1,n_node)
                        for k in range(n_veh)
                        for t in range(1,t_period))

    #(11) for each visited cust, there must be a node before and after 
    m.add_constraints((m.sum(x[(i,j,k,t)] for j in range(n_node) if i!=j) ==
                        m.sum(x[(j,i,k,t)] for j in range(n_node) if j!=i))
                        for i in range(1,n_node)
                        for k in range(n_veh)
                        for t in range(1,t_period))
    m.add_constraints(((m.sum(x[(i,j,k,t)] for j in range(n_node) if i!=j) ==
                        m.sum(y[(i,k,t)]))
                        for i in range(1,n_node)
                        for k in range(n_veh)
                        for t in range(1,t_period)))
    m.add_constraints(((m.sum(x[(j,i,k,t)] for j in range(n_node) if i!=j) ==
                        m.sum(y[(i,k,t)]))
                        for i in range(1,n_node)
                        for k in range(n_veh)
                        for t in range(1,t_period)))

    #(12) at most one route from each node (more for Depot)
    m.add_constraints((m.sum(x[(0,j,k,t)] for j in range(1, n_node)) <= 1)
                        for k in range(n_veh) 
                        for t in range(1,t_period))

    #(13) at most one veh visited each cust 
    m.add_constraints((m.sum(y[(i,k,t)] for k in range(n_veh)) <= 1)
                        for i in range(1,n_node) 
                        for t in range(1,t_period))

    #(14) routing logic (subtour elimination)
    m.add_constraints(((w[(i,k,t)] - w[(j,k,t)] + (cap[k])*(x[(i,j,k,t)]))
                        <= (cap[k]) - q[(j,k,t)])
                        for i in range(1,n_node) 
                        for j in range(1,n_node) if j!=i 
                        for k in range(n_veh) 
                        for t in range(1,t_period))

    #(15) variable logic
    m.add_constraints((q[(i,k,t)] <= w[(i,k,t)])
                        for i in range(1,n_node) 
                        for k in range(n_veh) 
                        for t in range(1,t_period))
    m.add_constraints((w[(i,k,t)] <= cap[k])
                        for i in range(1,n_node) 
                        for k in range(n_veh) 
                        for t in range(1,t_period))
    #m.add_constraints(q[(i,k,t)] <= cap[k]
    #                  for i in range(1,n_node) 
    #                  for k in range(n_veh) 
    #                  for t in range(1,t_period))
    #m.add_constraints(q[(i,k,t)] <= w[(i,k,t)] <= cap[k] 
    #                  for i in range(1,n_node) 
    #                  for k in range(n_veh) 
    #                  for t in range(1,t_period))


    #(16) & (17)
    # covered in var definition

    ### SOLUTION

    solution = m.solve(log_output = True)

    str_builder = ['Instance: {}\nObjective: {}\nTime Taken: {} - {}\n'.format(ivrp.name, solution.get_objective_value(), solution.solve_details.time, solution.solve_details.status)]

    str_builder.append('Inventory Levels:-\n')
    for i in range(n_node):
        inven_level_temp = []
        for j in range(t_period):
            inven_level_temp.append(round(inv[i, j].solution_value))
        str_builder.append('Node ' + str(i) + ': ' + ', '.join(str(k) for k in inven_level_temp))
    str_builder.append('\n')

    str_builder.append('Consolidated Routes:-')
    for t in range(1, t_period):
        str_builder.append('\nTime Peroid ' + str(t) + ":")
        for k in range(n_veh):
            vehicle_temp_str = []
            vehicle_temp_str.append('Vehicle ' + str(k) + ': 0')
            next_node = -1
            for i in range(1, n_node):
                if x[0, i, k, t].solution_value:
                    next_node = i
            while next_node > 0:
                for j in range(n_node):
                    if j != next_node:
                        if x[next_node, j, k, t].solution_value == 1:
                            if next_node != 0:
                                vehicle_temp_str.append("(" + str(next_node) + ", " + str(round(q[next_node, k, t].solution_value)) + ")")
                            else:
                                vehicle_temp_str.append(str(j))
                            next_node = j
                            break
            str_builder.append(', '.join(vehicle_temp_str) + ', 0')

    with open('{}_MP_Solution.txt'.format(ivrp.name), 'w') as f:
        f.write('\n'.join(str_builder))

    with open('Overall Results.txt'.format(ivrp.name), 'a') as f:
        f.write('Instance: {}\nObjective: {}\nTime Taken: {} - {}\n'.format(ivrp.name, solution.get_objective_value(), solution.solve_details.time, solution.solve_details.status))
        
    overall_results.append((ivrp.name, solution.get_objective_value(), solution.solve_details.time, solution.solve_details.status))
    
if __name__=="main":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--filename', action='store_false')    
    
    args = parser.parse_args()
    
    path = args.path
    filename = args.filename

    overall_results = []

    with open('Overall Results.txt', 'w') as f:   
        f.write('')
    
    if not filename:
        for name in os.listdir(path):
            run_cplex_vrp(path,name)
    
    else:
        run_cplex_vrp(path,filename)