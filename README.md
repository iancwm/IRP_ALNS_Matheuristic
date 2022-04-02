# CS606 AI Planning Group Project - Inventory Vehicle Routing Problem

The Inventory Routing Problem (IRP) is one of the most challenging of the VRP variants, as it combines both inventory management and routing decisions into a single problem.

This group project follows the conventions and rules of the 12th DIMACS Implementation Challenge.

----

# Instructions to run code

Execution code are all in respective `.ipynb` files

- Exact Method: `ivrp_mp.ipynb`
- Metaheuristic Approach: `ivrp_metaheuristic.ipynb`
- Matheuristic Approach: `ivrp_matheuristic.ipynb`

Only the Exact Method iterates through the whole folder. The Metaheuristic and Matheuristic Approaches are single instance.

----

# Problem Statement

## Description

The IRP consists of routing a single commodity from a single supplier (called the depot in the following) to multiple customers over a given planning horizon. Input for IRP consists of locations for the depot and the set of $n$ customers $\{1,\dots,n\}$, a time horizon consisting of $T$ time periods $\{1,\dots ,T\}$, and a ﬂeet of $M$ vehicles $\{1,\dots ,M\}$. The depot is located at node 0. Each vehicle has capacity $Q$.

In each time period $t$, $r_{0t}$ units of the commodity are made available at the depot, and rit units are  consumed by customer $i$. Each customer $i\in \{1,\dots ,n\}$ begins with an initial inventory level of $I_{i0}$ and the depot begins with an initial inventory level of $I_{00}$. Each customer $i \in \{1,\dots , n\}$ must maintain an inventory level of at least $L_i$ and at most $U_i$ and will incur a per-period cost of $h_i$ for each unit of inventory held. There is no maximum inventory level deﬁned for the depot, but it cannot fall below 0, and there is per-period inventory holding cost of $h_0$. A matrix $D$ speciﬁes the distance (or some other cost) to travel between each pair of nodes.

The IRP is the problem of determining, for each time period t, the quantity to deliver to each customer i and the routes by which to serve those customers. An optimal solution is one that minimizes the total cost while assuring that constraints on vehicle capacity, vehicle routing, and node inventory levels are satisﬁed at all times. The total cost of a solution includes both the total of the inventory holding costs at all nodes (customers and depot) and the costs of the vehicle routes. A feasible solution must assure that:

- No customer stock-outs occur
- The depot has enough resource to meet deliveries in each period
- Each route begins and ends at the depot and honors the vehicle capacity constraint
- In each time period, a customer receives at most one delivery

This is the so-called Maximum Level (ML) replenishment policy, where any quantity can be delivered to the customers, provided that the maximum inventory level $U_i$ is not exceeded and the minimum inventory level $L_i$ is satisﬁed.

Maximum and minimum inventory level constraints: The maximum inventory level constraint establishes that, at each time period $t$, the sum of the inventory level in $t−1$ plus the quantity received in $t$ should not exceed $U_i$. The minimum inventory level constraint establishes that the inventory level at time $t$ should not be lower than $L_i$ for each $t$.

## Data format

Each IRP instance has the following format:

- The ﬁrst line contains four parameters: the number of nodes including the depot $(n+1)$, number of time periods $T$, vehicle capacity $Q$, and number of vehicles $M$.
- The second line describes the depot and provides: depot identiﬁer (which is $``0"$), $x$ coordinate, $y$ coordinate, starting inventory level $I_{00}$, daily production $r_{0t}$, inventory cost $h_0$.
- The $n$ lines that follow each describes a customer and provides: customer identiﬁer $(``i")$, $x$ coordinate, $y$ coordinate, starting inventory level $I_{i0}$, maximum inventory level $U_i$, minimum inventory level $L_i$, daily consumption $r_{it}$, inventory cost $h_i$.

For further definitions and conditions, please consult [the DIMACS Inventory Routing website](http://dimacs.rutgers.edu/programs/challenge/vrp/irp/).

----

# Methodology

We aim to use adaptive large neighbourhood search (ALNS) as introduced in (paper) as well as conventional combinatorial optimization. We have adapted the python `alns` package for our use, and will use `docplex` under the academic license granted to us by the University.

We will compare the performance of `docplex` with our ALNS solution for the small problems. For the large problems, the solution space will expand out of the capabilities of `docplex`, and we will continue to use ALNS.

Our results will be compared against other DIMACS solutions, and a thorough analysis our solution performance will be carried out.
