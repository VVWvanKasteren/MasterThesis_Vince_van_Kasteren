import sys

import tsplib95
import math
import statistics
import matplotlib.pyplot as plt
import time
import random
import itertools
import copy

from pyCombinatorial.utils import graphs, util

def loadData(file):
    # Load an instance from a TSPLIB file
    problem = tsplib95.load(file)

    # Get the number of nodes
    n = len(problem.node_coords)
    node_coords = problem.node_coords

    maxGridx = 0
    maxGridy = 0
    for i in node_coords:
        maxGridx = max(maxGridx, node_coords[i][0])
        maxGridy = max(maxGridy, node_coords[i][1])

    ratioX = 15000/maxGridx
    ratioY = 15000/maxGridy

    node_coords[0][0] += int(maxGridx*0.2)
    node_coords[0][1] += int(maxGridy*0.2)

    # If desired, coordinates could be scaled to fit the 15*15 km range
    #for i in node_coords:
        #node_coords[i][0] = int(node_coords[i][0]*ratioX)
        #node_coords[i][1] = int(node_coords[i][1]*ratioY)

    Cprime = []
    for i in range(1, n):
        p = random.uniform(0, 1)
        if p > 0.2:
            Cprime.append(i)


    return n, node_coords, Cprime


def generateData():
    gridRange = 15000   # Grid range in meters
    n = random.randint(121, 122) # number of customers
    node_coords = {}
    node_coords[0] = [3000, 3000]

    for i in range(1, n):
        node_coords[i]= [random.randint(0, gridRange), random.randint(0, gridRange)]

    Cprime = []
    for i in range(1, n):
        p = random.uniform(0, 1)
        if p > 0.2:
            Cprime.append(i)

    return n, node_coords, Cprime

def load_random_generated_instance(file_path):
    with open(file_path, 'r') as file:
        instance_data = file.readlines()

    # Initialize variables to store data
    n = None
    node_coords = {}
    Cprime = []

    # Iterate through each line in the file
    for line in instance_data:
        if line.startswith("DIMENSION:"):
            n = int(line.split(":")[1])
        elif line.startswith("NODE_COORDS:"):
            # Extract node coordinates
            for node_line in instance_data[instance_data.index(line) + 1:]:
                if node_line.startswith("CPRIME:"):
                    break
                node_info = node_line.split()
                node_id = int(node_info[0])
                node_coords[node_id] = [int(node_info[1]), int(node_info[2])]
        elif line.startswith("CPRIME:"):
            # Extract cprime nodes
            for cprime_line in instance_data[instance_data.index(line) + 1:]:
                Cprime.append(int(cprime_line))

    return n, node_coords, Cprime



# Load benchmark TSP instance
#n, node_coords, Cprime = loadData('ch130.tsp')

# Generate random instance
#n, node_coords, Cprime = generateData()

# Load random generated instance
n, node_coords, Cprime = load_random_generated_instance('TEST_INSTANCE_81.txt')

# Print input
print(f"Number of nodes: {n}")
print(f"Node coordinates: {node_coords}")
print(f"Cprime: {Cprime}")
print(f"Cprime %: {(len(Cprime)/(n-1))*100}%\n")

# Truck and UAV speed (km/h)
truckSpeed = 30
UAVspeed = 50

# UAV endurance (hours)
endu = 20/60

# UAV launch and retrieve time (hours)
sL = 1/60
sR = 1/60

# Average time to make a delivery (hours)
deliveryTime = 2/60

# Number of iterations
iterations = 5

distance_matrix = [[0 for j in range(n)] for i in range(n)]

for i in range(n):
    for j in range(n):
        distance_matrix[i][j] = abs((node_coords[i][0] - node_coords[j][0])) + abs((node_coords[i][1] - node_coords[j][1]))

time_distance_matrix = [[0 for j in range(n)] for i in range(n)]

for i in range(n):
    for j in range(n):
        time_distance_matrix[i][j] = (distance_matrix[i][j]/1000)/truckSpeed

UAV_distance_matrix = [[0 for j in range(n)] for i in range(n)]

for i in range(n):
    for j in range(n):
        UAV_distance_matrix[i][j] = math.sqrt((node_coords[i][0] - node_coords[j][0])**2 + (node_coords[i][1] - node_coords[j][1])**2)

time_UAV_distance_matrix = [[0 for j in range(n)] for i in range(n)]

for i in range(n):
    for j in range(n):
        time_UAV_distance_matrix[i][j] = (UAV_distance_matrix[i][j]/1000)/UAVspeed



def PDSTSP(time_UAV_distance_matrix, time_distance_matrix, Cprime, endu, deliveryTime, sL, sR, threshold_time):

    inUAVrange = []
    truck_locations = []

    for i in Cprime:
        if time_UAV_distance_matrix[0][i]*2 <= endu:
            inUAVrange.append(i)

    for i in node_coords:
        if i not in Cprime:
            truck_locations.append(i)

    timeThreshold = threshold_time / 1.9

    min_distance_truckLocations = {}

    for i in inUAVrange:
        min_distance_truckLocations[i] = [sys.maxsize, 0]

    for i in inUAVrange:
        for j in truck_locations:
            min_distance_truckLocations[i][0] = min(min_distance_truckLocations[i][0], time_distance_matrix[i][j])

    min_distance_truckLocations = sorted(min_distance_truckLocations.items(), key=lambda x:x[1], reverse=True)
    min_distance_truckLocations = dict(min_distance_truckLocations)

    for i in min_distance_truckLocations:
        min_distance_truckLocations[i][1] = min_distance_truckLocations[i][0]/ min_distance_truckLocations[list(min_distance_truckLocations.keys())[0]][0]

    inUAVrange_copy = copy.deepcopy(inUAVrange)
    random.shuffle(inUAVrange_copy)

    directServed = []
    timeCount = 0
    iteration = 0
    maxIteration = len(inUAVrange)/2

    while timeThreshold - timeCount > 0.1 and iteration < maxIteration:
        iteration += 1
        for i in inUAVrange_copy:
            p = random.uniform(0, 1)
            if timeCount + time_UAV_distance_matrix[0][i]*2 + sL + sR< timeThreshold and p < min_distance_truckLocations[i][1]:
                directServed.append(i)
                timeCount += time_UAV_distance_matrix[0][i]*2 + sL + sR
                min_distance_truckLocations[i][1] = -1

    save_coords = {}

    Cprime = [ele for ele in Cprime if ele not in directServed]
    for i in directServed:
        save_coords[i] = copy.deepcopy(node_coords[i])

    return Cprime, node_coords, time_distance_matrix, directServed, save_coords, inUAVrange


# Method that creates a tsp tour using the nearest neighbor algorithm
def solve_tsp_nearest(time_distance_matrix):
    num_cities = len(time_distance_matrix)
    visited = [False] * num_cities
    tour = []
    total_distance = 0

    # Start at the first city
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True


    # Repeat until all cities have been visited
    while len(tour) < num_cities:
        nearest_city = None
        nearest_distance = math.inf

        # Find the nearest unvisited city
        for city in range(num_cities):
            if not visited[city]:
                distance = time_distance_matrix[current_city][city]
                if distance < nearest_distance:
                    nearest_city = city
                    nearest_distance = distance

        # Move to the nearest city
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True
        total_distance += nearest_distance

    # Complete the tour by returning to the starting city
    tour.append(0)
    total_distance += time_distance_matrix[current_city][0]

    return tour, total_distance

# Method that creates a tsp tour using the sweep algorithm
def solve_tsp_sweep(time_distance_matrix, node_coords, n):

    rad = [0 for i in range(n)]

    for i in range(1, n):
        rad[i] = math.atan2(node_coords[i][1], node_coords[i][0])

    num_cities = n
    visited = [False] * num_cities
    tour = []
    total_distance = 0

    # Start at the first city
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True


    # Repeat until all cities have been visited
    while len(tour) < num_cities:
        nearest_city = None
        nearest_rad = math.inf

        # Find the next city hit by the ray
        for city in range(num_cities):
            if not visited[city]:
                rads = rad[city]
                if rads < nearest_rad:
                    nearest_city = city
                    nearest_rad = rads

        # Move to the next city
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True
        total_distance += time_distance_matrix[tour[-1]-1][tour[-1]]

    # Complete the tour by returning to the starting city
    tour.append(0)
    total_distance += time_distance_matrix[current_city][0]

    return tour, total_distance

# Method that creates a tsp tour using the savings algorithm
def solve_tsp_savings(time_distance_matrix, n):

    savings = [[0 for j in range(n)] for i in range(n)]
    tour_temp = []
    total_distance = 0

    for i in range(n):
        for j in range(n):
            savings[i][j] = time_distance_matrix[0][i] + time_distance_matrix[0][j] - time_distance_matrix[i][j]

    edge_count = [0 for i in range(n)]

    sorted_edges = sorted([(i, j, savings[i][j]) for i in range(n) for j in range(i + 1, n)], key=lambda x: x[2], reverse=True)

    visited = [0] * n

    # Method in order to check that no cycles are created
    def cycle(edge, tour_temp):

        cycle = False
        tour = []
        cycle_tour = []
        cycle_tour.append(edge[0])

        for i in range(len(tour_temp)):
            temp = []
            for elem in tour_temp[i]:
                temp.append(elem)
            tour.append(temp)

        for j in range(len(tour)):
            for i in range(len(tour)):
                if cycle_tour[-1] == tour[i][0]:
                    if tour[i][1] == edge[1]:
                        cycle = True
                    else:
                        cycle_tour.append(tour[i][1])
                        del(tour[i])
                    break
                if cycle_tour[-1] == tour[i][1]:
                    if tour[i][0] == edge[1]:
                        cycle = True
                    else:
                        cycle_tour.append(tour[i][0])
                        del(tour[i])
                    break

        return cycle

    # Method to prevent nodes from having a degree > 2
    def can_add_edge(edge, visited):
        if visited[edge[0]] < 2 and visited[edge[1]] < 2:
            return True

    # Add edge to the tour
    def add_edge(edge, tour_temp):
        visited[edge[0]] += 1
        visited[edge[1]] += 1
        tour_temp.append(edge)

    for edge in sorted_edges:
        if len(tour_temp)<(n-1):
            if can_add_edge(edge, visited)  and  cycle(edge, tour_temp) == False:
                add_edge(edge, tour_temp)
        else:
            if can_add_edge(edge, visited):
                add_edge(edge, tour_temp)

    tour = []
    tour.append(tour_temp[-1][0])
    del(tour_temp[-1])

    while len(tour)<n:
        for i in range(len(tour_temp)):
            if tour_temp[i][0] == tour[-1]:
                tour.append(tour_temp[i][1])
                del(tour_temp[i])
                break
            if tour_temp[i][1] == tour[-1]:
                tour.append(tour_temp[i][0])
                del(tour_temp[i])
                break

    tour.append(0)

    for i in range(n):
        total_distance += time_distance_matrix[tour[i]][tour[i+1]]

    return tour, total_distance

def calc_t(tour, time_distance_matrix):
    # t represents the arrival time at that node, t is in order of the tour
    # t_b is ordered in a way that arrival time of node 7 is at location 7 in the array
    t = [0 for i in range(len(tour))]
    t_b = [0 for i in range(n)]

    for i in range(len(tour)-2):
        t[i+1] = t[i] + deliveryTime + time_distance_matrix[tour[i]][tour[i+1]]

    t[-1] = t[-2] + time_distance_matrix[tour[-1]][tour[-2]]

    i = 0

    for elem in tour:
        t_b[elem] = t[i]
        i+=1

    return t, t_b

def calcT_direct_served(directServed, time_UAV_distance_matrix, sL, sR):

    totalTime = 0

    for i in directServed:
        totalTime += time_UAV_distance_matrix[0][i]*2 + sL + sR

    return totalTime

def FSTSP(tour, time_distance_matrix, deliveryTime, Cprime, endu, sL, sR):
    # Initialization
    same_subroute = True
    jStar = 0
    iStar = 0
    kStar = 0
    servedByUAV = False

    t, t_b = calc_t(tour, time_distance_matrix)

    Cprime_copy = copy.deepcopy(Cprime)

    TruckSubRoutes = {}
    TruckSubRoutes[1] = [copy.deepcopy(tour), False, [0]]

    maxSavings = 0

    # Calculate savings method from Murray and Chu
    def calcSavings(j, t, time_distance_matrix, tour, TruckSubRoutes, sR):

        tPrime_b = 0
        a = 0
        b = 0
        jPrime = 0
        i = 0
        k = 0

        for w in range(len(tour)):
            if tour[w] == j:
                i = tour[w-1]
                k = tour[w+1]

        savings = time_distance_matrix[i][j] + time_distance_matrix[j][k] - time_distance_matrix[i][k]

        for q in TruckSubRoutes:
            for r in range(len(TruckSubRoutes[q][0])):
                if TruckSubRoutes[q][0][r] == j:
                    if TruckSubRoutes[q][1] == True:
                        a = TruckSubRoutes[q][2][0]
                        b = TruckSubRoutes[q][2][-1]
                        jPrime = TruckSubRoutes[q][2][1]
                        tPrime_b = t[i] - time_distance_matrix[i][j] - time_distance_matrix[j][k] + time_distance_matrix[i][k]

                        savings = min(savings, tPrime_b - (t[a] + time_UAV_distance_matrix[a][jPrime] + time_UAV_distance_matrix[jPrime][b] + sR - deliveryTime))

        if savings > 0:
            return savings
        else:
            return 0

    # Calculates savings for placing a node in a different spot of the tour
    def calcCostTruck(j, t, subroute, savings, time_distance_matrix, maxSavings, endu, servedByUAV, jStar, iStar, kStar):

        a = subroute[0][0]
        b = subroute[0][-1]

        for i in range(len(subroute[0])-1):
            cost = time_distance_matrix[subroute[0][i]][j] + time_distance_matrix[j][subroute[0][i+1]] \
                   - time_distance_matrix[subroute[0][i]][subroute[0][i+1]] + deliveryTime

            if subroute[0][i] != j and subroute[0][i+1] != j:
                if (cost < savings):
                    if (t[b] - t[a] + cost <= endu):
                        if (savings - cost > maxSavings):
                            servedByUAV = False
                            jStar = j
                            iStar = subroute[0][i]
                            kStar = subroute[0][i+1]
                            maxSavings = savings - cost

        return maxSavings, servedByUAV, jStar, iStar, kStar

    # Calculates savings for making a node a UAV delivery node
    def calcCostUAV(j, t, subroute, savings, time_distance_matrix, maxSavings, endu, servedByUAV, jStar, iStar, kStar, same_subroute, tour):
        same_subroute_temp = True
        if j not in subroute[0]:
            same_subroute_temp = False

        t_copy = []
        sub_copy = []
        before_k = False
        before_i = False
        s = 0
        endu_test = []
        endu_test_copy = []
        for i in range(len(subroute[0])-1):
            for k in range(i+1, len(subroute[0])):
                if subroute[0][i] != j and subroute[0][k] != j:
                    if same_subroute_temp == False:
                        t_copy = copy.deepcopy(subroute[0][i:k+1])
                    if same_subroute_temp == True:
                        sub_copy = copy.deepcopy(subroute[0])
                        for p in range(len(sub_copy)):
                            if sub_copy[p] == j:
                                s = p
                                if p<i:
                                    before_k = True
                                    before_i = True
                                    break
                                elif p>i and p<k:
                                    before_i = False
                                    before_k = True
                                    break
                                else:
                                    before_i = False
                                    before_k = False
                                    break
                        del(sub_copy[s])
                        if before_k == True and before_i == True:
                            t_copy = copy.deepcopy(sub_copy[i-1: k])
                        if before_k == True and before_i == False:
                            t_copy = copy.deepcopy(sub_copy[i: k])
                        if before_k == False and before_i == False:
                            t_copy = copy.deepcopy(sub_copy[i: k+1])
                    endu_test, endu_test_copy = calc_t(t_copy, time_distance_matrix)
                    if (time_UAV_distance_matrix[subroute[0][i]][j] + time_UAV_distance_matrix[j][subroute[0][k]] <= endu and
                            endu_test[-1]<= endu):
                        tPrimeK = 0

                        copy_tour = copy.deepcopy(tour)
                        copy_tour.remove(j)
                        tPrime, tPrime_b = calc_t(copy_tour, time_distance_matrix)
                        tPrimeK = tPrime_b[subroute[0][k]]
                        cost =  max(0, max(tPrimeK - t[subroute[0][i]] + sR + sL, time_UAV_distance_matrix[subroute[0][i]][j] + time_UAV_distance_matrix[j][subroute[0][k]]
                                           +sR +sL) - (tPrimeK - t[subroute[0][i]]))

                        if savings - cost > maxSavings:
                            servedByUAV = True
                            jStar = j
                            iStar = subroute[0][i]
                            kStar = subroute[0][k]
                            maxSavings = savings - cost
                            same_subroute = same_subroute_temp

        return maxSavings, servedByUAV, jStar, iStar, kStar, same_subroute

    # Updates all necessary structures
    def performUpdate(servedByUAV, jStar, iStar, kStar, tour, TruckSubRoutes, Cprime_copy, t, t_b):

        # jStar becomes new drone destination and launch/retrieve (iStar/jStar) nodes are on the same subroute as jStar
        if servedByUAV == True and same_subroute == True:
            temp_a = []
            d = 1
            done = False

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0]) - 1):
                    if TruckSubRoutes[i][0][j] == jStar:
                        del(TruckSubRoutes[i][0][j])
                        break

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0]) - 1):
                    if done == False:
                        if TruckSubRoutes[i][0][j] == iStar:
                            d = i
                            temp_a.append(TruckSubRoutes[i][0][j])
                            while len(TruckSubRoutes[i][0]) > j+1:
                                if TruckSubRoutes[i][0][j+1] != jStar:
                                    temp_a.append(TruckSubRoutes[i][0][j+1])
                                del(TruckSubRoutes[i][0][j+1])
                            done = True

            q = 0
            temp_b = []
            for i in range(len(temp_a)):
                temp_b.append(temp_a[i])
                if temp_a[i] == kStar and i != (len(temp_a)-1):
                    q = i
                    break

            temp_c = []
            if q != 0:
                for i in range(q, len(temp_a)):
                    temp_c.append(temp_a[i])


            for i in range(len(tour)):
                if tour[i] == jStar:
                    del(tour[i])
                    break

            updated = False
            TruckSubRoutes_copy = copy.deepcopy(TruckSubRoutes)

            if len(TruckSubRoutes[d][0]) > 1 and len(temp_c)>0:
                for i in range(d+1, len(TruckSubRoutes)+1):
                    TruckSubRoutes[i+2] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d+1] = [temp_b, True, [iStar, jStar, kStar]]
                TruckSubRoutes[d+2] = [temp_c, False, [0]]
                updated = True

            if len(TruckSubRoutes[d][0]) == 1 and len(temp_c) > 0 and updated == False:
                del(TruckSubRoutes[d])
                for i in range(d+1, len(TruckSubRoutes)+2):
                    TruckSubRoutes[i+1] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d] = [temp_b, True, [iStar, jStar, kStar]]
                TruckSubRoutes[d+1] = [temp_c, False, [0]]
                updated = True

            if len(TruckSubRoutes[d][0]) == 1 and len(temp_c) == 0 and updated == False:
                del(TruckSubRoutes[d])
                TruckSubRoutes[d] = [temp_b, True, [iStar, jStar, kStar]]
                updated = True

            if len(TruckSubRoutes[d][0]) > 1 and len(temp_c) == 0 and updated == False:
                for i in range(d+1, len(TruckSubRoutes)+1):
                    TruckSubRoutes[i+1] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d+1] = [temp_b, True, [iStar, jStar, kStar]]
                updated = True

            i_j_k = {iStar,jStar,kStar}

            Cprime_copy = [ele for ele in Cprime_copy if ele not in i_j_k]

            t, t_b = calc_t(tour, time_distance_matrix)

        # jStar becomes new drone destination and launch/retrieve (iStar/kStar) nodes are NOT on the same subroute as jStar
        if servedByUAV == True and same_subroute == False:
            temp_a = []
            d = 1
            done = False

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0]) - 1):
                    if TruckSubRoutes[i][0][j] == jStar:
                        del(TruckSubRoutes[i][0][j])
                        break

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0]) - 1):
                    if done == False:
                        if TruckSubRoutes[i][0][j] == iStar:
                            d = i
                            temp_a.append(TruckSubRoutes[i][0][j])
                            while len(TruckSubRoutes[i][0]) > j+1:
                                temp_a.append(TruckSubRoutes[i][0][j+1])
                                del(TruckSubRoutes[i][0][j+1])
                            done = True
                            break

            q = 0

            temp_b = []
            for i in range(len(temp_a)):
                temp_b.append(temp_a[i])
                if temp_a[i] == kStar and i != (len(temp_a)-1):
                    q = i
                    break

            temp_c = []
            if q != 0:
                for i in range(q, len(temp_a)):
                    if temp_a[i] != jStar:
                        temp_c.append(temp_a[i])

            for i in range(len(tour)):
                if tour[i] == jStar:
                    del(tour[i])
                    break

            updated = False
            TruckSubRoutes_copy = copy.deepcopy(TruckSubRoutes)

            if len(TruckSubRoutes[d][0]) > 1 and len(temp_c) > 0:
                for i in range(d+1, len(TruckSubRoutes)+1):
                    TruckSubRoutes[i+2] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d+1] = [temp_b, True, [iStar, jStar, kStar]]
                TruckSubRoutes[d+2] = [temp_c, False, [0]]
                updated = True

            if len(TruckSubRoutes[d][0]) == 1 and len(temp_c) > 0 and updated == False:
                del(TruckSubRoutes[d])
                for i in range(d+1, len(TruckSubRoutes)+2):
                    TruckSubRoutes[i+1] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d] = [temp_b, True, [iStar, jStar, kStar]]
                TruckSubRoutes[d+1] = [temp_c, False, [0]]
                updated = True

            if len(TruckSubRoutes[d][0]) == 1 and len(temp_c) == 0 and updated == False:
                del(TruckSubRoutes[d])
                TruckSubRoutes[d] = [temp_b, True, [iStar, jStar, kStar]]
                updated = True

            if len(TruckSubRoutes[d][0]) > 1 and len(temp_c) == 0 and updated == False:
                for i in range(d+1, len(TruckSubRoutes)+1):
                    TruckSubRoutes[i+1] = TruckSubRoutes_copy[i]
                TruckSubRoutes[d+1] = [temp_b, True, [iStar, jStar, kStar]]
                updated = True
            i_j_k = {iStar,jStar,kStar}
            Cprime_copy = [ele for ele in Cprime_copy if ele not in i_j_k]

            t, t_b = calc_t(tour, time_distance_matrix)

        # jStar is removed from its current position and placed between iStar and kStar
        elif servedByUAV == False:

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0])):
                    if TruckSubRoutes[i][0][j] == jStar:
                        del(TruckSubRoutes[i][0][j])
                        break

            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0])):
                    if TruckSubRoutes[i][0][j] == iStar:
                        TruckSubRoutes[i][0] = TruckSubRoutes[i][0][0:j+1] + [jStar] + TruckSubRoutes[i][0][j+1:]

            for i in range(len(tour)):
                if tour[i] == jStar:
                    del tour[i]
                    break

            for i in range(len(tour)):
                if tour[i] == iStar:
                    tour = tour[0:i+1] + [jStar] + tour[i+1:]
                    break

            t, t_b = calc_t(tour, time_distance_matrix)

        return t, t_b, tour, Cprime_copy, TruckSubRoutes


    # Main FSTSP algorith
    stop = False
    while stop == False:

        for j in Cprime_copy:
            savings = calcSavings(j, t_b, time_distance_matrix, tour, TruckSubRoutes, sR)
            for k in TruckSubRoutes:
                if TruckSubRoutes[k][1] == True:
                    maxSavings, servedByUAV, jStar, iStar, kStar = calcCostTruck(j, t_b, TruckSubRoutes[k], savings, time_distance_matrix, maxSavings, endu, servedByUAV, jStar, iStar, kStar)
                else:
                    maxSavings, servedByUAV, jStar, iStar, kStar, same_subroute_b = calcCostUAV(j , t_b, TruckSubRoutes[k], savings, time_distance_matrix, maxSavings, endu, servedByUAV, jStar, iStar, kStar, same_subroute, tour)

        if maxSavings > 0.00001:
            t, t_b, tour, Cprime_copy, TruckSubRoutes = performUpdate(servedByUAV, jStar, iStar, kStar, tour, TruckSubRoutes, Cprime_copy, t, t_b)
            maxSavings = 0

        else:
            stop = True

    return tour, TruckSubRoutes, t, t_b

def local_search_drones(TruckSubRoutes, tour_final, directServed, time_distance_matrix, inUAVrange, Cprime, list_launch_dest_retrieve, save_coords):

    savings_direct = {}
    temp = 0

    for i in tour_final:
        if i not in list_launch_dest_retrieve and i in inUAVrange and i in Cprime:
            for j in range(len(tour_final)):
                if tour_final[j] == i:
                    temp = j
            savings_direct[i] = time_distance_matrix[tour_final[temp-1]][tour_final[temp]] + time_distance_matrix[tour_final[temp]][tour_final[temp+1]] - \
                                time_distance_matrix[tour_final[temp-1]][tour_final[temp+1]] + deliveryTime
    if savings_direct != {}:
        savings_direct = sorted(savings_direct.items(), key=lambda x:x[1], reverse=True)
        savings_direct = dict(savings_direct)

        if savings_direct[list(savings_direct.keys())[0]] > 0:
            save_coords[list(savings_direct.keys())[0]] = node_coords[list(savings_direct.keys())[0]]
            new_drone_loc = list(savings_direct.keys())[0]
            directServed.append(new_drone_loc)

            save_a = 0
            save_b = 0
            for i in TruckSubRoutes:
                for j in range(len(TruckSubRoutes[i][0])):
                    if TruckSubRoutes[i][0][j] == new_drone_loc:
                        save_a = i
                        save_b = j
                        break

            del(TruckSubRoutes[save_a][0][save_b])

            for i in range(len(tour_final)):
                if tour_final[i] == new_drone_loc:
                    del(tour_final[i])
                    break

    return tour_final, TruckSubRoutes, directServed, save_coords

def get_launch_dest_retrieve_nodes(TruckSubRoutes):

    list_launch_dest_retrieve = []

    for i in TruckSubRoutes:
        if TruckSubRoutes[i][1] == True:
            list_launch_dest_retrieve.append(TruckSubRoutes[i][2][0])
            list_launch_dest_retrieve.append(TruckSubRoutes[i][2][1])
            list_launch_dest_retrieve.append(TruckSubRoutes[i][2][2])

    return list_launch_dest_retrieve

def twoOpt_b(tour_final, time_distance_matrix):

    def OptCost(time_distance_matrix, a, b, c, d):
        return time_distance_matrix[a][c] + time_distance_matrix[b][d] - time_distance_matrix[a][b] - time_distance_matrix[c][d]

    improved = True
    copy_tour = copy.deepcopy(tour_final)

    while improved == True:
        improved = False
        for i in range(1, len(tour_final)-2):
            for j in range(i+1, len(tour_final)):
                if j - i == 1: continue
                if OptCost(time_distance_matrix, copy_tour[i-1], copy_tour[i], copy_tour[j-1], copy_tour[j]) < -0.000001:
                    copy_tour[i:j] = copy_tour[j-1: i-1: -1]
                    improved = True
        tour_final = copy_tour

    return tour_final


def twoOpt(TruckSubRoutes, tour_final, time_distance_matrix, list_launch_retrieve, endu):

    def OptCost(time_distance_matrix, a, b, c, d):
        return time_distance_matrix[a][c] + time_distance_matrix[b][d] - time_distance_matrix[a][b] - time_distance_matrix[c][d]

    improved = True
    iteration = 0
    maxIteration = 500

    while improved == True and iteration < maxIteration:
        iteration += 1
        improved = False
        for i in range(1, len(tour_final)-2):
            for j in range(i+1, len(tour_final)):
                if j - i == 1: continue
                if OptCost(time_distance_matrix, tour_final[i-1], tour_final[i], tour_final[j-1], tour_final[j]) < -0.000001:
                    copy_tour = copy.deepcopy(tour_final)
                    copy_tour[i:j] = copy_tour[j-1: i-1: -1]
                    feasible = True
                    for t in copy_tour[i+1:j]:
                        if t in list_launch_retrieve:
                            a = 0
                            b = 0
                            c = 0
                            d = 0
                            endu_test = []
                            endu_test_copy = []
                            for k in TruckSubRoutes:
                                if TruckSubRoutes[k][1] == True:
                                    if TruckSubRoutes[k][2][0] == t:
                                        a = TruckSubRoutes[k][2][0]
                                        b = TruckSubRoutes[k][2][2]
                                    if TruckSubRoutes[k][2][2] == t:
                                        a = TruckSubRoutes[k][2][0]
                                        b = TruckSubRoutes[k][2][2]
                            for r in range(len(copy_tour)):
                                if copy_tour[r] == a:
                                    c = r
                                if copy_tour[r] == b:
                                    d = r
                            if d > c:
                                endu_test, endu_test_copy = calc_t(copy_tour[c:d+1], time_distance_matrix)
                            if d < c:
                                endu_test, endu_test_copy = calc_t(copy_tour[d:c+1], time_distance_matrix)
                            if endu_test[-1] > endu:
                                feasible = False

                            order_launch_retrieve = []
                            for z in range(len(copy_tour)):
                                if copy_tour[z] in list_launch_retrieve:
                                    order_launch_retrieve.append(copy_tour[z])

                            for h in range(0, len(list_launch_retrieve), 2):
                                for l in range(len(order_launch_retrieve)-1):
                                    if list_launch_retrieve[h] == order_launch_retrieve[l]:
                                        if list_launch_retrieve[h+1] != order_launch_retrieve[l-1] and list_launch_retrieve[h+1] != order_launch_retrieve[l+1]:
                                            feasible = False
                    if feasible == True:
                        tour_final = copy_tour
                        improved = True
    return tour_final

def graph(TruckSubRoutes, tour_final, directServed, node_coords, save_coords, UAVspeed, endu):

    plt.figure(figsize=(10,6))

    for i in range(len(tour_final)-1):
        plt.plot([node_coords[tour_final[i]][0], node_coords[tour_final[i+1]][0]],[node_coords[tour_final[i]][1], node_coords[tour_final[i+1]][1]], 'r-')

    if TruckSubRoutes != 0:
        for i in TruckSubRoutes:
            if TruckSubRoutes[i][1] == True:
                plt.plot([node_coords[TruckSubRoutes[i][2][0]][0], node_coords[TruckSubRoutes[i][2][1]][0]],
                        [node_coords[TruckSubRoutes[i][2][0]][1], node_coords[TruckSubRoutes[i][2][1]][1]], 'k--')
                plt.plot([node_coords[TruckSubRoutes[i][2][1]][0], node_coords[TruckSubRoutes[i][2][2]][0]],
                        [node_coords[TruckSubRoutes[i][2][1]][1], node_coords[TruckSubRoutes[i][2][2]][1]], 'k--')


    for i in directServed:
        plt.plot([save_coords[i][0], node_coords[0][0]],[save_coords[i][1], node_coords[0][1]], 'b--')
        x = save_coords[i][0]
        y = save_coords[i][1]
        plt.scatter(x, y, color = 'y')


    for i in node_coords:
        x = node_coords[i][0]
        y = node_coords[i][1]
        plt.scatter(x, y, color = 'y')

    plt.scatter(node_coords[0][0], node_coords[0][1], color = 'k', s=100, zorder=100)
    plt.text(node_coords[0][0], node_coords[0][1]-800, 'Depot', fontsize=14, ha='center', va='bottom', zorder=100, fontweight='bold')

    ax = plt.gca()
    ax.set_xlim([0, 15000])
    ax.set_ylim([0, 15000])
    ax.set(xlabel=r'X Coordinate', ylabel=r'Y Coordinate')
    rangeMetersUAV = (UAVspeed*endu*1000)/2

    plt.title('Route after DDTSP algorithm')

    UAVrange = plt.Circle((node_coords[0][0], node_coords[0][1]), rangeMetersUAV, color='g', fill=False)
    ax.add_patch(UAVrange)
    ax.legend([UAVrange], ['UAV range'])


    plt.show()

# Plot original coordinates
#graph(0, [], [], node_coords, [], UAVspeed, endu)

# Nearest neighbor TSP algorithm
begin_time_nearN = time.time()
tour_nearN_b, total_time_nearN_b = solve_tsp_nearest(time_distance_matrix)
end_time_nearN = time.time()
running_time_nearN = end_time_nearN - begin_time_nearN
total_time_nearN_b = total_time_nearN_b + deliveryTime*(n-1)

# Two opt optimization
#graph(0, tour_nearN_b, [], node_coords, [], UAVspeed, endu)
tour_nearN_twoOpt = twoOpt_b(tour_nearN_b, time_distance_matrix)
#graph(0, tour_nearN_twoOpt, [], node_coords, [], UAVspeed, endu)
time_nearN_twoOpt, dummy = calc_t(tour_nearN_twoOpt, time_distance_matrix)
total_time_nearN_twoOpt = time_nearN_twoOpt[-1]

# Sweep TSP algorithm
begin_time_sweep = time.time()
tour_sweep_b, total_time_sweep_b = solve_tsp_sweep(time_distance_matrix, node_coords, n)
end_time_sweep = time.time()
running_time_sweep = end_time_sweep - begin_time_sweep
total_time_sweep_b = total_time_sweep_b + deliveryTime*(n-1)

# Two opt optimization
#graph(0, tour_sweep_b, [], node_coords, [], UAVspeed, endu)
tour_sweep_twoOpt = twoOpt_b(tour_sweep_b, time_distance_matrix)
#graph(0, tour_sweep_twoOpt, [], node_coords, [], UAVspeed, endu)
time_sweep_twoOpt, dummy = calc_t(tour_sweep_twoOpt, time_distance_matrix)
total_time_sweep_twoOpt = time_sweep_twoOpt[-1]

# Savings TSP algorithm
begin_time_savings = time.time()
tour_savings_b, total_time_savings_b = solve_tsp_savings(time_distance_matrix, n)
end_time_savings = time.time()
running_time_savings = end_time_savings - begin_time_savings
total_time_savings_b = total_time_savings_b + deliveryTime*(n-1)

# Two opt optimization
#graph(0, tour_savings_b, [], node_coords, [], UAVspeed, endu)
tour_savings_twoOpt = twoOpt_b(tour_savings_b, time_distance_matrix)
#graph(0, tour_savings_twoOpt, [], node_coords, [], UAVspeed, endu)
time_savings_twoOpt, dummy = calc_t(tour_savings_twoOpt, time_distance_matrix)
total_time_savings_twoOpt = time_savings_twoOpt[-1]

threshold_time = min(total_time_nearN_b,total_time_savings_b,total_time_sweep_b)



# Greedy approach in order to compare
start_time_greedy = time.time()
dummy = []
direct_served_greedy = []
for i in Cprime:
    if time_UAV_distance_matrix[0][i]*2 <= endu:
        direct_served_greedy.append(i)

droneTime_greedy = calcT_direct_served(direct_served_greedy, time_UAV_distance_matrix, sL, sR)
end_time_greedy = time.time()

greedy_time_savings = end_time_greedy - start_time_greedy + running_time_savings
greedy_time_nearN = end_time_greedy - start_time_greedy + running_time_nearN
greedy_time_sweep = end_time_greedy - start_time_greedy + running_time_sweep

# Nearest neighbor TSP algorithm
tour_nearN_greedy = [ele for ele in tour_nearN_b if ele not in direct_served_greedy]
tourTime_nearN_greedy, dummy = calc_t(tour_nearN_greedy, time_distance_matrix)
makespan_nearN_greedy = max(droneTime_greedy, tourTime_nearN_greedy[-1])

# Sweep TSP algorithm
tour_sweep_greedy = [ele for ele in tour_sweep_b if ele not in direct_served_greedy]
tourTime_sweep_greedy, dummy = calc_t(tour_sweep_greedy, time_distance_matrix)
makespan_sweep_greedy = max(droneTime_greedy, tourTime_sweep_greedy[-1])

# Savings TSP algorithm
tour_savings_greedy = [ele for ele in tour_savings_b if ele not in direct_served_greedy]
tourTime_savings_greedy, dummy = calc_t(tour_savings_greedy, time_distance_matrix)
makespan_savings_greedy = max(droneTime_greedy, tourTime_savings_greedy[-1])

# 2-opt for greedy

# Nearest Neighbour
twoOpt_greedy_nearN = twoOpt_b(tour_nearN_greedy, time_distance_matrix)
tourTime_nearN_greedy_twoOpt, dummy = calc_t(twoOpt_greedy_nearN, time_distance_matrix)
makespan_nearN_greedy_twoOpt = max(droneTime_greedy, tourTime_nearN_greedy_twoOpt[-1])

# Sweep
twoOpt_greedy_sweep = twoOpt_b(tour_sweep_greedy, time_distance_matrix)
tourTime_sweep_greedy_twoOpt, dummy = calc_t(twoOpt_greedy_sweep, time_distance_matrix)
makespan_sweep_greedy_twoOpt = max(droneTime_greedy, tourTime_sweep_greedy_twoOpt[-1])

# Savings
twoOpt_greedy_savings = twoOpt_b(tour_savings_greedy, time_distance_matrix)
tourTime_savings_greedy_twoOpt, dummy = calc_t(twoOpt_greedy_savings, time_distance_matrix)
makespan_savings_greedy_twoOpt = max(droneTime_greedy, tourTime_savings_greedy_twoOpt[-1])


# FSTSP for DDTSPgreedy

savingsCprime = [i for i in Cprime if i not in direct_served_greedy]
nearNCprime = [i for i in Cprime if i not in direct_served_greedy]
sweepCprime = [i for i in Cprime if i not in direct_served_greedy]

save_coords_test = {}

for i in direct_served_greedy:
    save_coords_test[i] = copy.deepcopy(node_coords[i])

# Savings
DDTSPgreedy_savings, TruckSubroutes_DDTSPgreedy_savings, t_DDTSPgreedy_savings, dummy = FSTSP(twoOpt_greedy_savings, time_distance_matrix, deliveryTime, savingsCprime, endu, sL, sR)
DDTSPgreedy_makespan_savings = max(t_DDTSPgreedy_savings[-1], droneTime_greedy)

#graph(TruckSubroutes_DDTSPgreedy_savings, DDTSPgreedy_savings, direct_served_greedy, node_coords, save_coords_test, UAVspeed, endu)

# NearN
DDTSPgreedy_NearN, TruckSubroutes_DDTSPgreedy_NearN, t_DDTSPgreedy_NearN, dummy = FSTSP(twoOpt_greedy_nearN, time_distance_matrix, deliveryTime, nearNCprime, endu, sL, sR)
DDTSPgreedy_makespan_NearN = max(t_DDTSPgreedy_NearN[-1], droneTime_greedy)

#graph(TruckSubroutes_DDTSPgreedy_NearN, DDTSPgreedy_NearN, direct_served_greedy, node_coords, save_coords_test, UAVspeed, endu)

# Sweep
DDTSPgreedy_Sweep, TruckSubroutes_DDTSPgreedy_Sweep, t_DDTSPgreedy_Sweep, dummy = FSTSP(twoOpt_greedy_sweep, time_distance_matrix, deliveryTime, sweepCprime, endu, sL, sR)
DDTSPgreedy_makespan_Sweep = max(t_DDTSPgreedy_Sweep[-1], droneTime_greedy)



DDTSP_savings = []
DDTSP_nearN = []
DDTSP_sweep = []

DDTSP_OPTsol_alg = []
DDTSP_OPTsol_value = []

RT_DDTSP_savings = []
RT_DDTSP_nearN = []
RT_DDTSP_sweep = []

percentDiff_greedy_savings = [0 for i in range(iterations)]
percentDiff_greedy_nearN = [0 for i in range(iterations)]
percentDiff_greedy_sweep = [0 for i in range(iterations)]

percentDiff_tsp_savings = [0 for i in range(iterations)]
percentDiff_tsp_nearN = [0 for i in range(iterations)]
percentDiff_tsp_sweep = [0 for i in range(iterations)]

percentDiff_DDTSPgreedy_savings = [0 for i in range(iterations)]
percentDiff_DDTSPgreedy_nearN = [0 for i in range(iterations)]
percentDiff_DDTSPgreedy_sweep = [0 for i in range(iterations)]

copy_Cprime = copy.deepcopy(Cprime)
copy_node_coords = copy.deepcopy(node_coords)
copy_time_distance_matrix = copy.deepcopy(time_distance_matrix)
copy_time_UAV_distance_matrix = copy.deepcopy(time_UAV_distance_matrix)

count = 0

for i in range(iterations):
    count+=1

    Cprime = copy_Cprime
    node_coords = copy_node_coords
    time_distance_matrix = copy_time_distance_matrix
    time_UAV_distance_matrix = copy_time_UAV_distance_matrix


    # DDTSP algorithm

    # First, PDSTSP part of the algorithm
    b_RT_PDSTSP = time.time()
    Cprime, node_coords, time_distance_matrix, directServed, save_coords, inUAVrange = PDSTSP(time_UAV_distance_matrix, time_distance_matrix, Cprime, endu, deliveryTime, sL, sR, threshold_time)
    totalTimeDirect = calcT_direct_served(directServed, time_UAV_distance_matrix, sL, sR)
    e_RT_PDSTSP = time.time()
    total_RT_PDSTSP = e_RT_PDSTSP - b_RT_PDSTSP

    totalTimeDirect_savings = copy.deepcopy(totalTimeDirect)
    totalTimeDirect_nearN = copy.deepcopy(totalTimeDirect)
    totalTimeDirect_sweep = copy.deepcopy(totalTimeDirect)




    # SAVINGS

    # Creating a solution based on the savings algorithm
    print(f"SAVINGS {count}")
    b_RT_savings = time.time()

    # TSP savings heuristic
    tour_savings, total_time_savings = solve_tsp_savings(time_distance_matrix, n)
    tour_savings = [ele for ele in tour_savings if ele not in directServed]

    # 2-opt procedure before introducing drones
    tour_savings = twoOpt_b(tour_savings, time_distance_matrix)

    start_FSTSP_RT_savings = time.time()
    tour_final_savings, TruckSubRoutes_final_savings, t_savings, t_b_savings = FSTSP(tour_savings, time_distance_matrix, deliveryTime, Cprime, endu, sL, sR)
    end_FSTSP_RT_savings = time.time()
    total_FSTSP_RT_savings = end_FSTSP_RT_savings - start_FSTSP_RT_savings
    TruckSubRoutes_final_savings = sorted(TruckSubRoutes_final_savings.items(), key=lambda x:x[0])
    TruckSubRoutes_final_savings = dict(TruckSubRoutes_final_savings)

    list_launch_dest_retrieve_savings = get_launch_dest_retrieve_nodes(TruckSubRoutes_final_savings)
    list_launch_retrieve_savings = []
    for i in range(len(list_launch_dest_retrieve_savings)):
        if i % 3 != 1:
            list_launch_retrieve_savings.append(list_launch_dest_retrieve_savings[i])

    tour_final_savings = twoOpt(TruckSubRoutes_final_savings, tour_final_savings, time_distance_matrix, list_launch_retrieve_savings, endu)

    update = True
    iteration = 0
    maxIteration = n / 2
    directServed_savings = copy.deepcopy(directServed)
    save_coords_savings = copy.deepcopy(save_coords)

    while update == True and iteration < maxIteration:
        iteration += 1
        update = False
        if t_savings[-1] - totalTimeDirect_savings > (endu/2):
            tour_final_savings, TruckSubRoutes_final_savings, directServed_savings, save_coords_savings = local_search_drones(TruckSubRoutes_final_savings, tour_final_savings, directServed_savings, time_distance_matrix, inUAVrange, Cprime, list_launch_dest_retrieve_savings, save_coords_savings)
            totalTimeDirect_savings = calcT_direct_served(directServed_savings, time_UAV_distance_matrix, sL, sR)
            t_savings, t_b_savings  = calc_t(tour_final_savings, time_distance_matrix)
            update = True

    e_RT_savings = time.time()
    T_RT_savings = (e_RT_savings - b_RT_savings) + total_RT_PDSTSP

    RT_DDTSP_savings.append(T_RT_savings)
    DDTSP_savings.append(max(t_savings[-1], totalTimeDirect_savings))

    maxTime_savings = max(t_savings[-1], totalTimeDirect_savings)

    #print(t_savings[-1], totalTimeDirect_savings)

    #graph(TruckSubRoutes_final_savings, tour_final_savings, directServed_savings, node_coords, save_coords_savings, UAVspeed, endu)


    # NEARN

    print(f"NEARN {count}")

    # Creating a solution based on the nearest neighbour algorithm
    b_RT_nearN = time.time()

    # Nearest Neighbour heuristic
    tour_nearN, total_time_nearN = solve_tsp_nearest(time_distance_matrix)
    tour_nearN = [ele for ele in tour_nearN if ele not in directServed]

    # 2-opt procedure before introducing drones
    tour_nearN = twoOpt_b(tour_nearN, time_distance_matrix)

    tour_final_nearN, TruckSubRoutes_final_nearN, t_nearN, t_b_nearN = FSTSP(tour_nearN, time_distance_matrix, deliveryTime, Cprime, endu, sL, sR)
    TruckSubRoutes_final_nearN = sorted(TruckSubRoutes_final_nearN.items(), key=lambda x:x[0])
    TruckSubRoutes_final_nearN = dict(TruckSubRoutes_final_nearN)

    list_launch_dest_retrieve_nearN = get_launch_dest_retrieve_nodes(TruckSubRoutes_final_nearN)
    list_launch_retrieve_nearN = []
    for i in range(len(list_launch_dest_retrieve_nearN)):
        if i % 3 != 1:
            list_launch_retrieve_nearN.append(list_launch_dest_retrieve_nearN[i])

    tour_final_nearN = twoOpt(TruckSubRoutes_final_nearN, tour_final_nearN, time_distance_matrix,list_launch_retrieve_nearN, endu)

    update = True
    iteration = 0
    maxIteration = n / 2
    directServed_nearN = copy.deepcopy(directServed)
    save_coords_nearN = copy.deepcopy(save_coords)

    while update == True and iteration < maxIteration:
        iteration += 1
        update = False
        if t_nearN[-1] - totalTimeDirect_nearN > (endu/2):
            tour_final_nearN, TruckSubRoutes_final_nearN, directServed_nearN, save_coords_nearN = local_search_drones(TruckSubRoutes_final_nearN, tour_final_nearN, directServed_nearN, time_distance_matrix, inUAVrange, Cprime, list_launch_dest_retrieve_nearN, save_coords_nearN)
            totalTimeDirect_nearN = calcT_direct_served(directServed_nearN, time_UAV_distance_matrix, sL, sR)
            t_nearN, t_b_nearN  = calc_t(tour_final_nearN, time_distance_matrix)
            update = True

    e_RT_nearN = time.time()
    T_RT_nearN = (e_RT_nearN - b_RT_nearN) + total_RT_PDSTSP

    RT_DDTSP_nearN.append(T_RT_nearN)
    DDTSP_nearN.append(max(t_nearN[-1], totalTimeDirect_nearN))

    maxTime_nearN = max(t_nearN[-1], totalTimeDirect_nearN)

    #print(t_nearN[-1], totalTimeDirect_nearN)

    #graph(TruckSubRoutes_final_nearN, tour_final_nearN, directServed_nearN, node_coords, save_coords_nearN, UAVspeed, endu)


    # SWEEP

    print(f"SWEEP {count}")

    # Creating a solution based on the sweep algorithm
    b_RT_sweep = time.time()

    # TSP sweep heuristic
    tour_sweep, total_time_sweep = solve_tsp_sweep(time_distance_matrix, node_coords, n)
    tour_sweep = [ele for ele in tour_sweep if ele not in directServed]

    # 2-opt procedure before introducing drones
    tour_sweep = twoOpt_b(tour_sweep, time_distance_matrix)

    tour_final_sweep, TruckSubRoutes_final_sweep, t_sweep, t_b_sweep = FSTSP(tour_sweep, time_distance_matrix, deliveryTime, Cprime, endu, sL, sR)
    TruckSubRoutes_final_sweep = sorted(TruckSubRoutes_final_sweep.items(), key=lambda x:x[0])
    TruckSubRoutes_final_sweep = dict(TruckSubRoutes_final_sweep)

    list_launch_dest_retrieve_sweep = get_launch_dest_retrieve_nodes(TruckSubRoutes_final_sweep)
    list_launch_retrieve_sweep = []
    for i in range(len(list_launch_dest_retrieve_sweep)):
        if i % 3 != 1:
            list_launch_retrieve_sweep.append(list_launch_dest_retrieve_sweep[i])

    tour_final_sweep = twoOpt(TruckSubRoutes_final_sweep, tour_final_sweep, time_distance_matrix,list_launch_retrieve_sweep, endu)

    update = True
    iteration = 0
    maxIteration = n / 2
    directServed_sweep = copy.deepcopy(directServed)
    save_coords_sweep = copy.deepcopy(save_coords)

    while update == True and iteration < maxIteration:
        iteration += 1
        update = False
        if t_sweep[-1] - totalTimeDirect_sweep > (endu/2):
            tour_final_sweep, TruckSubRoutes_final_sweep, directServed_sweep, save_coords_sweep = local_search_drones(TruckSubRoutes_final_sweep, tour_final_sweep, directServed_sweep, time_distance_matrix, inUAVrange, Cprime, list_launch_dest_retrieve_sweep, save_coords_sweep)
            totalTimeDirect_sweep = calcT_direct_served(directServed_sweep, time_UAV_distance_matrix, sL, sR)
            t_sweep, t_b_sweep  = calc_t(tour_final_sweep, time_distance_matrix)
            update = True

    e_RT_sweep = time.time()
    T_RT_sweep = (e_RT_sweep - b_RT_sweep) + total_RT_PDSTSP

    RT_DDTSP_sweep.append(T_RT_sweep)
    DDTSP_sweep.append(max(t_sweep[-1], totalTimeDirect_sweep))

    maxTime_sweep = max(t_sweep[-1], totalTimeDirect_sweep)

    #print(t_sweep[-1], totalTimeDirect_sweep)

    #graph(TruckSubRoutes_final_sweep, tour_final_sweep, directServed_sweep, node_coords, save_coords_sweep, UAVspeed, endu)

    if min(maxTime_sweep, maxTime_nearN, maxTime_savings) == maxTime_savings:
        minMakespan = maxTime_savings
        best_Sol_ALg = 'Savings'
    if min(maxTime_sweep, maxTime_nearN, maxTime_savings) == maxTime_nearN:
        minMakespan = maxTime_nearN
        best_Sol_ALg = 'Nearest Neighbour'
    else:
        minMakespan = maxTime_sweep
        best_Sol_ALg = 'Sweep'

    DDTSP_OPTsol_value.append(minMakespan)
    DDTSP_OPTsol_alg.append(best_Sol_ALg)

    print("")

for i in range(iterations):

    #percentDiff_greedy_savings[i] = ((DDTSP_savings[i] - makespan_savings_greedy_twoOpt)/makespan_savings_greedy_twoOpt)*100
    percentDiff_greedy_savings[i] = ((makespan_savings_greedy_twoOpt - DDTSP_savings[i])/DDTSP_savings[i])*100
    #percentDiff_tsp_savings[i] = ((DDTSP_savings[i] - total_time_savings_twoOpt)/total_time_savings_twoOpt)*100
    percentDiff_tsp_savings[i] = ((total_time_savings_twoOpt - DDTSP_savings[i])/DDTSP_savings[i])*100
    percentDiff_DDTSPgreedy_savings[i] = ((DDTSPgreedy_makespan_savings - DDTSP_savings[i])/DDTSP_savings[i])*100


    #percentDiff_greedy_nearN[i] = ((DDTSP_nearN[i] - makespan_nearN_greedy_twoOpt)/makespan_nearN_greedy_twoOpt)*100
    percentDiff_greedy_nearN[i] = ((makespan_nearN_greedy_twoOpt - DDTSP_nearN[i])/DDTSP_nearN[i])*100
    #percentDiff_tsp_nearN[i] = ((DDTSP_nearN[i] - total_time_nearN_twoOpt)/total_time_nearN_twoOpt)*100
    percentDiff_tsp_nearN[i] = ((total_time_nearN_twoOpt - DDTSP_nearN[i])/DDTSP_nearN[i])*100
    percentDiff_DDTSPgreedy_nearN[i] = ((DDTSPgreedy_makespan_NearN - DDTSP_nearN[i])/DDTSP_nearN[i])*100

    #percentDiff_greedy_sweep[i] = ((DDTSP_sweep[i] - makespan_sweep_greedy_twoOpt)/makespan_sweep_greedy_twoOpt)*100
    percentDiff_greedy_sweep[i] = ((makespan_sweep_greedy_twoOpt - DDTSP_sweep[i])/DDTSP_sweep[i])*100
    #percentDiff_tsp_sweep[i] = ((DDTSP_sweep[i] - total_time_sweep_twoOpt)/total_time_sweep_twoOpt)*100
    percentDiff_tsp_sweep[i] = ((total_time_sweep_twoOpt - DDTSP_sweep[i])/DDTSP_sweep[i])*100
    percentDiff_DDTSPgreedy_sweep[i] = ((DDTSPgreedy_makespan_Sweep - DDTSP_sweep[i])/DDTSP_sweep[i])*100


print("SAVINGS: \n")
print(f"Optimal values DDTSP (savings): {DDTSP_savings}")
print(f"Mean: {statistics.mean(DDTSP_savings)}")
print(f"SD: {statistics.stdev(DDTSP_savings)}\n")

print(f"Running times DDTSP (savings): {RT_DDTSP_savings}")
print(f"Mean: {statistics.mean(RT_DDTSP_savings)}")
print(f"SD: {statistics.stdev(RT_DDTSP_savings)}\n")

print(f"Optimal values greedy (savings): {makespan_savings_greedy_twoOpt}")
print(f"Percentage difference with DDTSP (savings): {percentDiff_greedy_savings}")
print(f"Mean: {statistics.mean(percentDiff_greedy_savings)}\n")

print(f"Optimal values tsp with two opt (savings): {total_time_savings_twoOpt}")
print(f"Percentage difference with DDTSP (savings): {percentDiff_tsp_savings}")
print(f"Mean: {statistics.mean(percentDiff_tsp_savings)}\n")

print(f"Optimal values DDTSPgreedy (savings): {DDTSPgreedy_makespan_savings}")
print(f"Percentage difference with DDTSP (savings): {percentDiff_DDTSPgreedy_savings}")
print(f"Mean: {statistics.mean(percentDiff_DDTSPgreedy_savings)}\n\n\n")


print("NEAREST NEIGHBOUR: \n")
print(f"Optimal values DDTSP (nearN): {DDTSP_nearN}")
print(f"Mean: {statistics.mean(DDTSP_nearN)}")
print(f"SD: {statistics.stdev(DDTSP_nearN)}\n")

print(f"Running times DDTSP (nearN): {RT_DDTSP_nearN}")
print(f"Mean: {statistics.mean(RT_DDTSP_nearN)}")
print(f"SD: {statistics.stdev(RT_DDTSP_nearN)}\n")

print(f"Optimal values greedy (nearN): {makespan_nearN_greedy_twoOpt}")
print(f"Percentage difference with DDTSP (nearN): {percentDiff_greedy_nearN}")
print(f"Mean: {statistics.mean(percentDiff_greedy_nearN)}\n")

print(f"Optimal values tsp with two opt (nearN): {total_time_nearN_twoOpt}")
print(f"Percentage difference with DDTSP (nearN): {percentDiff_tsp_nearN}")
print(f"Mean: {statistics.mean(percentDiff_tsp_nearN)}\n")

print(f"Optimal values DDTSPgreedy (nearN): {DDTSPgreedy_makespan_NearN}")
print(f"Percentage difference with DDTSP (nearN): {percentDiff_DDTSPgreedy_nearN}")
print(f"Mean: {statistics.mean(percentDiff_DDTSPgreedy_nearN)}\n\n\n")


print("SWEEP: \n")
print(f"Optimal values DDTSP (sweep): {DDTSP_sweep}")
print(f"Mean: {statistics.mean(DDTSP_sweep)}")
print(f"SD: {statistics.stdev(DDTSP_sweep)}\n")

print(f"Running times DDTSP (sweep): {RT_DDTSP_sweep}")
print(f"Mean: {statistics.mean(RT_DDTSP_sweep)}")
print(f"SD: {statistics.stdev(RT_DDTSP_sweep)}\n")

print(f"Optimal values greedy (sweep): {makespan_sweep_greedy_twoOpt}")
print(f"Percentage difference with DDTSP (sweep): {percentDiff_greedy_sweep}")
print(f"Mean: {statistics.mean(percentDiff_greedy_sweep)}\n")

print(f"Optimal values tsp with two opt (sweep): {total_time_sweep_twoOpt}")
print(f"Percentage difference with DDTSP (sweep): {percentDiff_tsp_sweep}")
print(f"Mean: {statistics.mean(percentDiff_tsp_sweep)}\n")

print(f"Optimal values DDTSPgreedy (sweep): {DDTSPgreedy_makespan_Sweep}")
print(f"Percentage difference with DDTSP (sweep): {percentDiff_DDTSPgreedy_sweep}")
print(f"Mean: {statistics.mean(percentDiff_DDTSPgreedy_sweep)}")


