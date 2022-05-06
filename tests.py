#########################################################
#
#                   Imports
#
#########################################################

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import pickle

import hilbert_map as hm
import util
from MCLlikelihood_test import *

#########################################################
#
#                   Model/data Setup
#
#########################################################


# Variables
map_name = "freiburg_campus"
train_percentage = 0.1
components = 100
gamma = 1.0
distance_cutoff = 0.001

# Setting up file namespaces
logfile = "datasets/" + map_name + "_corrected.log"
classifier_name = "classifiers/" + map_name + ".pkl"

# Load data and split it into training and testing data
train_data, test_data = util.create_test_train_split(logfile, train_percentage)

full_train_data = util.create_test_data(logfile)


# Extract poses and scans from the data
poses = full_train_data["poses"]
scans = full_train_data["scans"]

# Limits in metric space based on poses with a 2m buffer zone
xlim, ylim = util.bounding_box(poses, 2.0)

# Attempt to load classifier model, otherwise perform training
try:
    with open(classifier_name, 'rb') as inp:
        model = pickle.load(inp)
except:
    # Sampling locations distributed in an even grid over the area
    centers = util.sampling_coordinates(xlim, ylim, components)

    # Create model
    model = hm.SparseHilbertMap(centers, gamma, distance_cutoff)

    time1 = time.time()
    count = 0

    for data, label in util.data_generator(poses, scans):
        model.add(data, label)

        sys.stdout.write("\rTraining model: {: 6.2f}%".format(count / float(len(poses)-1) * 100))
        sys.stdout.flush()
        count += 1
    print("")

    time2 = time.time()
    dt = (time2 - time1)
    print("Training full data points took " + str(dt) + " seconds.")

    with open(classifier_name, 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
"""
pose1 = np.array([[0, 0, 0]])
scan1 = np.zeros([1,360])
scan1[0][0] = 10.0
scan1[0][179] = 10.0

for data, label in util.data_generator(pose1, scan1):
    model.add(data, label)

point1 = np.array([[0, -10]])
point2 = np.array([[0, 10]])
point3 = np.array([[0, -10],[0, 10], [0,0]])
print(model.classify(point1))
print(model.classify(point2))
print(model.classify(point3))
"""


######################################
# Use this model for localization now!
######################################


#########################################################
#
#                   Benchmarks
#
#########################################################

"""
query1 = np.random.rand(10,2) * 20
query2 = np.random.rand(10,2) * 20
query3 = np.random.rand(20,2) * 20
time1 = time.time()
model.classify(query1)
model.classify(query2)
time2 = time.time()
dt1 = time2 - time1
time1 = time.time()
model.classify(query3)
time2 = time.time()
dt2 = time2 - time1
print("Sequential classification took: " + str(dt1) + "s, while batch took: " + str(dt2) + "s.")
"""
"""
times = []
for i in range(500):
    query_list = np.random.rand(i+1,2) * 20
    time1 = time.time()
    model.classify(query_list)
    time2 = time.time()
    dt = (time2 - time1)
    print(str(i+1) + " number of points took " + str(dt) + " seconds.")
    times.append(dt)

plt.plot(times)
plt.ylabel('time [s]')
plt.show()
"""

# Test map

full_x = []
full_y = []
ground_x = []
ground_y = []
for line in open(logfile):
    x_list = []
    y_list = []
    ground_truth = []
    if line.startswith("FLASER"):
        arr = line.split()
        count = int(arr[1])
        scans = [float(v) for v in arr[2:2+count]]
        ground_truth = [float(v) for v in arr[-9:-6]]
        angle_increment = math.pi / len(scans)

        for i in range(len(scans)):
            if scans[i] < 40:
                heading = ground_truth[2] + i * angle_increment - (math.pi / 2.0)
                x = ground_truth[0] + math.cos(heading) * scans[i]
                y = ground_truth[1] + math.sin(heading) * scans[i]
                full_x.append(x)
                full_y.append(y)
                ground_x.append(ground_truth[0])
                ground_y.append(ground_truth[1])
    
plt.clf()
plt.scatter(full_x,full_y)
plt.scatter(ground_x, ground_y)
plt.pause(0.1)
plt.show()


util.generate_map(
            model,
            0.1,
            [xlim[0], xlim[1], ylim[0], ylim[1]],
            "hilbert_map.png"
    )
    

#########################################################
#
#                   Unit tests
#
#########################################################

"""
mcl = MCL(xlim, ylim, 4, model, [0, 0, 0])
print(str([o.weight for o in mcl.particles]))
ESS = mcl.calculate_ESS()
print(str(ESS))
for i in mcl.particles:
    i.weight = 0
mcl.particles[0].weight = 0.1
mcl.particles[1].weight = 0.7
mcl.particles[2].weight = 0.1
mcl.particles[3].weight = 0.1
print(str([o.weight for o in mcl.particles]))
mcl.normalize_weights()
print(str([o.weight for o in mcl.particles]))
ESS = mcl.calculate_ESS()
print(str(ESS))
if ESS < mcl.nbr_particles/2:
    mcl.resample()

"""
"""
test1 = np.array([[-2.195345,-4.11522]])
test2 = np.array([[-2.56547,-3.8573]])
print(model.classify(test1))
print(model.classify(test2))
"""


#########################################################
#
#                   Initialize filter
#
#########################################################
"""
fractions = []
space = np.linspace(0.001,2,100)
for i in space:
    mcl = MCL(xlim, ylim, 5, model, [0,0,0])
    mcl.tn_std = math.sqrt(i)
    mcl.a, mcl.b = (mcl.a_clip - mcl.tn_mean) / mcl.tn_std, (mcl.b_clip - mcl.tn_mean) / mcl.tn_std
    mcl.simulate(logfile, False)
    fractions.append(mcl.frac)
    print("Im alive!")
    """
#########################################################
#
#                   Data Plotting
#
#########################################################
"""
plt.clf()
plt.plot(space,fractions)
plt.show()
"""
"""
plt.clf()
plt.scatter(mcl.query_x, mcl.query_y)
plt.scatter(train_x, train_y)
plt.show()
print("Euclidean error sum: " + str(np.sum(mcl.euc_error)))
"""

