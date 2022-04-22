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

import hilbert_map as hm
import util
from MCLClass import *
from util_changed import create_test_data, generate_map

#########################################################
#
#                   Model/data Setup
#
#########################################################


# Variables
logfile = "belgioioso.gfs.log"
train_percentage = 0.1
components = 100
gamma = 1.0
distance_cutoff = 0.001

# Load data and split it into training and testing data
train_data, test_data = util.create_test_train_split(logfile, train_percentage)

full_train_data = create_test_data(logfile)


# Extract poses and scans from the data
poses = full_train_data["poses"]
scans = full_train_data["scans"]

 # Limits in metric space based on poses with a 10m buffer zone
xlim, ylim = util.bounding_box(poses, 10.0)

# Sampling locations distributed in an even grid over the area
centers = util.sampling_coordinates(xlim, ylim, components)

# Create model
model = hm.SparseHilbertMap(centers, gamma, distance_cutoff)

time1 = time.time()

#########################################################
#
#                   Train model
#
#########################################################
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



######################################
# Use this model for localization now!
######################################

# Extract poses and scans from the data
poses2 = test_data["poses"]
scans2 = test_data["scans"]


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
"""
for line in open(logfile):
    x_list = []
    y_list = []
    ground_truth = []
    if line.startswith("FLASER"):
        arr = line.split()
        count = int(arr[1])
        scans = [float(v) for v in arr[2:2+count]]
        ground_truth = [float(v) for v in arr[-9:-6]]
        plt.clf()
        for i in range(len(scans)):
            heading = (ground_truth[2] + (i * 2 * math.pi / len(scans))) % (2 * math.pi)
            x = ground_truth[0] + math.cos(heading) * scans[i]
            y = ground_truth[1] + math.sin(heading) * scans[i]
            x_list.append(x)
            y_list.append(y)
    
        plt.clf()
        plt.scatter(x_list,y_list)
        plt.scatter(ground_truth[0], ground_truth[1])
        plt.pause(0.1)
    plt.show()
    """
"""
generate_map(
            model,
            0.1,
            [xlim[0], xlim[1], ylim[0], ylim[1]],
            "hilbert_map.png"
    )
"""

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

mcl = MCL(xlim, ylim, 50, model, [0,0,0])
mcl.simulate(logfile, False)


print("Euclidean error sum: " + str(np.sum(mcl.euc_error)))

"""
plt.plot(mcl.euc_error)
plt.ylabel('Error [m]')
plt.show()
"""


plt.plot(mcl.x_path,mcl.y_path)
plt.plot(mcl.x_odom, mcl.y_odom)
plt.plot(mcl.x_truth,mcl.y_truth)
plt.show()



