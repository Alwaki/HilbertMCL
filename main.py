#Test running shit
import argparse
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

import hilbert_map as hm
import util
from MCLClass import *

####################################
# Create a model!
###################################


# Variables
logfile = "belgioioso.gfs.log"
train_percentage = 0.1
components = 100
gamma = 1.0
distance_cutoff = 0.001

# Load data and split it into training and testing data
train_data, test_data = util.create_test_train_split(logfile, train_percentage)

full_train_data = util.create_test_data(logfile)


# Extract poses and scans from the data
poses = full_train_data["poses"]
scans = full_train_data["scans"]

 # Limits in metric space based on poses with a 10m buffer zone
xlim, ylim = util.bounding_box(poses, 10.0)

# Sampling locations distributed in an even grid over the area
centers = util.sampling_coordinates(xlim, ylim, components)

# Create model
model = hm.SparseHilbertMap(centers, gamma, distance_cutoff)

 # Train the model with the data
count = 0


for data, label in util.data_generator(poses, scans):
    model.add(data, label)

    sys.stdout.write("\rTraining model: {: 6.2f}%".format(count / float(len(poses)-1) * 100))
    sys.stdout.flush()
    count += 1
print("")



######################################
# Use this model for localization now!
######################################

# Extract poses and scans from the data
poses2 = test_data["poses"]
scans2 = test_data["scans"]

"""
# Benchmark classifier time
import time

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




# Initialize filter

mcl = MCL(xlim, ylim, 100, model, [0, 0, 0])
mcl.simulate(logfile, False)
print("Euclidean error sum: " + str(np.sum(mcl.euc_error)))
plt.plot(mcl.euc_error)
plt.ylabel('Error [m]')
plt.show()
