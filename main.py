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
from MCLClass import *

#########################################################
#
#                   Model/data Setup
#
#########################################################


# Variables
map_name = "orebro"
logfile = "datasets/" + map_name + "_corrected.log"
classifier_name = "classifiers/" + map_name + ".pkl"
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

# Limits in metric space based on poses with a 2m buffer zone
xlim, ylim = util.bounding_box(poses, 2.0)

# Attempt to load classifier, otherwise perform training
try:
    with open(classifier_name, 'rb') as inp:
        model = pickle.load(inp)
except:
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

    with open('classifiers/orebro.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


#########################################################
#
#                   Initialize filter
#
#########################################################

mcl = MCL(xlim, ylim, 50, model, [0,0,0])
mcl.simulate(logfile, True)

#########################################################
#
#                   Data Plotting
#
#########################################################

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


