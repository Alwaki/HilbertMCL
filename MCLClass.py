from numpy.lib.function_base import angle
from ParticleClass import *
import util
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, uniform
import numpy as np
import random
import math
import copy

class MCL(object):
    """ Localization class using particle filter """

    def __init__(self, xlim, ylim, nbr_particles, classifier, pose = [], alpha = [0.1, 0.01, 0.1, 0.01]):
        """ Instantiates the class 
        :param xlim: horizontal borders of map [x1, x2]
        :param ylim: vertical borders of map [y1, y2]
        :param nbr_particles: how many particle samples to populate the filter with
        :type nbr_particles: int
        :param classifier: model handle for position classifier
        :param pose: the starting pose of the robot in 2D [x, y, theta]
        :param alpha: array of odometry motion model noise [a1, a2, a3, a4]
        """

        # Filter data containers
        self.pose = pose      
        self.map_limits = [xlim[0], xlim[1], ylim[0], ylim[1]]  
        self.prev_odom = []
        self.particles = []

        # Filter parameters
        self.nbr_particles = nbr_particles
        self.alpha = alpha
        self.ray_resolution = 0.4   # TODO: check these two
        self.sensor_variance = 0.1

        # Classifier model
        self.model = classifier

        # Truncated normal parameters
        self.tn_std = 0.5
        self.tn_mean = 1
        self.a_clip = 0
        self.b_clip = 1
        self.a, self.b = (self.a_clip - self.tn_mean) / self.tn_std, (self.b_clip - self.tn_mean) / self.tn_std

        # Error statistics collection
        self.euc_error = []
        self.ang_error = []
        self.x_path = []
        self.y_path = []
        self.x_odom = []
        self.y_odom = []
        self.x_truth = []
        self.y_truth = []
        
        # Initialize particle set, either by given pose or global distribution
        weight = 1.0 / self.nbr_particles
        if not self.pose:
            for i in range(self.nbr_particles):
                x = random.uniform(self.map_limits[0], self.map_limits[1])
                y = random.uniform(self.map_limits[2], self.map_limits[3])
                theta = random.uniform(-math.pi, math.pi)
                p = Particle([x,y,theta],weight)
                self.particles.append(p)
        else:
            for i in range(self.nbr_particles):
                p = Particle(self.pose,weight)
                self.particles.append(p)

    def simulate(self, logfile, plot_flag = False):
        """ Runs the localization filter on a given dataset

       :param logfile: CARMEN style file with measurements
       :param plot_flag: determines if simulation is to be visualized and data collected
       :type plot_flag: bool
       """
        for line in open(logfile):
            if line.startswith("FLASER"):
                arr = line.split()
                count = int(arr[1])
                scans = [float(v) for v in arr[2:2+count]]
                ground_truth = [float(v) for v in arr[-9:-6]]
                ground_truth[2] = util.normalize_angle(ground_truth[2])

                # Use scans for bayesian correction update
                self.propagate(1, scans)

                # Normalize particle weights
                self.normalize_weights()

                # Check ESS, and then resample if particle set degenerate
                if(self.calculate_ESS() < self.nbr_particles/2):
                    self.resample()

                # Use ground truth to generate error
                euclidean_error = math.hypot(ground_truth[0]-self.pose[0],ground_truth[1]-self.pose[1])
                angular_error = abs(ground_truth[2]-self.pose[2])
                self.euc_error.append(euclidean_error)
                self.ang_error.append(angular_error)
                self.x_truth.append(ground_truth[0])
                self.y_truth.append(ground_truth[1])

                # Display error
                print("Euclidean Error: " + str(self.euc_error[-1]))
            
            if line.startswith("ODOM"):
                arr = line.split()
                odom = [float(v) for v in arr[1:4]]
                odom[2] = util.normalize_angle(odom[2])
                self.x_odom.append(odom[0])
                self.y_odom.append(odom[1])
                self.propagate(0,odom)
                particle_weights = [o.weight for o in self.particles]
                max_weight = max(particle_weights)
                max_idx = particle_weights.index(max_weight)
                self.pose = self.particles[max_idx].pose
                self.x_path.append(self.pose[0])
                self.y_path.append(self.pose[1])

            # Visualization
            if(plot_flag):
                x_list = [o.pose[0] for o in self.particles]
                y_list = [o.pose[1] for o in self.particles]
                plt.clf()
                """dx = self.map_limits[1] - self.map_limits[0]
                dy = self.map_limits[3] - self.map_limits[2]
                screen = (self.map_limits[0], self.map_limits[2], dx, dy)
                plt.axes(screen)"""
                plt.scatter(x_list, y_list)
                plt.scatter(self.pose[0],self.pose[1])
                plt.pause(0.05)
    
        if(plot_flag):
            plt.show()
                
    def propagate(self, type, measurement):
        """ Takes a measurement to propagate the filter state
       
        :param type: specifies if the measurement is odom or range scans
        :type type: bool
        :param measurement: measurement data of either odom or range scans
        :type measurement: list 
        """

        # Motion update
        if type == 0:

            
            # Initialize odometry
            if not self.prev_odom:
                self.prev_odom = [measurement[0], measurement[1],measurement[2]]

            # If odometry exists, perform motion update of particles
            else:

                for i in range(self.nbr_particles):
                    [x,y,theta] = self.sample_motion_model_odometry(measurement, self.particles[i].pose)
                    self.particles[i].set_pose([x,y,theta])
                
                self.prev_odom = [measurement[0], measurement[1],measurement[2]]

        # Observation update
        if type == 1:
            
            for i in range(self.nbr_particles):

                # Calculate importance weight
                """
                self.particles[i].weight *= self.point_likelihood_model(self.particles[i].pose,
                                                                       measurement)
                                                                       """
                self.particles[i].weight *= self.beam_likelihood_model(self.particles[i].pose, measurement)
            
                

    def sample_motion_model_odometry(self, new_odom, particle_pose):
        """ Samples a new robot pose according to odometry motion model

        :param new_odom: incoming odometry measurement as array [x, y, theta]
        :type new_odom: list 
        :param particle_pose: the pose of the particle to move, as array [x, y, theta]
        :type particle_pose: list
        :returns: a new pose sample, in [x, y, theta] list format
        """
        d_rot_1     = math.atan2(new_odom[1] - self.prev_odom[1], 
                             new_odom[0] - self.prev_odom[0]) - self.prev_odom[2]

        d_trans     = math.hypot(self.prev_odom[0] - new_odom[0],
                             self.prev_odom[1] - new_odom[1])

        d_rot_2     = new_odom[2] - self.prev_odom[2] - d_rot_1

        d_rot_1_hat = d_rot_1 - random.gauss(0, self.alpha[0] * pow(d_rot_1, 2) + 
                                             self.alpha[1] * pow(d_trans, 2))

        d_trans_hat = d_trans - random.gauss(0, self.alpha[2] * pow(d_trans, 2) + 
                                             self.alpha[3] * pow(d_rot_1, 2) + 
                                             self.alpha[3] * pow(d_rot_2, 2))

        d_rot_2_hat = d_rot_2 - random.gauss(0, self.alpha[0] * pow(d_rot_2, 2) + 
                                             self.alpha[1] * pow(d_trans, 2))

        x     = particle_pose[0] + d_trans_hat * math.cos(particle_pose[2] + d_rot_1_hat)
        y     = particle_pose[1] + d_trans_hat * math.sin(particle_pose[2] + d_rot_1_hat)
        theta = (particle_pose[2] + d_rot_1_hat + d_rot_2_hat)
        theta = util.normalize_angle(theta)
        return [x, y, theta]

    def normalize_weights(self):
        """ Re-scales the particle weights so that the sum equals one """
        weights = [o.weight for o in self.particles]
        denominator = np.sum(weights)

        # Control against division by zero, recover by resetting weight
        if denominator == 0:
            scale_factor = 1.0/self.nbr_particles
            for particle in self.particles:
                particle.weight = scale_factor

        else:
            scale_factor = 1.0/denominator
            for particle in self.particles:
                particle.weight *= scale_factor

    def calculate_ESS(self):
        """ Calculates the effective sample size of particle set 
        
        :returns: effective sample size value (varies from 1.0 to nbr_particles)
        :rtype: float
        """
        weights = [o.weight for o in self.particles]
        ESS = 1.0/np.inner(weights,weights)
        return ESS

    def point_likelihood_model(self, particle_pose, measurement, nbr_scans = 8):
        """ Observation model which classifies the end point of the measurements 
        :param particle_pose: The position and heading of the particle
        :param measurement: List of the range scans
        :returns: Likelihood of measurements
        """
        const_2_pi = 2 * math.pi
        angle_increment = const_2_pi / nbr_scans
        list_increment = int(360 / nbr_scans)
        measurement_headings = np.arange(0, const_2_pi, angle_increment).tolist()
        points = np.empty((0,2))
        for i in range(nbr_scans):
            x = i * list_increment

            # Discard measurements outside of range
            if measurement[x] < 40: #TODO: specify as parameter instead
                heading = (self.pose[2] + measurement_headings[i]) % const_2_pi
                point_x = particle_pose[0] + math.cos(heading) * measurement[x]
                point_y = particle_pose[1] + math.sin(heading) * measurement[x]
                point = np.array([[point_x,point_y]])
                points = np.append(points, point, axis=0)

        p = self.model.classify(points)
        p_list = []
        for i in p[:,0]:
            p_list.append(truncnorm.pdf(i, self.a,self.b, loc=self.tn_mean, scale=self.tn_std))

        q = np.prod(p_list)
        return q
        

    def beam_likelihood_model(self, particle_pose, measurement_distances, nbr_scans = 8):
        """ """
        const_2_pi = 2 * math.pi
        angle_increment = const_2_pi / nbr_scans
        list_increment = int(360 / nbr_scans)
        n = 1.0 / math.sqrt(2 * math.pi * self.sensor_variance)
        expected_distances = []
        p = []

        # Calculate expected distances through raycasting
        for i in range(nbr_scans):
            theta = i * angle_increment
            heading = (self.pose[2] + theta)
            heading = util.normalize_angle(heading)
            expected_distance = self.raycast(particle_pose[0:2], heading, self.ray_resolution)
            expected_distances.append(expected_distance)

        # Calculate likelihoods through beam hit model
        for i in range(nbr_scans):
            x = i * list_increment
            p.append(n * math.exp(- pow(measurement_distances[x] - expected_distances[i], 2) / (2 * n)))

        return math.prod(p)

    def resample(self):
        """ Resamples the particle set according to importance weights """
        r = random.uniform(0.0, 1.0/self.nbr_particles)
        aux_particles = []
        c = self.particles[0].weight
        i = 0
        for m in range(self.nbr_particles):
            U = r + m/self.nbr_particles
            while U > c:
                i += 1
                c += self.particles[i].weight
            particle = copy.copy(self.particles[i])
            aux_particles.append(particle)
        self.particles = aux_particles.copy()



    def raycast(self, starting_position, heading, resolution):
        """ Traces a line in a given direction from a given point, checking occupancy 
        :param starting_position: where the line begins from, in [x, y]
        :param heading: direction of line in radians
        :type heading: float
        :param resolution: interval of checking occupancy along the line
        :type resolution: float
        :returns: point of first detected occupancy, or empty list if no detection
        """
        dx = resolution * math.cos(heading)
        dy = resolution * math.sin(heading)
        position = np.reshape(starting_position, [1, -1])
        count = 0
        while(count < 50):
            position[0][0] += dx
            position[0][1] += dy
            probability = self.model.classify(position)
            count += 1
            if probability[0][0] >= 0.98:
                distance = math.hypot(position[0][0]-starting_position[0], position[0][1] - starting_position[1])
                return distance

        return 40
                


        
