
class Particle():
    """ Data storage class for particles """

    def __init__(self, x, w):
        """ Instantiates the class 
        :param pose: list of particleposition and heading, given as [x, y, theta]
        :param weight: importance weight of particle, determined by measurement likelihood
        :type weight: float
        """
        self.pose = x
        self.weight = w

    def set_pose(self, x):
        self.pose = x

    def set_weight(self, w):
        self.weight = w
