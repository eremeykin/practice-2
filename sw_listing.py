from clustering.agglomerative.ik_means.ik_means import IKMeans
from clustering.agglomerative.a_ward_pb import AWardPB
import numpy as np

class AvgSilhouetteWidthCriterion:
    @staticmethod
    def distance(point1, point2):
        # squared euclidean distance
        return np.sum((point1 - point2) ** 2)

    def _a(self, point_index_i, cluster, cluster_structure):
        # extract data from ClusterStructure object
        data = cluster_structure.data
        dist_list = list()
        # iterate over all points in given cluster
        for point_index_j in cluster.points_indices:
            point_i = data[point_index_i]
            point_j = data[point_index_j]
            # calculate distance between each point and given point
            dist = self.distance(point_i, point_j)
            # append the list of distances
            dist_list.append(dist)
        return np.average(dist_list)  # return average of distances

    def _b(self, point_index_i, cluster, cluster_structure):
        # extract data from ClusterStructure object
        data = cluster_structure.data
        avg_list = list()
        # iterate over all clusters
        for curr_cluster in cluster_structure.clusters:
            dist_list = list()
            if cluster != curr_cluster:  # for all other clusters
                for point_index_j in curr_cluster.points_indices:
                    point_i = data[point_index_i]
                    point_j = data[point_index_j]
                    # calculate each distance
                    dist = self.distance(point_i, point_j)
                    # append the list of distances
                    dist_list.append(dist)
                # append the average distances list
                avg_list.append(np.average(dist_list))
        return np.min(avg_list)  # return minimum of average

    def __call__(self, cluster_structure):
        sw_list = list()
        # iterate over all clusters in ClusterStructure
        for cluster in cluster_structure.clusters:
            # iterate over all points in the current cluster
            for point_index in cluster.points_indices:
                a = self._a(point_index, cluster, cluster_structure)
                b = self._b(point_index, cluster, cluster_structure)
                # calculate SW value
                sw = (b - a) / max(b, a)
                sw_list.append(sw)
        return np.average(sw_list)  # return average SW value
