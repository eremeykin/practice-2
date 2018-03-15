class AvgSilhouetteWidthCriterion:
	@staticmethod
	def distance(point1, point2):
		return np.sum((point1 - point2) ** 2)

	def _a(self, point_index_i, cluster, cluster_structure):
		data = cluster_structure.data
		dist_list = list()
		for point_index_j in cluster.points_indices:
			# if point_index_i != point_index_j:
			point_i = data[point_index_i]
			point_j = data[point_index_j]
			dist = self.distance(point_i, point_j)
			dist_list.append(dist)
		return np.average(dist_list)

	def _b(self, point_index_i, cluster, cluster_structure):
		data = cluster_structure.data
		avg_list = []
		for curr_cluster in cluster_structure.clusters:
			dist_list = []
			if cluster != curr_cluster:
				for point_index_j in curr_cluster.points_indices:
					point_i = data[point_index_i]
					point_j = data[point_index_j]
					dist = self.distance(point_i, point_j)
					dist_list.append(dist)
				avg_list.append(np.average(dist_list))
		return np.min(avg_list)

	def __call__(self, cluster_structure):
		data = cluster_structure.data
		sw_list = list()
		for cluster in cluster_structure.clusters:
			for point_index in cluster.points_indices:
				a = self._a(point_index, cluster, cluster_structure)
				b = self._b(point_index, cluster, cluster_structure)
				sw = (b - a) / max(b, a)
				sw_list.append(sw)
		return np.average(sw_list)