import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, data, seed=None):
        """
        Initialize the KMeans object.

        Parameters
        ----------
        k : int
            The number of clusters.
        data : ndarray
            A 2D numpy array with data points as rows and features as columns.
        seed : int, optional
            The seed for numpy's random number generator. This parameter is optional, and if not supplied, numpy's random number generator will be used with its current state.

        Attributes
        ----------
        centroids : ndarray
            A 2D numpy array representing the centroids of the clusters. Initialized to zeros.
        labels : ndarray
            A 1D numpy array of the label (cluster) for each data point. Initialized to zeros.
        """
        self.k = k
        self.data = data
        self.seed = seed
        self.centroids = np.zeros((self.k, self.data.shape[1]))
        self.labels = np.zeros(data.shape[0], dtype=np.int32)

    def instantiate_centroids(self):
        """
        Instantiate the centroids at random data points.

        Returns
        -------
        ndarray
            A 2D numpy array representing the centroids of the clusters.
        """
        np.random.seed(self.seed)

        random_indices = np.random.choice(self.data.shape[0], size=self.k)
        for i in range(self.k):
            self.centroids[i, :] = self.data[random_indices[i], :]

        return self.centroids

    @staticmethod
    def calculate_euclidean_distance(point, centroid):
        """
        Calculate the Euclidean distance between a data point and a centroid.

        Parameters
        ----------
        point : ndarray
            A 1D numpy array representing a data point.
        centroid : ndarray
            A 1D numpy array representing a centroid.

        Returns
        -------
        float
            The Euclidean distance between the point and the centroid.
        """
        return np.linalg.norm(point - centroid)

    def update_labels(self):
        """
        Update the label (cluster) for each data point based on the closest centroid.

        Returns
        -------
        int
            The number of data points whose label (cluster) was changed in this step.
        """
        num_changed = 0
        for i in range(self.data.shape[0]):
            data_point = self.data[i, :]
            min_distance = float("inf")
            closest_centroid = -1
            for c in range(self.k):
                centroid = self.centroids[c, :]
                distance = KMeans.calculate_euclidean_distance(data_point, centroid)

                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = c

            if self.labels[i] != closest_centroid:
                self.labels[i] = closest_centroid
                num_changed += 1

        return num_changed

    def update_centroids(self):
        """
        Update the centroids based on the mean of the data points in each cluster.
        """
        for c in range(self.k):
            self.centroids[c] = np.mean(self.data[self.labels == c, :], axis=0)

    def apply_kmeans(self):
        """
        Apply the K-means algorithm until convergence (no changes in labels).

        Returns
        -------
        int
            The number of iterations required for convergence.
        """
        iterations = 0
        self.instantiate_centroids()
        while True:
            iterations += 1
            num_changes = self.update_labels()

            if num_changes == 0:
                break

            self.update_centroids()

        return iterations

    def _print_centroids(self):
        """
        Print the coordinates of the centroids.
        """
        print(f"There are {self.k} centroids:")
        for c in range(self.k):
            print(f"\tCentroid {c + 1} is at {tuple(self.centroids[c])}")

    def plot_data(self):
        """
        Plot the data points, color-coded by cluster, with the centroids marked with red crosses.
        """
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        plt.grid(True)
        plt.show()

    @staticmethod
    def calculate_distortion(data, k_value):
        """
        Calculate the distortion (sum of squared distances from each data point to its centroid) for a range of k values.

        Parameters
        ----------
        data : ndarray
            A 2D numpy array with data points as rows and features as columns.
        k_value : int
            The maximum number of clusters for which to calculate the distortion.

        Returns
        -------
        list
            A list of distortions for k from 1 to k_value.
        """
        distortions = []
        for k in range(1, k_value + 1):
            kmeans = KMeans(k=k, data=data)
            kmeans.instantiate_centroids()
            kmeans.apply_kmeans()
            distortion = 0
            for i in range(data.shape[0]):
                data_point = data[i, :]
                centroid = kmeans.centroids[kmeans.labels[i]]
                distance = kmeans.calculate_euclidean_distance(data_point, centroid)
                distortion += distance ** 2
            distortions.append(distortion)
        return distortions

    @staticmethod
    def plot_distortion(k_value, distortions):
        """
        Plot the distortions for a range of k values.

        Parameters
        ----------
        k_value : int
            The maximum number of clusters for which the distortion was calculated.
        distortions : list
            A list of distortions for k from 1 to k_value.
        """
        k_values = range(1, k_value + 1)
        plt.plot(k_values, distortions, color='teal', marker='o', linestyle='-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.show()
