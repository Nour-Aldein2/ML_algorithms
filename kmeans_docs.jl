#=
KMeans_docs:
- Julia version: 
- Author: nour
- Date: 2023-06-05
=#
"""
KMeans: A structure to represent the KMeans object

Fields:
k - number of clusters
data - 2D array with data points as rows and features as columns
seed - seed for Julia's random number generator
centroids - 2D array representing the centroids of the clusters. Initialized to zeros
labels - 1D array of the label (cluster) for each data point. Initialized to zeros
"""
KMeans

"""
instantiate_centroids!(km::KMeans)

Instantiate the centroids at random data points.

km: KMeans object

Returns: 2D array representing the centroids of the clusters
"""
instantiate_centroids!

"""
calculate_euclidean_distance(point::Array{Float64,1}, centroid::Array{Float64,1})

Calculate the Euclidean distance between a data point and a centroid.

point - 1D array representing a data point
centroid - 1D array representing a centroid

Returns: The Euclidean distance between the point and the centroid
"""
calculate_euclidean_distance

"""
update_labels!(km::KMeans)

Update the label (cluster) for each data point based on the closest centroid.

km: KMeans object

Returns: The number of data points whose label (cluster) was changed in this step
"""
update_labels!

"""
update_centroids!(km::KMeans)

Update the centroids based on the mean of the data points in each cluster.

km: KMeans object
"""
update_centroids!

"""
apply_kmeans!(km::KMeans)

Apply the K-means algorithm until convergence (no changes in labels).

km: KMeans object

Returns: The number of iterations required for convergence
"""
apply_kmeans!

"""
print_centroids(km::KMeans)

Print the coordinates of the centroids.

km: KMeans object
"""
print_centroids

"""
plot_data(km::KMeans)

Plot the data points, color-coded by cluster, with the centroids marked with red crosses.

km: KMeans object
"""
plot_data

"""
calculate_distortion(data::Array{Float64,2}, k_value::Int)

Calculate the distortion (sum of squared distances from each data point to its centroid) for a range of k values.

data - 2D array with data points as rows and features as columns
k_value - The maximum number of clusters for which to calculate the distortion

Returns: A list of distortions for k from 1 to k_value
"""
calculate_distortion

"""
plot_distortion(k_value::Int, distortions::Vector{Float64})

Plot the distortions for a range of k values.

k_value - The maximum number of clusters for which the distortion was calculated
distortions - A list of distortions for k from 1 to k_value
"""
plot_distortion
