using Plots
using Statistics
using Random
using DelimitedFiles


"""
KMeans(k::Int, data::Array{Float64,2}, seed::Int=rand(1:typemax(Int)))

Construct a KMeans struct with the provided number of clusters, data, and seed. The centroids 
and labels are initialized to empty arrays of appropriate dimensions.

# Arguments
- `k::Int`: The number of clusters.
- `data::Array{Float64,2}`: The data to be clustered.
- `seed::Int`: The seed for random number generation (optional).
"""
mutable struct KMeans
    k::Int
    data::Array{Float64,2}
    seed::Int
    centroids::Array{Float64,2}
    labels::Array{Int,1}

    KMeans(k::Int, data::Array{Float64,2}, seed::Int=rand(1:typemax(Int))) = new(k, data, seed, zeros(k, size(data,2)), zeros(Int, size(data,1)))

end

@doc """
instantiate_centroids!(km::KMeans)

Initialize the centroids of the KMeans object by randomly picking data points.

# Arguments
- `km::KMeans`: A KMeans object.
"""
function instantiate_centroids!(km::KMeans)
    Random.seed!(km.seed)
    random_indices = rand(1:size(km.data, 1), km.k)
    for i in 1:km.k
        km.centroids[i, :] = km.data[random_indices[i], :]
    end
    return km.centroids
end


"""
    calculate_euclidean_distance(point::Array{Float64,1}, centroid::Array{Float64,1})

    Calculates the Euclidean distance between a point and a centroid.

    # Arguments
    - `point::Array{Float64,1}`: A data point.
    - `centroid::Array{Float64,1}`: A centroid.
"""
function calculate_euclidean_distance(point::Array{Float64,1}, centroid::Array{Float64,1})
    return sqrt(sum((point .- centroid).^2))
end


"""
    update_labels!(km::KMeans)

    Assign each data point to the closest centroid and update the labels of the KMeans object.

    # Arguments
    - `km::KMeans`: A KMeans object.
"""
function update_labels!(km::KMeans)
    num_changed = 0
    for i in 1:size(km.data, 1)
        data_point = km.data[i, :]
        min_distance = Inf
        closest_centroid = -1
        for c in 1:km.k
            centroid = km.centroids[c, :]
            distance = calculate_euclidean_distance(data_point, centroid)
            if distance < min_distance
                min_distance = distance
                closest_centroid = c
            end
        end
        if km.labels[i] != closest_centroid
            km.labels[i] = closest_centroid
            num_changed += 1
        end
    end
    return num_changed
end


"""
    update_centroids!(km::KMeans)

    Update the centroids of the KMeans object based on the mean of the data points in each cluster.

    # Arguments
    - `km::KMeans`: A KMeans object.
"""
function update_centroids!(km::KMeans)
    for c in 1:km.k
        km.centroids[c, :] = mean(km.data[km.labels .== c, :], dims=1)
    end
end


"""
    apply_kmeans!(km::KMeans)

    Apply the K-Means algorithm to the KMeans object until convergence.

    # Arguments
    - `km::KMeans`: A KMeans object.
"""
function apply_kmeans!(km::KMeans)
    iterations = 0
    instantiate_centroids!(km)
    while true
        iterations += 1
        num_changes = update_labels!(km)
        if num_changes == 0
            break
        end
        update_centroids!(km)
    end
    return iterations
end


"""
    print_centroids(km::KMeans)

    Print the centroids of the KMeans object.

    # Arguments
    - `km::KMeans`: A KMeans object.
"""
function print_centroids(km::KMeans)
    println("There are $(km.k) centroids:")
    for c in 1:km.k
        println("\tCentroid $c is at $(Tuple(km.centroids[c, :]))")
    end
end


"""
    plot_data(km::KMeans)

    Plot the data points and centroids of the KMeans object.

    # Arguments
    - `km::KMeans`: A KMeans object.
"""
function plot_data(km::KMeans)
    plt = scatter(km.data[:, 1], km.data[:, 2], color=km.labels)
    scatter!(plt, km.centroids[:, 1], km.centroids[:, 2], color=:red, marker=:x)
    xlabel!(plt, "Feature 1")
    ylabel!(plt, "Feature 2")
    title!(plt, "K-Means Clustering")
    display(plt)
end


"""
    calculate_distortion(data::Array{Float64,2}, k_value::Int)

    Calculate the distortion (sum of squared distances from each point to its centroid) 
    for a range of number of clusters.

    # Arguments
    - `data::Array{Float64,2}`: The data to be clustered.
    - `k_value::Int`: The maximum number of clusters to consider.
"""
function calculate_distortion(data::Array{Float64,2}, k_value::Int)
    distortions = Vector{Float64}()  # Changed this line
    for k in 1:k_value
        km = KMeans(k, data)
        instantiate_centroids!(km)
        apply_kmeans!(km)
        distortion = 0
        for i in 1:size(data, 1)
            data_point = data[i, :]
            centroid = km.centroids[km.labels[i], :]
            distance = calculate_euclidean_distance(data_point, centroid)
            distortion += distance ^ 2
        end
        push!(distortions, distortion)
    end
    return distortions
end


"""
    plot_distortion(k_value::Int, distortions::Vector{Float64})

    Plot the distortions for a range of number of clusters.

    # Arguments
    - `k_value::Int`: The maximum number of clusters considered.
    - `distortions::Vector{Float64}`: The calculated distortions for each number of clusters.
"""
function plot_distortion(k_value::Int, distortions::Vector{Float64})
    plt = plot(1:k_value, distortions, color=:teal, marker=:circle, linestyle=:solid)
    xlabel!(plt, "Number of Clusters (K)")
    ylabel!(plt, "Distortion")
    title!(plt, "Elbow Method")
    display(plt)
end


"""
    main()

    The main function that performs K-Means clustering on the provided data, 
    and prints and plots the results.
"""
function main()
    # Read the data
    data = DelimitedFiles.readdlm("data.tsv", '\t')[2:end, :]

    # Convert data to Float64
    data = convert(Array{Float64,2}, data)

    # Set parameters for KMeans
    k = 2
    seed = 42

    # Initialize KMeans object
    km = KMeans(k, data, seed)

    # Apply KMeans algorithm
    iterations = apply_kmeans!(km)

    # Print results
    println("KMeans algorithm converged in $iterations iterations.")
    print_centroids(km)

    # Plot results
    plot_data(km)

    # Calculate distortions for k values from 1 to 10
    distortions = calculate_distortion(data, 10)

    # Plot distortions
    plot_distortion(10, distortions)
end

# Run the main function
main()

