using Plots
using Statistics
using Random
using DelimitedFiles


include("kmeans_docs.jl")

mutable struct KMeans
    k::Int
    data::Array{Float64,2}
    seed::Int
    centroids::Array{Float64,2}
    labels::Array{Int,1}

    KMeans(k::Int, data::Array{Float64,2}, seed::Int=rand(1:typemax(Int))) = new(k, data, seed, zeros(k, size(data,2)), zeros(Int, size(data,1)))

end

function instantiate_centroids!(km::KMeans)
    Random.seed!(km.seed)
    random_indices = rand(1:size(km.data, 1), km.k)
    for i in 1:km.k
        km.centroids[i, :] = km.data[random_indices[i], :]
    end
    return km.centroids
end

function calculate_euclidean_distance(point::Array{Float64,1}, centroid::Array{Float64,1})
    return sqrt(sum((point .- centroid).^2))
end

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

function update_centroids!(km::KMeans)
    for c in 1:km.k
        km.centroids[c, :] = mean(km.data[km.labels .== c, :], dims=1)
    end
end

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

function print_centroids(km::KMeans)
    println("There are $(km.k) centroids:")
    for c in 1:km.k
        println("\tCentroid $c is at $(Tuple(km.centroids[c, :]))")
    end
end

function plot_data(km::KMeans)
    plt = scatter(km.data[:, 1], km.data[:, 2], color=km.labels)
    scatter!(plt, km.centroids[:, 1], km.centroids[:, 2], color=:red, marker=:x)
    xlabel!(plt, "Feature 1")
    ylabel!(plt, "Feature 2")
    title!(plt, "K-Means Clustering")
    display(plt)
end

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


function plot_distortion(k_value::Int, distortions::Vector{Float64})
    plt = plot(1:k_value, distortions, color=:teal, marker=:circle, linestyle=:solid)
    xlabel!(plt, "Number of Clusters (K)")
    ylabel!(plt, "Distortion")
    title!(plt, "Elbow Method")
    display(plt)
end



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
