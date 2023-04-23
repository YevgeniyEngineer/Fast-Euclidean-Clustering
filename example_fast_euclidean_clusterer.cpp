
#include "fast_euclidean_clusterer.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <random>

int main()
{
    constexpr std::size_t NUMBER_OF_POINTS = 100'000;
    constexpr std::size_t NUMBER_OF_DIMENSIONS = 3;
    constexpr std::size_t NUMBER_OF_ITERATIONS = 100;
    constexpr std::size_t MIN_CLUSTER_SIZE = 30;
    constexpr std::size_t MAX_CLUSTER_SIZE = std::numeric_limits<std::size_t>::max();

    constexpr double NEAREST_NEIGHBOUR_PROXIMITY = 0.2;

    using CoordinateType = double;
    using PointType = clustering::FEC_Point<CoordinateType, NUMBER_OF_DIMENSIONS>;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<CoordinateType> dist(-10.0, 10.0);

    std::vector<PointType> points;

    points.reserve(NUMBER_OF_POINTS);
    for (auto i = 0; i < NUMBER_OF_POINTS; ++i)
    {
        PointType point_cache;
        for (auto j = 0; j < NUMBER_OF_DIMENSIONS; ++j)
        {
            point_cache[j] = dist(gen);
        }
        points.emplace_back(point_cache);
    }

    auto t1 = std::chrono::steady_clock::now();
    {
        clustering::FastEuclideanClusterer<CoordinateType, NUMBER_OF_DIMENSIONS> fast_euclidean_clusterer(
            NEAREST_NEIGHBOUR_PROXIMITY, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE, points);

        fast_euclidean_clusterer.formClusters();

        auto cluster_indices = fast_euclidean_clusterer.getClusterIndices();

        std::cout << "Number of clusters: " << cluster_indices.size() << std::endl;
    }
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Elapsed time (s): " << (t2 - t1).count() / 1e9 << std::endl;

    return EXIT_SUCCESS;
}