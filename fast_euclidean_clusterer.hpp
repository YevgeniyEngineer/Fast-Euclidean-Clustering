#ifndef FAST_EUCLIDEAN_CLUSTERER_HPP
#define FAST_EUCLIDEAN_CLUSTERER_HPP

#include "kdtree.hpp" // neighbour_search::KDTree

#include <array>         // std::array
#include <cmath>         // std::min
#include <cstdint>       // std::int32_t
#include <deque>         // std::deque
#include <iostream>      // std::cout
#include <stdexcept>     // std::runtime_error
#include <type_traits>   // std::enable_if_t
#include <unordered_map> // std::unordered_map
#include <utility>       // std::pair
#include <vector>        // std::vector

#define DEBUG_FAST_EUCLIDEAN_CLUSTERER 0

namespace clustering
{
/// @brief Definition of the point struct
template <typename CoordinateType, std::size_t number_of_dimensions,
          typename = std::enable_if_t<(number_of_dimensions == 2) || (number_of_dimensions == 3)>>
using FEC_Point = std::array<CoordinateType, number_of_dimensions>;

/// @brief Clustering class
template <typename CoordinateType, std::size_t number_of_dimensions> class FastEuclideanClusterer final
{
    using Point = FEC_Point<CoordinateType, number_of_dimensions>;

  public:
    static constexpr bool SORT_NEIGHBORS = true;

    FastEuclideanClusterer(const FastEuclideanClusterer &) = delete;
    FastEuclideanClusterer &operator=(const FastEuclideanClusterer &) = delete;
    FastEuclideanClusterer(FastEuclideanClusterer &&) = delete;
    FastEuclideanClusterer &operator=(FastEuclideanClusterer &&) = delete;
    FastEuclideanClusterer() = delete;

    explicit FastEuclideanClusterer(CoordinateType distance_threshold, std::size_t min_cluster_size,
                                    std::size_t max_cluster_size, const std::vector<Point> &points)
        : distance_threshold_squared_(distance_threshold * distance_threshold), min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size), points_(points), kdtree_(points_, false)
    {
        if (min_cluster_size_ == 0)
        {
            throw std::runtime_error("min cluster size set to 0!");
        }
        if (min_cluster_size_ > max_cluster_size_)
        {
            throw std::runtime_error("min cluster size must be strictly less than or equal to max cluster size!");
        }
        cluster_indices_.reserve(points_.size() / min_cluster_size_);
    }

    ~FastEuclideanClusterer() = default;

    const auto getClusterIndices() const
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        // Must have at least 2 points
        if (points_.size() < 2)
        {
            return;
        }

        // Prepare a container for neighbour indices and distances from the target point
        std::vector<std::pair<std::size_t, CoordinateType>> neighbors;
        neighbors.reserve(10000);

        std::vector<bool> processed(points_.size(), false);
        std::int32_t cluster_no = 0;

        std::vector<std::int32_t> current_cluster_indices;
        current_cluster_indices.reserve(10000);

        std::deque<std::int32_t> neighbor_index_queue;

        for (std::int32_t i = 0; i < points_.size(); ++i)
        {
            if (!processed[i])
            {
                current_cluster_indices.clear();
                neighbor_index_queue.clear();

                neighbor_index_queue.push_back(i);
                processed[i] = true;

                while (!neighbor_index_queue.empty())
                {
                    std::int32_t current_point_index = neighbor_index_queue.front();
                    neighbor_index_queue.pop_front();
                    current_cluster_indices.push_back(current_point_index);

                    neighbors.clear();
                    kdtree_.findAllNearestNeighboursWithinRadiusSquared(
                        points_[current_point_index], distance_threshold_squared_, neighbors, SORT_NEIGHBORS);

                    for (auto neighbor_iterator = neighbors.cbegin() + 1; neighbor_iterator != neighbors.cend();
                         ++neighbor_iterator)
                    {
                        if (!processed[neighbor_iterator->first])
                        {
                            neighbor_index_queue.push_back(neighbor_iterator->first);
                            processed[neighbor_iterator->first] = true;
                        }
                    }
                }

                if ((current_cluster_indices.size() >= min_cluster_size_) &&
                    (current_cluster_indices.size() <= max_cluster_size_))
                {
#if DEBUG_FAST_EUCLIDEAN_CLUSTERER
                    std::cout << "Added new cluster " << cluster_no << std::endl;
#endif
                    cluster_indices_[cluster_no++] = current_cluster_indices;
                }
            }
        }
    }

  private:
    const CoordinateType distance_threshold_squared_;
    const std::size_t min_cluster_size_;
    const std::size_t max_cluster_size_;
    const std::vector<Point> points_;
    neighbour_search::KDTree<CoordinateType, number_of_dimensions> kdtree_;

    std::unordered_map<std::int32_t, std::vector<std::int32_t>> cluster_indices_;
};
} // namespace clustering

#endif // FAST_EUCLIDEAN_CLUSTERER_HPP