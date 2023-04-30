#ifndef FAST_EUCLIDEAN_CLUSTERING_HPP
#define FAST_EUCLIDEAN_CLUSTERING_HPP

#include "fec_point_cloud.hpp"

#include <nanoflann.hpp>

#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clustering
{
template <typename CoordinateType, std::uint32_t number_of_dimensions> class FECClustering
{
    static_assert(std::is_floating_point<CoordinateType>::value,
                  "FECClustering only works with floating point precision points");

    using PointCloudT = FECPointCloud<CoordinateType, number_of_dimensions>;
    using KdTreeT = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<CoordinateType, PointCloudT>,
                                                        PointCloudT, number_of_dimensions>;
    using IndicesT = std::vector<std::uint32_t>;
    using ClusterT = std::unordered_map<std::int32_t, IndicesT>;

  public:
    static constexpr std::int32_t MAX_LEAF_SIZE = 10;
    static constexpr std::int32_t IGNORE_CHECKS = 32;
    static constexpr float USE_APPROXIMATE_SEARCH = 0.0f;
    static constexpr bool SORT_RESULTS = false;

    FECClustering(const FECClustering &) = delete;
    FECClustering &operator=(const FECClustering &) = delete;
    FECClustering(FECClustering &&) = delete;
    FECClustering &operator=(FECClustering &&) = delete;
    FECClustering() = delete;

    /// @brief Constructor of FECClustering object
    /// @param points Input point cloud compatible with nanoflann::KDTreeSingleIndexAdaptor
    explicit FECClustering(const PointCloudT &points, CoordinateType distance_threshold = 0,
                           std::uint32_t min_cluster_size = 1,
                           std::uint32_t max_cluster_size = std::numeric_limits<std::uint32_t>::max(),
                           CoordinateType quality = 0.5)
        : distance_threshold_(distance_threshold), min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size), quality_(quality), points_(points),
          kdtree_index_(number_of_dimensions, points_, {MAX_LEAF_SIZE}),
          search_parameters_(IGNORE_CHECKS, USE_APPROXIMATE_SEARCH, SORT_RESULTS)
    {
        if (min_cluster_size_ < 1)
        {
            throw std::runtime_error("Minimum cluster size should not be less than 1");
        }
        if (max_cluster_size_ < min_cluster_size_)
        {
            throw std::runtime_error("Maximum cluster size should not be less than minimum cluster size");
        }
        if (quality_ < 0.1)
        {
            throw std::runtime_error("Minimum allowed cluster quality should not be less than 0.1");
        }
        if (quality_ > 1.0)
        {
            throw std::runtime_error("Maximum allowed cluster quality should not be greater than 1.0");
        }
        if (points_.empty())
        {
            return;
        }
    }

    const ClusterT getClusterIndices() const noexcept
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        cluster_indices_.clear();

        if (points_.empty())
        {
            return;
        }
        else if (points_.size() == 1)
        {
            cluster_indices_[0] = {0};
            return;
        }

        const auto number_of_points = static_cast<std::uint32_t>(points_.size());

        const auto distance_threshold_squared = distance_threshold_ * distance_threshold_;

        const auto nn_distance_threshold =
            static_cast<CoordinateType>(std::pow((1.0 - quality_) * distance_threshold_, 2.0));

        std::vector<bool> removed(number_of_points, false);

        std::vector<std::pair<std::uint32_t, CoordinateType>> neighbours;
        neighbours.reserve(number_of_points);

        std::vector<std::uint32_t> indices;
        indices.reserve(number_of_points);

        std::queue<std::uint32_t> queue;

        std::int32_t label = 0;

        for (std::uint32_t index = 0; index < number_of_points; ++index)
        {
            if (removed[index])
            {
                continue;
            }

            queue.push(index);
            indices.clear();

            while (!queue.empty())
            {
                const auto p = queue.front();
                queue.pop();

                if (removed[p])
                {
                    continue;
                }

                neighbours.clear();
                const auto number_of_neighbours = kdtree_index_.radiusSearch(&points_[p][0], distance_threshold_squared,
                                                                             neighbours, search_parameters_);

                if (number_of_neighbours > 0)
                {
                    for (const auto &[neighbour_index, neighbour_distance] : neighbours)
                    {
                        if (removed[neighbour_index])
                        {
                            continue;
                        }
                        if (neighbour_distance <= nn_distance_threshold)
                        {
                            removed[neighbour_index] = true;
                            indices.push_back(neighbour_index);
                        }
                        else
                        {
                            queue.push(neighbour_index);
                        }
                    }
                }
            }

            // Note that neighbours should return current index, hence no need to consider it
            // Add cluster indices to the group of clusters for current label
            const auto cluster_size = indices.size();
            if (cluster_size >= min_cluster_size_ && cluster_size <= max_cluster_size_)
            {
                cluster_indices_[label++] = indices;
            }
        }
    }

  private:
    // Inputs
    CoordinateType distance_threshold_;
    std::uint32_t max_cluster_size_;
    std::uint32_t min_cluster_size_;
    CoordinateType quality_;
    PointCloudT points_;
    KdTreeT kdtree_index_;
    nanoflann::SearchParams search_parameters_;

    // Output [label, indices] map
    ClusterT cluster_indices_;
};
} // namespace clustering

#endif // FAST_EUCLIDEAN_CLUSTERING_HPP