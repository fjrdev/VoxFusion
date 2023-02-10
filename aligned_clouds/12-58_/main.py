
#Point Density: measures the number of points in the cloud per unit volume
#Completeness: measures how well the point cloud covers the underlying surface or object
#Uniformity: measures the distribution of points in the cloud, and how evenly they are spaced
#Accuracy: measures how closely the points in the cloud match the true shape or position of the underlying surface or object
#Precision: measures the repeatability of the point cloud measurement process
#Normality: measures how much the normals of the points in the cloud deviate from their expected orientation


import open3d as o3d
import numpy as np
import argparse

def evaluate_point_density(point_cloud):
    #print(len(point_cloud.compute_convex_hull()))
    #return np.shape(point_cloud.points)[0] / point_cloud.compute_convex_hull()[1].get_volume()

def evaluate_completeness(point_cloud, ground_truth):
    difference = point_cloud.compute_point_cloud_distance(ground_truth)
    return np.mean(difference.flatten())

def evaluate_accuracy(point_cloud, ground_truth):
    difference = point_cloud.compute_point_cloud_distance(ground_truth)
    return np.max(difference.flatten())

def evaluate_normality(point_cloud):
    covariance = point_cloud.get_covariance_matrix()
    eigen_values, _ = np.linalg.eig(covariance)
    return np.sum(eigen_values) / np.trace(covariance)

def evaluate_uniformity(point_cloud):
    mean = np.mean(point_cloud.points, axis=0)
    variance = np.var(point_cloud.points, axis=0)
    return np.mean(variance)

def evaluate_precision(point_cloud, ground_truth):
    difference = point_cloud.compute_point_cloud_distance(ground_truth)
    return np.mean(np.min(difference, axis=1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View two point clouds")
    parser.add_argument("--map", type=str,
                        help="Path to mapping-derived point cloud, e.g. /Users/julietteburkhardt/Documents/WiSe 22:23/MedInf Visualization/example/Bearded guy.ply")

    parser.add_argument("--ct", type=str,
                        help="Path to mapping-derived point cloud, e.g. /Users/julietteburkhardt/Documents/WiSe 22:23/MedInf Visualization/example/Axle shaft.ply")

    args = parser.parse_args()

    """
    Load Map
    """
    map = o3d.io.read_point_cloud(args.map)

    """
    Load CT
    """
    ct = o3d.io.read_point_cloud(args.ct)

    """
    Evaluate Point Density
    """
    map_density = evaluate_point_density(map)
    #evaluate_point_density(map)
    #ct_density = evaluate_point_density(ct)
    print("Point Density (Map):", map_density)
    #print("Point Density (CT):", ct_density)

    """
    Evaluate Completeness
    """
    #completeness = evaluate_completeness(map, ct)
    #print("Completeness:", completeness)

    """
    Evaluate Accuracy
    """
    #accuracy = evaluate_accuracy(map, ct)
    #print("Accuracy:", accuracy)

    """
    Evaluate Normality
    """
    #map_normality = evaluate_normality(map)
    #ct_normality = evaluate_normality(ct)
    #print("Normality (Map):", map_normality)
    #print("Normality (CT):", ct_normality)

    """
    Evaluate Uniformity
    """
    #map_uniformity = evaluate_uniformity(map)
    #ct_uniformity = evaluate_uniformity(ct)
    #print("Uniformity (Map):", map_uniformity)
    #print("Uniformity (CT):", ct_uniformity)

    """
    Evaluate Precision
    """
    #precision = evaluate_precision(map, ct)
    #print("Precision:", map_uniformity)


