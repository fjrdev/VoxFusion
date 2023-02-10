import open3d as o3d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View two point clouds")
    parser.add_argument("--map", type=str, help="Path to mapping-derived point cloud")
    #parser.add_argument("--ct", type=str, help="Path to ct-derived point cloud")
    args = parser.parse_args()

    """
    Load Map
    """

    map = o3d.io.read_point_cloud(args.map)

    """
    Load CT
    """
    #ct = o3d.io.read_point_cloud(args.ct)

    """
    You can access point data with, e.g.:
        map_points = np.asarray(map.points)
    """

    """
    Visualise
    """
    #o3d.visualization.draw_geometries([ ct , map ])
    
    o3d.visualization.draw_geometries([map])
