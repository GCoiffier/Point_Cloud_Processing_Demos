import numpy as np
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim
from tqdm import trange

import mouette as M
import sys
import triangle

from demo_4_normals import estimate_normals_svd

def estimate_local_areas(points: np.ndarray, normals: np.ndarray, KNN):
    n_pts = points.shape[0]
    A = np.zeros(n_pts)
    X, Y, Z = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
    for i in trange(n_pts):
        ni = normals[i] / np.linalg.norm(normals[i])
        Xi,Yi,Zi = (np.cross(basis, ni) for basis in (X,Y,Z))
        Xi = [_v for _v in (Xi,Yi,Zi) if np.linalg.norm(_v)>1e-8][0]
        Xi /= np.linalg.norm(Xi)
        Yi = np.cross(ni, Xi)

        neighbors = [points[j] for j in KNN[i,:]] # coordinates of k neighbors
        neighbors = [M.geometry.project_to_plane(_X,normals[i],points[i]) for _X in neighbors] # Project onto normal plane
        neighbors = [np.array((Xi.dot(_X), Yi.dot(_X))) for _X in neighbors]

        for (p1,p2,p3) in triangle.triangulate({"vertices" : neighbors})["triangles"]:
            A[i] += M.geometry.triangle_area_2D(neighbors[p1], neighbors[p2], neighbors[p3])
    return A

def estimate_local_lengths(points: np.ndarray, tree: KDTree):
    global n_neighbors
    n_pts = points.shape[0]
    A = np.zeros(n_pts)
    dist, KNN = tree.query(points, 11)
    for i in range(n_pts):
        A[i] = np.mean(dist[i,1:])
    return A

if __name__ == "__main__":
        
    ps.init()
    ps.set_ground_plane_mode("none")

    try:
        file_path = sys.argv[1]
    except:
        print("no file path provided. Loading front.xyz")
        file_path = "../data/front.xyz"

    data_mouette = M.mesh.load(file_path)
    if "simple_roof" in file_path:
        data_mouette = M.transform.rotate(data_mouette, [-np.pi/2, 0, 0])
    data_mouette = M.transform.scale(data_mouette, 100.)
    data_mouette = M.transform.fit_into_unit_cube(data_mouette)

    points = np.array(data_mouette.vertices)
    ps_points = ps.register_point_cloud("points", points, radius=0.01)

    n_neighbors = 10

    tree = KDTree(points)
    _,KNN = tree.query(points, n_neighbors+1)
    KNN = KNN[:,1:]

    normals,_ = estimate_normals_svd(points, KNN)
    ps_points.add_vector_quantity("normals", normals)

    areas = estimate_local_areas(points, normals, KNN)
    ps_points.add_scalar_quantity("area", areas, enabled=True)

    lengths = estimate_local_lengths(points, tree)
    ps_points.add_scalar_quantity("lengths", lengths)
    ps.show()