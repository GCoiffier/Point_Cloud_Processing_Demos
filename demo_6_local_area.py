import numpy as np
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim

import mouette as M
import sys
import triangle

from demo_4_normals import estimate_normals_svd

def show_plane():
    global point_id, n_neighbors
    _,KNN = tree.query(points, n_neighbors+1)
    
    ps.remove_surface_mesh("local_surf", error_if_absent=False)
    ps.remove_point_cloud("local_pts", error_if_absent=False)
    X, Y, Z = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
    ni = normals[point_id] / np.linalg.norm(normals[point_id])
    Xi,Yi,Zi = (np.cross(basis, ni) for basis in (X,Y,Z))
    Xi = [_v for _v in (Xi,Yi,Zi) if np.linalg.norm(_v)>1e-8][0]
    Xi /= np.linalg.norm(Xi)
    Yi = np.cross(ni, Xi)

    neighbors = [points[j] for j in KNN[point_id,:]] # coordinates of k neighbors
    neighbors = [M.geometry.project_to_plane(_X,normals[point_id],points[point_id]) for _X in neighbors] # Project onto normal plane
    neighbors_plane = [np.array((Xi.dot(_X), Yi.dot(_X))) for _X in neighbors]
    tris = triangle.delaunay(neighbors_plane)
    ps.register_surface_mesh("local_surf", np.array(neighbors), np.array(tris), edge_width=1., color=[0.5,0.5,0.5])
    ps.register_point_cloud("local_pts", points[point_id].reshape((1,3)), radius=0.003, color=[1,0,0])

 
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
ps_points = ps.register_point_cloud("points", points, radius=0.002)

point_id = 0
n_neighbors = 10

tree = KDTree(points)
_,KNN = tree.query(points, n_neighbors+1)
KNN = KNN[:,1:]
normals,_ = estimate_normals_svd(points, KNN)
ps_points.add_vector_quantity("normals", normals)

def callback():
    global point_id, n_neighbors, invert_normals, plane_size, plane_at_center
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Normal computation")
    if(psim.Button("Run")):
        show_plane()

    # == Set parameters
    _, point_id = psim.InputInt("point id", point_id)
    _, n_neighbors = psim.InputInt("number of neighbors", n_neighbors)

ps.set_user_callback(callback)
ps.show()