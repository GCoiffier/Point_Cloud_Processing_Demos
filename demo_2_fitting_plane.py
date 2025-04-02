import numpy as np
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim

import mouette as M
import sys

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

point_id = 10
n_neighbors = 30
plane_size = 0.1
plane_at_center = False
invert_normals = False

tree = KDTree(points) 

def show_normals():
    global point_id, n_neighbors, invert_normals, plane_size, plane_at_center
    ps.remove_curve_network("knn", error_if_absent=False)
    ps.remove_point_cloud("knn_pts", error_if_absent=False)
    ps.remove_curve_network("normals", error_if_absent=False)
    ps.remove_surface_mesh("plane", error_if_absent=False)
    ps.remove_point_cloud("pt", error_if_absent=False)

    _,KNN = tree.query(points[point_id], n_neighbors+1)   
    pts_i = points[KNN,:]
    edges = [(0,i) for i in range(1,n_neighbors+1)]
    ps.register_curve_network("knn", np.array(pts_i), np.array(edges), radius=0.001, color=[0,1,0])
    ps.register_point_cloud("knn_pts", np.array(pts_i), radius=0.001, color=[28,99,227])

    center = np.mean(pts_i, axis=0)
    mat = pts_i-center
    svd = np.linalg.svd(mat.T, full_matrices=False)
    X,Y,normal = plane_size*svd[0][:,0], plane_size*svd[0][:,1], svd[0][:,2]
    if invert_normals: normal *= -1

    pt_ref = center if plane_at_center else points[point_id,:]
    ps_pt = ps.register_point_cloud("pt", pt_ref.reshape((1,3)), radius=0.003, color=[1,1,0])
    ps_pt.add_vector_quantity("normals", normal.reshape((1,3)), color=[1,0,0], enabled=True)
    quad = M.procedural.quad(pt_ref+X+Y, pt_ref-X+Y, pt_ref+X-Y)
    ps.register_surface_mesh("plane", np.array(quad.vertices), np.array(quad.faces), transparency=0.6, color=[0,0,0])

def callback():
    global point_id, n_neighbors, invert_normals, plane_size, plane_at_center
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Normal computation")
    if(psim.Button("Run")):
        show_normals()

    # == Set parameters
    _, point_id = psim.InputInt("point id", point_id)
    _, n_neighbors = psim.InputInt("number of neighbors", n_neighbors)
    _, plane_size = psim.InputFloat("tangent plane size", plane_size)
    _, plane_at_center = psim.Checkbox("plane at centroid", plane_at_center)
    _, invert_normals = psim.Checkbox("invert", invert_normals)
 

ps.set_user_callback(callback)
ps.show()