import numpy as np
from scipy.spatial import KDTree, Delaunay

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
points = np.array(data_mouette.vertices)
ps_points = ps.register_point_cloud("points", points, radius=0.002)

point_id = 0
n_neighbors = 10
full_knn = False
tree = KDTree(points) 

def show_knn():
    global point_id, n_neighbors, full_knn
    ps.remove_curve_network("knn", error_if_absent=False)
    ps.remove_point_cloud("pt", error_if_absent=False)

    if full_knn:
        _,KNN = tree.query(points, n_neighbors+1)
        edges = []
        for i in range(points.shape[0]):
            for j in range(n_neighbors):
                edges.append((i,KNN[i,j+1]))
        ps.register_curve_network("knn", points, np.array(edges), radius=0.001, color=[1,0,0])

    else:
        _,KNN = tree.query(points[point_id,:], n_neighbors+1)
        
        pts_nn = [points[point_id]] + [points[i] for i in KNN[1:]]
        edges = [(0,i) for i in range(1,n_neighbors+1)]
        ps.register_curve_network("knn", np.array(pts_nn), np.array(edges), radius=0.001, color=[1,0,0])
        ps.register_point_cloud("pt", points[point_id:point_id+1, :], radius=0.003, color=[1,1,0])


def callback():
    global point_id, n_neighbors, full_knn
    psim.PushItemWidth(150)

    # == Show text in the UI
    psim.TextUnformatted("K-nn visualization")
    if(psim.Button("Run")):
        show_knn()
        
    # == Set parameters
    _, point_id = psim.InputInt("point id", point_id)
    _, n_neighbors = psim.InputInt("number of neighbors", n_neighbors)
    _, full_knn = psim.Checkbox("all points", full_knn) 

ps.set_user_callback(callback)
ps.show()