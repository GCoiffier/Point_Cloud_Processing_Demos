import numpy as np

import polyscope as ps
import polyscope.imgui as psim
import triangle

import mouette as M
import sys

ps.init()
ps.set_ground_plane_mode("none")
ps.set_navigation_style("planar")

try:
    file_path = sys.argv[1]
except:
    print("no file path provided. Loading key.xyz")
    file_path = "../data/2d/key.xyz"

data_mouette = M.mesh.load(file_path)
data_mouette = M.transform.fit_into_unit_cube(data_mouette)
points = np.array(data_mouette.vertices)[:,:2] # remove z coordinate (=0)
ps_points = ps.register_point_cloud("points", points, radius=0.005)

delaunay_mouette : M.mesh.SurfaceMesh = None
length = 0.

def show_reconstruction():
    global length, delaunay_mouette
    ps.remove_curve_network("reconstruct", error_if_absent=False)
    if delaunay_mouette is None:
        delaunay_tris = triangle.delaunay(points)
        delaunay_tris = np.asarray(delaunay_tris)
        delaunay_mouette = M.mesh.from_arrays(points, F=delaunay_tris)
        M.attributes.edge_length(delaunay_mouette)
        ps.register_surface_mesh("delaunay", points, delaunay_tris, color=[1.,1.,1.], transparency=0.7, edge_width=1.)
    
    edge_lengths = delaunay_mouette.edges.get_attribute("length")
    edges = [(0,0)]
    for e,(a,b) in enumerate(delaunay_mouette.edges):
        if edge_lengths[e]<length:
            edges.append([a,b])
    ps.register_curve_network("reconstruct", np.asarray(points), np.asarray(edges), color=[1,0,0], radius=0.003)

def callback():
    global length
    psim.PushItemWidth(150)

    # == Show text in the UI
    psim.TextUnformatted("Naive Delaunay reconstruction")        
    # == Set parameters
    changed, length = psim.SliderFloat("length threshold", length, v_min=0., v_max=0.3)
    if changed:
        show_reconstruction()

ps.set_user_callback(callback)
show_reconstruction()
ps.show()