import numpy as np

import polyscope as ps
import polyscope.imgui as psim
import triangle
from alpha_shapes import Alpha_Shaper
from alpha_shapes.boundary import get_boundaries
from scipy.spatial import KDTree

import mouette as M
import sys

ps.init()
ps.set_ground_plane_mode("none")
ps.set_navigation_style("planar")

def circumcenter(a,b,c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a[0] * a[0] + a[1] * a[1]) * (b[1] - c[1]) + (b[0] * b[0] + b[1] * b[1]) * (c[1] - a[1]) + (c[0] * c[0] + c[1] * c[1]) * (a[1] - b[1])) / d
    uy = ((a[0] * a[0] + a[1] * a[1]) * (c[0] - b[0]) + (b[0] * b[0] + b[1] * b[1]) * (a[0] - c[0]) + (c[0] * c[0] + c[1] * c[1]) * (b[0] - a[0])) / d
    return np.array([ux, uy])

try:
    file_path = sys.argv[1]
except:
    print("no file path provided. Loading key.xyz")
    file_path = "../data/2d/key.xyz"

try:
    alpha_max = float(sys.argv[2])
except:
    alpha_max = 2.

data_mouette = M.mesh.load(file_path)
data_mouette = M.transform.fit_into_unit_cube(data_mouette)
points = np.array(data_mouette.vertices)[:,:2] # remove z coordinate (=0)
ps_points = ps.register_point_cloud("points", points, radius=0.005)
AS = Alpha_Shaper(points)
tree = KDTree(points)

delaunay_mouette = None
alpha = 0.01
show_complex = False

delaunay_tris = triangle.delaunay(points)
delaunay_tris = np.asarray(delaunay_tris)
ps.register_surface_mesh("delaunay", points, delaunay_tris, color=[1.,1.,1.], transparency=0.7, edge_width=1.)
delaunay_mouette = M.mesh.from_arrays(points, F=delaunay_tris)
circ_center = delaunay_mouette.faces.create_attribute("circumcenter", float, 2)
circ_radius = delaunay_mouette.faces.create_attribute("circumradius", float)

for i,(A,B,C) in enumerate(delaunay_mouette.faces):
    circ_center[i] = circumcenter(points[A], points[B], points[C])
    circ_radius[i] = M.geometry.distance(circ_center[i][:2], points[A])

def show_alpha_shape():
    global alpha, points, AS
    ps.remove_curve_network("alphashape", error_if_absent=False)
    ps.remove_surface_mesh("alpha_complex", error_if_absent=False)
    
    shape = AS.get_shape(1/alpha)
    pts_alpha = []
    edges_alpha = []
    for boundary in get_boundaries(shape):
        pts_alpha_i = list(boundary.exterior)
        edges_i = M.utils.iterators.cyclic_pairs([len(pts_alpha) + i for i in range(len(pts_alpha_i))])
        pts_alpha += pts_alpha_i
        edges_alpha += edges_i
    ps.register_curve_network("alphashape", np.asarray(pts_alpha), np.asarray(edges_alpha), color=[1,0,0], radius=0.003)

def show_alpha_complex():
    global alpha, points, delaunay_mouette, tree    
    ps.remove_curve_network("alphashape", error_if_absent=False)
    ps.remove_surface_mesh("alpha_complex", error_if_absent=False)
    circ_centers = delaunay_mouette.faces.get_attribute("circumcenter")
    circ_radius = delaunay_mouette.faces.get_attribute("circumradius")

    alpha_complex_faces = []
    for i,(A,B,C) in enumerate(delaunay_mouette.faces):
        if circ_radius[i]<alpha and len(tree.query_ball_point(circ_centers[i][:2], circ_radius[i]))<4:
            alpha_complex_faces.append([A,B,C])
    if len(alpha_complex_faces)>0:
        ps.register_surface_mesh("alpha_complex", points, np.asarray(alpha_complex_faces), color=[1,0,0])

def callback():
    global alpha, show_complex
    psim.PushItemWidth(150)

    # == Show text in the UI
    psim.TextUnformatted("Alpha-shape")        
    # == Set parameters
    changed, show_complex = psim.Checkbox("show complex", show_complex)
    if changed:
        if show_complex:
            show_alpha_complex()
        else:
            show_alpha_shape()

    changed, alpha = psim.SliderFloat("alpha", alpha, v_min=0.01, v_max=alpha_max)
    if changed:
        if show_complex:
            show_alpha_complex()
        else:
            show_alpha_shape()

ps.set_user_callback(callback)
show_alpha_complex()
ps.show()