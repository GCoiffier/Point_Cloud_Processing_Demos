import numpy as np
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim

import mouette as M
import sys

ps.init()
ps.set_ground_plane_mode("none")

data_mouette = M.mesh.load("../data/tree_2roofs.xyz")
data_mouette = M.transform.rotate(data_mouette, [-np.pi/2, 0,0])
data_mouette = M.transform.fit_into_unit_cube(data_mouette)
ps.register_point_cloud("2roofs", np.array(data_mouette.vertices))

data_mouette = M.mesh.load("../data/spot.obj")
data_mouette = M.transform.fit_into_unit_cube(data_mouette)
m = ps.register_surface_mesh("spot", np.array(data_mouette.vertices), np.array(data_mouette.faces), enabled=False)
m.set_edge_width(1)
ps.show()