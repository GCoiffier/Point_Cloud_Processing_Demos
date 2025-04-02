import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import triangle
import mouette as M
import sys

ps.init()
ps.set_ground_plane_mode("none")
ps.set_navigation_style("planar")
ps.set_automatically_compute_scene_extents(False)

try:
    file_path = sys.argv[1]
except:
    print("no file path provided. Loading key.xyz")
    file_path = "../data/2d/key.xyz"

data_mouette = M.mesh.load(file_path)
data_mouette = M.transform.fit_into_unit_cube(data_mouette)
points = np.array(data_mouette.vertices)[:,:2] # remove z coordinate (=0)
ps_points = ps.register_point_cloud("points", points, radius=0.005)

def show_crust():
    global length, delaunay_mouette
    ps.remove_curve_network("reconstruct", error_if_absent=False)
    Npts = points.shape[0]
    
    # bb = M.geometry.AABB.of_points(points, 0.5)

    ### Compute Voronoi Diagram
    vor_points, edges, ray_origin, ray_direction = triangle.voronoi(points)
    pl_voronoi = M.mesh.from_arrays(vor_points, E=np.array(edges))
    for iray, orig in enumerate(ray_origin):
        dir = ray_direction[iray]
        final = vor_points[orig] + 50.*dir
        pl_voronoi.edges.append((orig,len(pl_voronoi.vertices)))
        pl_voronoi.vertices.append(np.pad(final,(0,1)))
    
    ps.register_curve_network("voronoi_edges", np.asarray(pl_voronoi.vertices), np.array(pl_voronoi.edges), radius=0.001, enabled=False)
    ps.register_point_cloud("voronoi_points", np.asarray(pl_voronoi.vertices), color=[0,0,0], radius=0.005, enabled=False)
    
    all_points = np.concatenate((points,vor_points))
    tris = triangle.delaunay(all_points)
    ps.register_surface_mesh("delaunay", all_points, np.asarray(tris), enabled=False, color=[1.,1.,1.], edge_width=1)
    edges = []
    seen = set()
    for A,B,C in tris:
        for u,v in [sorted((A,B)), sorted((B,C)), sorted((C,A))]:
            # take a convention that an edge (u,v) is always ordered as u < v
            if (u,v) in seen: continue
            if u<Npts and v<Npts:
                # both points were original points -> edge is added
                edges.append((u,v))
                seen.add((u,v)) # prevent the edge (v,u) from also being added
    ps.register_curve_network("reconstruct", points, np.asarray(edges), radius=0.002, color=[1,0,0], enabled=False)

def callback():
    global length
    psim.PushItemWidth(150)

    # == Show text in the UI
    psim.TextUnformatted("CRUST")    
    if(psim.Button("Run")):
        show_crust()

# ps.set_user_callback(callback)
show_crust()
ps.set_bounding_box([-1,-1,-1], [1,1,1])
ps.show()