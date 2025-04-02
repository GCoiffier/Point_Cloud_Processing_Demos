import numpy as np
from scipy.spatial import KDTree

import polyscope as ps
import polyscope.imgui as psim

import mouette as M
import sys

def estimate_normals_svd(points, KNN):
    n_pts = points.shape[0]
    normals = np.zeros_like(points)
    pca = np.zeros((n_pts,3))
    for i in range(n_pts):
        pts_i = points[KNN[i,:],:]
        center = np.mean(pts_i, axis=0)
        mat_i = pts_i-center
        svd = np.linalg.svd(mat_i.T, full_matrices=False)
        normals[i,:] = svd[0][:,-1]
        sg = svd[1]
        pca[i] = sg
    return normals, pca

def correct_normal_orientation_mst(points, normals, tree, KNN, show_tree=False):
    n_pts = normals.shape[0]
    ### Build Minimal Spanning tree
    edges = set()
    for i in range(n_pts):
        for j in KNN[i,:]:
            if j==i: continue
            edges.add(M.utils.keyify(i,j))
    pr_queue = M.utils.PriorityQueue()
    for (i,j) in edges:
        pr_queue.push((i,j), 1 - abs(np.dot(normals[i], normals[j])))

    UF = M.utils.UnionFind(range(n_pts))
    adjacency = [[] for _ in range(n_pts)]
    edges_mst = []
    while not pr_queue.empty():
        elem = pr_queue.pop()
        i,j = elem.x
        if UF.find(i) != UF.find(j):
            UF.union(i,j)
            adjacency[i].append(j)
            adjacency[j].append(i)
            edges_mst.append([i,j])

    if show_tree:
        ps.remove_curve_network("tree", error_if_absent=False)
        ps.register_curve_network("tree", points, np.array(edges_mst), enabled=False, radius=0.001)

    ### Set starting orientation
    exterior_pt = M.Vec(0,10,0)
    _,top_pt = tree.query(exterior_pt)
    if np.dot(normals[top_pt,:], exterior_pt - points[top_pt,:])<0.:
        normals[top_pt,:] *= -1 # if this normal is not outward, we are doomed
    
    ### Traverse spanning tree
    to_visit = []
    visited = np.zeros(n_pts, dtype=bool)
    for adj_pt in adjacency[top_pt]:
        to_visit.append((top_pt, adj_pt))
    visited[top_pt] = True
    
    while len(to_visit)>0:
        parent, node = to_visit.pop()
        if visited[node] : continue
        visited[node] = True
        Nv = normals[node, :]
        Np = normals[parent,:] # suppose that this one is correctly oriented
        if np.dot(-Nv,Np)>np.dot(Nv,Np):
            normals[node,:] *= -1
        for child in adjacency[node]:
            if not visited[child]:
                to_visit.append((node,child))
    return normals

def correct_normal_orientation_bfs(points, normals, KNN):
    # another heuristic to compute a consistent normal orientation
    n_pts = normals.shape[0]
    visited = np.zeros(n_pts,dtype=bool)
    to_visit = M.utils.PriorityQueue()

    exterior_pt = M.Vec(10,0,0)
    _,top_pt = tree.query(exterior_pt)
    if np.dot(normals[top_pt,:], exterior_pt - points[top_pt,:])<0.:
        normals[top_pt,:] *= -1 # if this normal is not outward, we are doomed
    to_visit.push(top_pt,0.)
    while not to_visit.empty():
        elem = to_visit.pop()
        iv,wv = elem.x, elem.priority 
        if visited[iv] : continue
        visited[iv] = True
        Nv = normals[iv, :]
        for nn in KNN[iv,:]:
            if np.dot(Nv, normals[nn,:])<0.:
                normals[nn,:] *= -1
            to_visit.push(nn, wv+M.geometry.distance(points[iv], points[nn]))
    return normals


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
    ps_points = ps.register_point_cloud("points", points, radius=0.002)

    def show_normals():
        global n_neighbors, correct_orient, invert_normals
        n_pts = points.shape[0]
        ps.remove_curve_network("knn", error_if_absent=False)

        _,KNN = tree.query(points, n_neighbors+1)
        edges = []
        for i in range(n_pts):
            for j in range(n_neighbors):
                edges.append((i,KNN[i,j+1]))
        ps.register_curve_network("knn", points, np.array(edges), radius=0.001, color=[0,1,0], enabled=False)
        normals, pca = estimate_normals_svd(points, KNN)
        if data_mouette.vertices.has_attribute("normals"):
            normals = data_mouette.vertices.get_attribute("normals").as_array(n_pts)
        elif correct_orient:
            normals = correct_normal_orientation_mst(points, normals, tree, KNN, show_tree=True)
        if invert_normals:
            normals *= -1
        ps_points.add_vector_quantity("normals", normals, color=[1,0,0], enabled=True)
        
        curv = np.zeros(n_pts)
        crease = np.zeros(n_pts)
        bnd = np.zeros(n_pts)
        corner = np.zeros(n_pts)
        for i in range(n_pts):
            l0,l1,l2 = pca[i,2], pca[i,1], pca[i,0]
            assert l0 <= l1 <= l2
            curv[i] = l0/(l0+l1+l2)
            crease[i] = max(l1-l0, abs(l2-l1-l0))/l2
            bnd[i] = abs(l2-2*l1)/l2
            corner[i] = (l2-l0)/l2

        ps_points.add_scalar_quantity("curvature", curv, enabled=False, cmap="reds")
        ps_points.add_scalar_quantity("crease", crease, enabled=False, cmap="reds")
        ps_points.add_scalar_quantity("edge", bnd, enabled=False, cmap="reds")
        ps_points.add_scalar_quantity("corner", corner, enabled=False, cmap="reds")

    def callback():
        global n_neighbors, correct_orient, invert_normals
        psim.PushItemWidth(150)

        # == Show text in the UI
        psim.TextUnformatted("Normal computation")
        if(psim.Button("Run")):
            show_normals()

        # == Set parameters
        _, n_neighbors = psim.InputInt("number of neighbors", n_neighbors)
        _, correct_orient = psim.Checkbox("orient", correct_orient)
        psim.SameLine() 
        _, invert_normals = psim.Checkbox("invert", invert_normals)

    n_neighbors = 30
    correct_orient = False
    invert_normals = False

    tree = KDTree(points) 

    ps.set_user_callback(callback)
    ps.show()