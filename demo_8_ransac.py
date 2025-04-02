import numpy as np
import sys
import mouette as M
from dataclasses import dataclass

from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
import polyscope as ps

def estimate_normals_svd(points, KNN):
    n_pts = points.shape[0]
    normals = np.zeros_like(points)
    for i in range(n_pts):
        mat_i = np.array([points[nn,:]-points[i,:] for nn in KNN[i,:]])
        svd = np.linalg.svd(mat_i.T, full_matrices=False)
        normals[i,:] = svd[0][:,-1]
    return normals

def correct_normal_orientation(points, normals, tree, KNN):
    # Consistent normal orientation
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

def linear_regression_3D(pts):
    center = np.mean(pts, axis=0)
    svd = np.linalg.svd((pts-center).T, full_matrices=False)
    normal = svd[0][:,-1]
    return normal/np.linalg.norm(normal), center

class Hyperplane:
    def __init__(self, normal, orig):
        self.normal = normal # normal direction of the plane
        self.orig = orig # a point on the plane
    
    @classmethod
    def from_three_points(cls, p1, p2, p3):
        normal = M.geometry.cross(p2-p1, p3-p1)
        if M.geometry.norm(normal)<1e-8:
            raise ValueError("colinear points")
        normal = M.Vec.normalized(normal)
        if normal.y < 0 : normal *= -1
        return Hyperplane(normal, (p1+p2+p3)/3)

    def distance(self, x):
        return abs(M.geometry.dot(self.normal,x-self.orig))

@dataclass
class RansacParameters:
    N_ITER_MAX : int = 200
    MIN_CLUSTER_SIZE : int = 100
    N_CANDIDATES : int = 100
    INLINER_DIST_THRESHOLD : float = 0.3
    INLINER_NORMAL_THRESHOLD : float = 0.8
    CONNECTED_DISTANCE : float = 0.5


class RANSAC:

    def __init__(self, points, normals):
        self.pts = points
        self.nrmls = normals
        self.parameters : RansacParameters = RansacParameters() 
        
        self.models = []
        self._free_indices = None # indices of points that do not belong to a cluster
        self.labels = np.full(points.shape[0], -1)
        self._i_label = 0

    def sample_candidates(self, k=50):
        npts = len(self._free_indices)
        free_pts = self.pts[self._free_indices, :]
        tree = KDTree(free_pts)
        _, knn = tree.query(free_pts,k+1)
        knn = knn[:,1:] # remove self from closest points
        candidates = []
        for _ in range(self.parameters.N_CANDIDATES):
            ip1 = np.random.randint(0,npts)
            ip2,ip3 = np.random.choice(range(k), size=2, replace=False)
            p1 = free_pts[ip1]
            p2 = free_pts[knn[ip1, ip2]]
            p3 = free_pts[knn[ip1, ip3]]
            try:
                hp = Hyperplane.from_three_points(p1,p2,p3)
                candidates.append(hp)
            except ValueError:
                continue
        return candidates

    def keep_biggest_connected_component(self, inds):
        if len(inds)==0: return inds
        pts = self.pts[inds,:]
        tree = KDTree(pts)
        mat = tree.sparse_distance_matrix(tree, self.parameters.CONNECTED_DISTANCE).tocsr()
        n_components, labels = connected_components(mat, directed=False)
        if n_components == 1:
            return inds
        counts = np.zeros(n_components)
        for l in labels:
            counts[l] +=1
        biggest_component = np.argmax(counts)
        return [inds[i] for i in range(len(inds)) if labels[i]==biggest_component]

    def get_inliers_of_candidate(self, candidate, threshold=None):
        if threshold is None: threshold = self.parameters.INLINER_DIST_THRESHOLD
        inliers = []
        for ind in self._free_indices:
            pt = self.pts[ind, :]
            if candidate.distance(pt) < threshold and \
                abs(np.dot(candidate.normal, self.nrmls[ind])) > self.parameters.INLINER_NORMAL_THRESHOLD:
                inliers.append(ind)
        inliers = self.keep_biggest_connected_component(inliers)
        return inliers

    def improve_candidate(self, model):
        normal,orig = model.normal, model.orig
        n_inliers = None
        for _ in range(10):
            inliers = self.get_inliers_of_candidate(Hyperplane(normal, orig))
            # break if the result gets worse
            if n_inliers is not None and len(inliers)<n_inliers: break
            n_inliers = len(inliers)
            new_n, new_o = linear_regression_3D(self.pts[inliers,:])
            if M.geometry.distance(new_n, normal) > M.geometry.distance(-new_n, normal):
                new_n *= -1
            if M.geometry.distance(new_n, normal)<1e-8: 
                break
            normal, orig = new_n, new_o
        return Hyperplane(normal, orig)

    def update_labels(self, candidate):
        inliers = self.get_inliers_of_candidate(candidate)
        if len(inliers)<self.parameters.MIN_CLUSTER_SIZE: return False
        print("Creating a cluster of", len(inliers), "points")
        for ind in inliers:
            self.labels[ind] = self._i_label
        self._i_label += 1
        return True

    def run(self):
        n_consecutive_fails = 0
        for n_iter in range(self.parameters.N_ITER_MAX):
            if n_consecutive_fails > 5: break
            self._free_indices = [x for x in range(self.pts.shape[0]) if self.labels[x]==-1]
            print("iter:", n_iter, " | Remaining points:", len(self._free_indices))
            if len(self._free_indices) < self.parameters.MIN_CLUSTER_SIZE: break
            
            # sample N candidates models
            candidates = self.sample_candidates()
            evaluations = [len(self.get_inliers_of_candidate(cdt)) for cdt in candidates]
            i_maxi = np.argmax(evaluations)
            if evaluations[i_maxi]>=self.parameters.MIN_CLUSTER_SIZE:
                chosen_candidate = candidates[i_maxi]
                chosen_candidate = self.improve_candidate(chosen_candidate)
                if self.update_labels(chosen_candidate):
                    self.models.append(chosen_candidate)
                    n_consecutive_fails = 0
                else:
                    n_consecutive_fails += 1
            else:
                print("Largest cluster of size", evaluations[i_maxi], "<", self.parameters.MIN_CLUSTER_SIZE)
                n_consecutive_fails += 1
            print()

if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
    except:
        print("no file path provided. Loading tree_2roofs.xyz")
        file_path = "../data/tree_2roofs.xyz"

    data_mouette = M.mesh.load(file_path)
    data_mouette = M.transform.fit_into_unit_cube(data_mouette)
    if "roof" in file_path:
        data_mouette = M.transform.rotate(data_mouette, [-np.pi/2,0,0])

    points = np.array(data_mouette.vertices)
    if data_mouette.vertices.has_attribute("normals"):
        normals = data_mouette.vertices.get_attribute("normals").as_array(points.shape[0])
    else:
        tree = KDTree(points)
        _,KNN = tree.query(points, 11)
        KNN = KNN[:,1:]
        normals = estimate_normals_svd(points, KNN)
        normals = correct_normal_orientation(points, normals, tree, KNN)


    ps.init()
    ps.set_ground_plane_mode("none")

    ps_cloud = ps.register_point_cloud("pts", points, enabled=False)
    ps_cloud.add_vector_quantity("normals", normals)

    rs = RANSAC(points,normals)
    rs.parameters.N_CANDIDATES = 50
    rs.parameters.MIN_CLUSTER_SIZE = 200
    rs.parameters.CONNECTED_DISTANCE = 0.02
    rs.parameters.INLINER_DIST_THRESHOLD = 0.01
    rs.parameters.INLINER_NORMAL_THRESHOLD = 0.8
    rs.run()

    ps_cluster_group = ps.create_group("clusters")
    for c_id in range(-1, np.max(rs.labels)+1):
        indices = (rs.labels == c_id)
        pt_clusters = points[indices, :]

        if c_id == -1: # outliers 
            ps_pt_cluster = ps.register_point_cloud("outliers", pt_clusters, radius=0.0005)
            ps_pt_cluster.set_color([0,0,0]) # set color to black for better visualization
        
        else:
            ps_pt_cluster = ps.register_point_cloud(f"cluster_{c_id}", pt_clusters)
            ps_pt_cluster.add_to_group(ps_cluster_group)
    ps.show()
