import numpy as np

import polyscope as ps
import polyscope.imgui as psim
from igl import fast_winding_number_for_points
import triangle

from scipy.spatial import KDTree
import mouette as M
import sys

def winding_number(O, V, F):
    """
    Compute the sum of solid angles subtended by the faces of a mesh at a set of points. In 2D, this outputs the exact winding number by summing up solid angles; in 3D, this uses the fast winding number approximation by Barrill et al. "Fast Winding Numbers for Soups and Clouds" (SIGGRAPH 2018).

    Parameters
    ----------
    O : (p,dim) numpy double array
        Matrix of query point positions
    V : (v,dim) numpy double array
        Matrix of mesh/polyline/pointcloud coordinates (in 2D, this is a polyline)
    F : (f,s) numpy int array
        Matrix of mesh/polyline/pointcloud indices into V

    Returns
    -------
    W : (p,) numpy double array
        Vector of winding numbers

    See Also
    --------
    signed_distance, squared_distance, fast_winding_number

    Examples
    --------
    ```python
    v,f = gpytoolbox.read_mesh("bunny.obj") # Read a mesh
    v = gpytoolbox.normalize_points(v) # Normalize mesh
    # Generate query points
    P = 2*np.random.rand(num_samples,3)-4
    # Compute winding numbers
    W = gpytoolbox.winding_number(P,v,f)
    ```
    """  
    # Compute solid angles     
    VS = V[F[:, 0], :]
    VD = V[F[:, 1], :]

    # 2D vectors from O to VS and VD
    O2VS = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VS[:, :2], axis=0)
    O2VD = np.expand_dims(O[:, :2], axis=1) - np.expand_dims(VD[:, :2], axis=0)

    S = - np.arctan2(O2VD[:, :, 0] * O2VS[:, :, 1] - O2VD[:, :, 1] * O2VS[:, :, 0], O2VD[:, :, 0] * O2VS[:, :, 0] + O2VD[:, :, 1] * O2VS[:, :, 1])
    W = np.sum(S, axis=1) / (2 * np.pi)
    return W


if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("none")

    try:
        file_path = sys.argv[1]
    except:
        print("no file path provided. Loading botijo.xyz")
        file_path = "../data/botijo.xyz"

    data_mouette = M.mesh.load(file_path)

    data_mouette = M.transform.fit_into_unit_cube(data_mouette)
    points = np.array(data_mouette.vertices)
    n_pts = points.shape[0]
    ps_points = ps.register_point_cloud("points", points, radius=0.003, color=[0.5,0.5,0.5])
    tree = KDTree(points)
    _,KNN = tree.query(points, 11)
    KNN = KNN[:,1:]

    if data_mouette.vertices.has_attribute("normals"):
        normals = data_mouette.vertices.get_attribute("normals").as_array(len(data_mouette.vertices))
    else:
        print("Estimate normals")
        normals,_ = estimate_normals_svd(points, KNN)
        normals = correct_normal_orientation_mst(points, normals, tree, KNN)
        data_mouette.vertices.register_array_as_attribute("normals", normals)

    if data_mouette.vertices.has_attribute("area"):
        A = data_mouette.vertices.get_attribute("area").as_array(len(data_mouette.vertices))
    else:
        print("Estimate vertex local area")
        A = estimate_local_areas(points, normals, KNN)
        data_mouette.vertices.register_array_as_attribute("area", A)

    extension = M.utils.get_extension(file_path)
    new_file_path = file_path.split("."+extension)[0] + "_with_wn_data.geogram_ascii"
    M.mesh.save(data_mouette, new_file_path)

    domain = M.geometry.AABB.of_points(points, 0.2)
    points_visu = M.sampling.sample_AABB(domain, 1_000_000)
    WN = fast_winding_number_for_points(points, normals, A, points_visu)
    WN = np.clip(WN, -1, 1)
    
    ps_points_visu = ps.register_point_cloud("visu", points_visu, radius=0.003)
    ps_points_visu.set_point_render_mode("quad")
    ps_points_visu.add_scalar_quantity("GWN", WN, enabled=True, vminmax=(-1,1), cmap="coolwarm")
    ps_plane = ps.add_scene_slice_plane()
    ps_plane.set_pose(np.mean(points,axis=0), (1., 0., 0.))
    ps.show()