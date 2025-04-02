import numpy as np
import polyscope as ps

ps.init()
ps.set_ground_plane_mode("none")

np.random.seed(3)

# generate some random test points 
m = 1000 # number of points

points = np.random.normal(np.zeros(3), [1., 0.3, 0.05], size=(m,3))
ps_points = ps.register_point_cloud("points", points, radius=0.002)

center = np.mean(points, axis=0)
svd = np.linalg.svd((points-center).T, full_matrices=False)
s1,s2,s3 = svd[1]
print(s1,s2,s3)
A,B,normal = s1*svd[0][:,0], s2*svd[0][:,1], s3*svd[0][:,2]


ps_pca = ps.register_point_cloud("pca", np.array([center, center, center, center]))
ps_pca.add_vector_quantity("pca", np.array([np.zeros(3), A, B, normal]), enabled=True, length=0.4, radius=0.005)
ps.show()