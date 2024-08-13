import warnings
import os
import numpy as np
import open3d as o3d
import pyvista as pv


from scipy.spatial.transform import Rotation 


from superquadric_helpers import load_superquadric_params


def vertices_from_superquadric(shape, scale, euler, translation):
    epsilon1, epsilon2 = shape
    a1, a2, a3 = scale

    # Turn Euler angles into a rotation matrix
    R = Rotation.from_euler('ZYX', euler).as_matrix()

    # Create transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = translation
    
    # Create a meshgrid for u and v
    u = np.linspace(-np.pi/2, np.pi/2, 100)
    v = np.linspace(-np.pi, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Parametric equations for the superquadric
    x = a1 * np.sign(np.cos(u)) * (np.abs(np.cos(u))**epsilon1) * np.sign(np.cos(v)) * (np.abs(np.cos(v))**epsilon2)
    y = a2 * np.sign(np.cos(u)) * (np.abs(np.cos(u))**epsilon1) * np.sign(np.sin(v)) * (np.abs(np.sin(v))**epsilon2)
    z = a3 * np.sign(np.sin(u)) * (np.abs(np.sin(u))**epsilon1)

    # north_pole = np.array([0, 0, a3])
    # south_pole = np.array([0, 0, -a3])

    # Flatten the arrays and stack them to get a list of vertices
    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    # Create the triangles for the mesh
    triangles = []
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            # Reverse the order of the vertices
            triangles.append([i * len(v) + j, (i + 1) * len(v) + j, i * len(v) + (j + 1)])
            triangles.append([(i + 1) * len(v) + j, (i + 1) * len(v) + (j + 1), i * len(v) + (j + 1)])
            
    triangles = np.array(triangles)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.transform(T)
    vertices = np.asarray(pcd.points)

    return vertices, triangles


def plot_quadric_open3d(vertices, triangles, color=[1.0, 0.0, 0.0], frame_size=1.0, show_frames=True, visualize=True):
    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Optionally, compute the vertex normals to visualize the mesh better
    mesh.compute_vertex_normals()

    # apply color to the mesh
    mesh.paint_uniform_color(color)

    geometries = [mesh]

    if show_frames: 
        # Create frame at the origin
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)

        # Create frame at the quadric center
        frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        # frame2.transform(T)

        geometries.append(frame)
        geometries.append(frame2)

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # Visualize the mesh
        # o3d.visualization.draw_geometries([pcd, frame, frame2])
        o3d.visualization.draw_geometries([*geometries, pcd])

    return geometries


def plot_quadric_pyvista(vertices, color=[1.0, 0.0, 0.0], opacity=1.0, visualize=True):
    cloud = pv.PolyData(vertices)
    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()

    if visualize:
        plotter = pv.Plotter()
        plotter.add_mesh(shell, opacity=opacity, color=color)
        plotter.show()
    
    return shell


def superquadric_plotter(shape, scale, euler, translation, plotter="open3d", opacity=1.0):
    vertices, triangles = vertices_from_superquadric(shape, scale, euler, translation)

    if plotter == "open3d":
        warnings.warn("Some of the meshes generated with Open3D have holes and there is a gap in the mesh. " + \
                      "Try pyvista instead.")
        if opacity < 1.0:
            warnings.warn("Current usage of Open3D does not support opacity for meshes. Setting opacity to 1.0")
            opacity = 1.0
        plot_quadric_open3d(vertices, triangles, show_frames=True)
    elif plotter == "pyvista":
        plot_quadric_pyvista(vertices, opacity=opacity)


def add_mesh_from_file(file_path, backend="pyvista", opacity=1.0, geometries=None, plotter=None, color=None):
    shape, scale, euler, translation = load_superquadric_params(file_path)
    vertices, triangles = vertices_from_superquadric(shape, scale, euler, translation)

    if backend == "open3d":
        if geometries is None:
            warnings.warn("No geometries provided. Creating a new list of geometries.")
            geometries = []
        mesh = plot_quadric_open3d(vertices, triangles, show_frames=True, visualize=False)
        geometries.append(mesh)
        return None, geometries
    elif backend == "pyvista":
        if plotter is None:
            warnings.warn("No plotter provided. Creating a new PyVista plotter.")
            plotter = pv.Plotter()
        mesh = plot_quadric_pyvista(vertices, visualize=False)
        if color is not None:
            plotter.add_mesh(mesh, opacity=opacity, color=color)
        else:
            plotter.add_mesh(mesh, opacity=opacity)
        return plotter, None
    else:
        raise ValueError("Invalid backend")


def plot_scene(scene_dir, backend="pyvista", collision_constraints=False, semantic_constraints=False, opacity=1.0,
               collision_constraints_color=[1.0, 0.0, 0.0], semantic_constraints_color=[0.0, 0.0, 1.0]):
    # Create a scene with a collection of point clouds and potentially multiple superquadrics
    if backend == "open3d":
        geometries = []
        plotter = None
    elif backend == "pyvista":
        plotter = pv.Plotter()
        geometries = None
    else:
        raise ValueError("Invalid backend")

    # Get all the point cloud files in the scene directory
    point_cloud_files = [f for f in os.listdir(scene_dir) if f.endswith(".ply") and not "segmented" in f]

    # Load the point clouds
    point_clouds = []
    for file in point_cloud_files:
        if "segmented" in file:
            continue
        if backend == "open3d":
            pcd = o3d.io.read_point_cloud(os.path.join(scene_dir, file))
            geometries.append(pcd)
        elif backend == "pyvista":
            # Having a hard time loading an RGB point cloud with PyVista. 
            # Using Open3D to load the point cloud and then converting it to PyVista
            pcd_tmp = o3d.io.read_point_cloud(os.path.join(scene_dir, file))
            pcd = pv.read(os.path.join(scene_dir, file))
            plotter.add_points(pcd, rgb=pcd_tmp.colors)
        else:
            raise ValueError("Invalid backend")
        point_clouds.append(pcd)

    if collision_constraints:
        # Load the superquadric parameters for collision avoidance
        superquadric_files = [f for f in os.listdir(scene_dir) if f.endswith("_recovered.txt")]       

        for superquadric_file in superquadric_files:
            print(f"Loading superquadric parameters from {superquadric_file}")
            
            file_path = os.path.join(scene_dir, superquadric_file)
            plotter, geometries = add_mesh_from_file(file_path, backend=backend, opacity=opacity, geometries=geometries, plotter=plotter, color=collision_constraints_color)
            
        if semantic_constraints:
            # Load the semantic constraints for the superquadrics
            for key, values in semantic_constraints.items():
                print(f"Semantic constraints for object {key}: {values}")

                # Check if there are files that have been segmented into parts
                segmented_files = [f for f in os.listdir(scene_dir) if "{}_segmented".format(key) in f and f.endswith(".txt")]
                if len(segmented_files) > 0:
                    for segmented_file in segmented_files:
                        for value in values:
                            if value in segmented_file:
                                file_path = os.path.join(scene_dir, segmented_file)
                                plotter, geometries = add_mesh_from_file(file_path, backend=backend, opacity=opacity, geometries=geometries, plotter=plotter, 
                                                                         color=semantic_constraints_color)
                else:
                    print(key)
                    for value in values:
                        file_name = [f for f in os.listdir(scene_dir) if "{}_recovered_{}".format(key, value) in f and f.endswith(".txt")][0]
                        file_path = os.path.join(scene_dir, file_name)
                        print(f"Loading superquadric parameters from {file_name}")
                        plotter, geometries = add_mesh_from_file(file_path, backend=backend, opacity=opacity, geometries=geometries, plotter=plotter, 
                                                                 color=semantic_constraints_color)

    # Plot the entire scene
    if backend == "open3d":
        o3d.visualization.draw_geometries(geometries)
    elif backend == "pyvista":
        plotter.show()


if __name__ == "__main__":
    plotter = "pyvista"
    plotters = ["pyvista", "open3d"]
    opacity = 0.3

    # semantic_const_dict = None
    collision_constraints = True
    collision_constraints_color = [1.0, 0.0, 0.0]
    semantic_constraints = False
    semantic_constraints_color = [0.0, 0.0, 1.0]
    scene = "laptop_books"
    scenes = ["books", "laptop_books", "balloons_paper_towel"]

    if scene not in scenes:
        raise ValueError(f"Invalid scene. Choose from {scenes}")
    elif scene == "books":
        # First scene: Books
        scene_name = "2024-06-06--00-28-50_books"
        semantic_const_dict = {"1": ["above", "around"]
                               }  # {1: ["above", "around"]}
    elif scene == "laptop_books":
        # Second scene: Laptop and books
        scene_name = "2024-06-06--06-23-34_laptop_books"    
        semantic_const_dict = {"1_4_12": ["above", "around"], 
                            "7": ["above", "around"]
                            }  # {1: ["above", "around"], 7: ["above", "around"]}
    elif scene == "balloons_paper_towel":
        # Third scene: Balloons and paper towel
        scene_name = "2024-06-06--17-05-49_balloons_paper_towel"
        semantic_const_dict = {"0": ["above", "around"], 
                               "2": ["around", "under"], 
                               "7_13_15_18": ["around", "under"]
                               }  # {0: ["above", "around"], 2: ["around", "under"], 7: ["around", "under"]}
    
    if not semantic_constraints or not collision_constraints:
        semantic_const_dict = None

    scene_dir = "data/{}".format(scene_name)

    plot_scene(scene_dir, collision_constraints=collision_constraints, semantic_constraints=semantic_const_dict, 
               opacity=opacity, collision_constraints_color=collision_constraints_color, semantic_constraints_color=semantic_constraints_color)
