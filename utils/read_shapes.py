import numpy as np
import matplotlib.pyplot as plt

def read_off(file_path):
    with open(file_path, 'r') as file:
        # Read the first line (header)
        header = file.readline().strip()
        lst = ['sofa', 'desk', 'monitor', 'night', 'dresser', 'table', 'bathtub']
        if header != 'OFF' and (file_path.split('/')[-3] in lst):
            num_vertices, num_faces, num_edges = map(int, header[3:].split())

        elif header != 'OFF':
            print("Invalid OFF file")
            return None, None
        else:
            # Read the second line containing numVertices, numFaces, and numEdges
            num_vertices, num_faces, num_edges = map(int, file.readline().split())

        # Read vertices
        vertices = []
        for _ in range(num_vertices):
            vertex = list(map(float, file.readline().split()))
            vertices.append(vertex)

        # Read faces
        faces = []
        for _ in range(num_faces):
            face_data = list(map(int, file.readline().split()[1:]))
            faces.append(face_data)

    return np.array(vertices), np.array(faces)




def voxelization(vertices, faces, grid_size=64):
    # Calculate the bounding box of the mesh
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Determine the voxel size
    voxel_size = np.max(max_coords - min_coords) / grid_size

    # Initialize the voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Iterate over each triangle in the mesh
    for face in faces:
        triangle_vertices = vertices[face]

        # Triangulate the triangle into smaller triangles
        triangles = [triangle_vertices]

        # Iterate over each smaller triangle and fill the corresponding voxels
        for tri in triangles:
            min_tri_coords = np.min(tri, axis=0)
            max_tri_coords = np.max(tri, axis=0)

            min_voxel = ((min_tri_coords - min_coords) / voxel_size).astype(int)
            max_voxel = ((max_tri_coords - min_coords) / voxel_size).astype(int)

            # Fill the voxels within the bounding box of the triangle
            voxel_grid[min_voxel[0]:max_voxel[0]+1, min_voxel[1]:max_voxel[1]+1, min_voxel[2]:max_voxel[2]+1] = 1

    return voxel_grid

def visualize_voxelization(voxel_grid, threshold=0.8):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the coordinates of the non-zero voxels (where the object is present)
    x, y, z = np.where(voxel_grid >= threshold)

    # Plot the non-zero voxels
    ax.scatter(x, y, z, c='r', marker='s', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()