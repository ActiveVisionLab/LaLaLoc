"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""
import numpy as np
import pymesh


def project(x, meta):
    """ project 3D to 2D for polygon clipping
    """
    proj_axis = max(range(3), key=lambda i: abs(meta["normal"][i]))

    return tuple(c for i, c in enumerate(x) if i != proj_axis)


def project_inv(x, meta):
    """ recover 3D points from 2D
    """
    # Returns the vector w in the walls' plane such that project(w) equals x.
    proj_axis = max(range(3), key=lambda i: abs(meta["normal"][i]))

    w = list(x)
    w[proj_axis:proj_axis] = [0.0]
    c = -meta["offset"]
    for i in range(3):
        c -= w[i] * meta["normal"][i]
    c /= meta["normal"][proj_axis]
    w[proj_axis] = c
    return tuple(w)


def triangulate(points):
    """ triangulate the plane for operation and visualization
    """

    num_points = len(points)
    indices = np.arange(num_points, dtype=np.int)
    segments = np.vstack((indices, np.roll(indices, -1))).T
    tri = pymesh.triangle()
    tri.points = np.array(points)

    tri.segments = segments
    tri.verbosity = 0
    tri.run()
    return tri.mesh


def clip_polygon(polygons, vertices_hole, junctions, meta, clip_holes=True):
    """ clip polygon the hole
    """
    if len(polygons) == 1:
        junctions = [junctions[vertex] for vertex in polygons[0]]
        mesh_wall = triangulate(junctions)
        vertices = np.array(mesh_wall.vertices)
        faces = np.array(mesh_wall.faces)

        return vertices, faces

    else:
        wall = []
        holes = []
        for polygon in polygons:
            if np.any(np.intersect1d(polygon, vertices_hole)):
                holes.append(polygon)
            else:
                wall.append(polygon)

        # extract junctions on this plane
        indices = []
        junctions_wall = []
        for plane in wall:
            for vertex in plane:
                indices.append(vertex)
                junctions_wall.append(junctions[vertex])
        junctions_wall = [project(x, meta) for x in junctions_wall]
        mesh_wall = triangulate(junctions_wall)

        if clip_holes:
            junctions_holes = []
            for plane in holes:
                junctions_hole = []
                for vertex in plane:
                    indices.append(vertex)
                    junctions_hole.append(junctions[vertex])
                junctions_holes.append(junctions_hole)

            junctions_holes = [
                [project(x, meta) for x in junctions_hole]
                for junctions_hole in junctions_holes
            ]
            
            for hole in junctions_holes:
                mesh_hole = triangulate(hole)
                mesh_wall = pymesh.boolean(mesh_wall, mesh_hole, "difference")

        vertices = [project_inv(vertex, meta) for vertex in mesh_wall.vertices]

        return vertices, np.array(mesh_wall.faces)


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons
