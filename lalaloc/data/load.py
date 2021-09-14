"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""
import os
import json

import numpy as np
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene

from ..utils.polygons import convert_lines_to_vertices, clip_polygon


def load_scene_annos(root, scene_id):
    with open(
        os.path.join(root, f"scene_{scene_id:05d}", "annotation_3d.json")
    ) as file:
        annos = json.load(file)
    return annos


def prepare_geometry_from_annos(annos, for_visualisation=False):
    junctions = [item["coordinate"] for item in annos["junctions"]]

    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(
                    np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)
    _, vertices_holes = np.where(np.array(annos["lineJunctionMatrix"])[lines_holes])
    vertices_holes = np.unique(vertices_holes)

    # load polygons
    rooms = []
    floor_verts = []
    floor_faces = []
    min_x = 1e15
    max_x = -1e15
    min_y = 1e15
    max_y = -1e15
    for semantic in annos["semantics"]:
        if semantic["type"] in ["outwall", "door", "window"]:
            continue
        polygons = []
        for planeID in semantic["planeID"]:
            plane_anno = annos["planes"][planeID]
            lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
            junction_pairs = [
                np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist()
                for lineID in lineIDs
            ]
            polygon = convert_lines_to_vertices(junction_pairs)
            vertices, faces = clip_polygon(
                polygon, vertices_holes, junctions, plane_anno, clip_holes=False
            )
            polygons.append(
                [
                    vertices,
                    faces,
                    planeID,
                    plane_anno["normal"],
                    plane_anno["type"],
                    semantic["type"],
                ]
            )

        room_verts = []
        room_faces = []
        for vertices, faces, planeID, normal, plane_type, semantic_type in polygons:
            vis_verts = np.array(vertices)
            vis_faces = np.array(faces)
            if len(vis_faces) == 0:
                continue

            room_verts.append(torch.Tensor(vertices))
            room_faces.append(torch.Tensor(faces))

            min_x = min(min_x, np.min(vis_verts[:, 0]))
            max_x = max(max_x, np.max(vis_verts[:, 0]))
            min_y = min(min_y, np.min(vis_verts[:, 1]))
            max_y = max(max_y, np.max(vis_verts[:, 1]))

            if plane_type == "floor":
                floor_verts.append(torch.Tensor(vertices))
                floor_faces.append(torch.Tensor(faces))
        if not for_visualisation:
            room = join_meshes_as_scene(Meshes(room_verts, room_faces))
        else:
            room = Meshes(
                room_verts, room_faces
            )  # This provides the correct form for visualisation
        rooms.append(room)
    floors = Meshes(verts=floor_verts, faces=floor_faces)
    limits = (min_x, max_x, min_y, max_y)
    return rooms, floors, limits
