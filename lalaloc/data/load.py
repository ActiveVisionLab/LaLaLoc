"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""
import json
import math
import os

import cv2
import numpy as np
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene

from ..utils.polygons import clip_polygon, convert_lines_to_vertices


def round_up_to_multiple(f, factor=2):
    return math.ceil(f / float(factor)) * factor


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


def create_floorplan_from_annos(annos, scene_id, pix_per_mm=0.025, min_factor=32):
    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos["semantics"]:
        for planeID in semantic["planeID"]:
            if annos["planes"][planeID]["type"] == "floor":
                planes.append({"planeID": planeID, "type": semantic["type"]})

        if semantic["type"] == "outwall":
            outerwall_planes = semantic["planeID"]

    # extract hole vertices
    lines_holes = []
    for semantic in annos["semantics"]:
        if semantic["type"] in ["window", "door"]:
            for planeID in semantic["planeID"]:
                lines_holes.extend(
                    np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc["coordinate"] for junc in annos["junctions"]])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][plane["planeID"]]))[
            0
        ].tolist()
        junction_pairs = [
            np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane["type"]])

    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos["planeLineMatrix"][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [
            np.where(np.array(annos["lineJunctionMatrix"][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.insert(0, [outerwall_polygon[0], "outwall"])

    floorplan, affine_params = plot_floorplan(
        polygons, junctions, scene_id, pix_per_mm=pix_per_mm, round_multiple=min_factor
    )
    return floorplan, affine_params


def plot_floorplan(
    polygons, junctions, scene_id, size=512, pix_per_mm=0.025, round_multiple=32
):

    junctions = junctions[:, :2]

    used_junctions = []
    for polygon, _ in polygons:
        used_junctions.append(junctions[np.array(polygon)])
    used_junctions = np.concatenate(used_junctions)
    # shift so floorplan fits in unit square 0 and 1
    min_x = np.min(used_junctions[:, 0])
    max_x = np.max(used_junctions[:, 0])
    min_y = np.min(used_junctions[:, 1])
    max_y = np.max(used_junctions[:, 1])
    shift = np.array((min_x, min_y))

    if pix_per_mm < 0:
        range = max(max_x - min_x, max_y - min_y)
        scale = size / range
        floorplan_shape = (size, size, 3)
    else:
        scale = pix_per_mm
        range_x = max_x - min_x
        range_y = max_y - min_y
        w_ind = round_up_to_multiple(pix_per_mm * range_x, round_multiple)
        h_ind = round_up_to_multiple(pix_per_mm * range_y, round_multiple)
        w = 1216
        h = 960
        floorplan_shape = (h, w, 3)

    junctions -= shift
    junctions *= scale

    floorplan = np.zeros(floorplan_shape, dtype=np.float32)
    for (polygon, poly_type) in polygons:
        contours = junctions[np.array(polygon)].astype(np.int32)
        if poly_type in ["door", "window", "outwall"]:
            cv2.fillPoly(floorplan, pts=[contours], color=(1.0, 1.0, 1.0))
        else:
            cv2.fillPoly(floorplan, pts=[contours], color=(0.5, 0.5, 0.5))

    return floorplan, {"scale": scale, "shift": shift, "w": w_ind, "h": h_ind}
