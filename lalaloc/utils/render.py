import numpy as np
import pyredner
import torch
from pyredner import scene

device = torch.device("cpu:0")

pyredner.set_print_timing(False)
pyredner.set_device(device)


def create_camera(pose, img_size):
    # look at x axis equivalent to an offset x'
    look_at = pose.clone() + torch.Tensor([1, 0, 0])
    up = torch.Tensor([0.0, 0.0, 1.0])

    return pyredner.Camera(
        position=pose,
        look_at=look_at,
        up=up,
        camera_type=pyredner.camera_type.panorama,
        resolution=img_size,
    )


def create_objects(mesh):
    material = pyredner.Material()
    objects = []
    for verts, faces in zip(mesh.verts_list(), mesh.faces_list()):
        objects.append(pyredner.Object(verts, faces.int(), material))
    return objects


def render_scene(config, mesh, pose):
    img_size = config.RENDER.IMG_SIZE
    pose = torch.Tensor(pose)

    camera = create_camera(pose, img_size)
    objects = create_objects(mesh)
    scene = pyredner.Scene(camera=camera, objects=objects)

    img = pyredner.render_g_buffer(scene, [pyredner.channels.depth], device=device)
    # flip x axis for equivalent to a flipped x'
    img = img.flip(dims=[1])
    img = img.cpu().squeeze(2).numpy()
    img = np.ascontiguousarray(img)
    return img


def render_semantics(config, mesh, pose):
    img_size = config.RENDER.IMG_SIZE
    pose = torch.Tensor(pose)

    camera = create_camera(pose, img_size)
    objects = create_objects(mesh)
    scene = pyredner.Scene(camera=camera, objects=objects)

    img = pyredner.render_g_buffer(scene, [pyredner.channels.shape_id], num_samples=1)
    # flip x axis for equivalent to a flipped x'
    img = img.flip(dims=[1])
    img = img.cpu().squeeze(2).numpy()
    img = np.ascontiguousarray(img)
    return img


def render_scene_batched(config, geometry, poses):
    scenes = []
    rooms = [r for _, r in poses]
    rooms = np.unique(rooms)

    objects = {}
    for room in rooms:
        mesh = geometry[room]
        objs = create_objects(mesh)
        objects[room] = objs

    img_size = config.RENDER.IMG_SIZE
    for pose, room in poses:
        pose = torch.Tensor(pose)
        camera = create_camera(pose, img_size)
        scenes.append(pyredner.Scene(camera=camera, objects=objects[room]))

    imgs = pyredner.render_g_buffer(
        scenes, [pyredner.channels.depth], device=device, num_samples=1
    )
    # flip x axis for equivalent to a flipped x'
    imgs = imgs.flip(dims=[2])
    imgs = imgs.cpu().squeeze(3).numpy()
    imgs = np.ascontiguousarray(imgs)
    return imgs


def render_semantic_batched(config, geometry, poses):
    scenes = []
    rooms = [r for _, r in poses]
    rooms = np.unique(rooms)

    objects = {}
    for room in rooms:
        mesh = geometry[room]
        objs = create_objects(mesh)
        objects[room] = objs

    img_size = config.RENDER.IMG_SIZE
    for pose, room in poses:
        pose = torch.Tensor(pose)
        camera = create_camera(pose, img_size)
        scenes.append(pyredner.Scene(camera=camera, objects=objects[room]))

    imgs = pyredner.render_g_buffer(
        scenes, [pyredner.channels.shape_id], device=device, num_samples=1
    )
    # flip x axis for equivalent to a flipped x'
    imgs = imgs.flip(dims=[2])
    imgs = imgs.cpu().squeeze(3).numpy()
    imgs = np.ascontiguousarray(imgs)
    return imgs
