import pyredner
import torch


class PoseConvergenceChecker:
    def __init__(self, config):
        self.best_loss = 1e5
        self.best_pose = None
        self.converge_count = 0
        self.converge_threshold = config.POSE_REFINE.CONVERGANCE_THRESHOLD
        self.converge_patience = config.POSE_REFINE.CONVERGANCE_PATIENCE

    def has_converged(self, current_loss, current_pose):
        delta = 0
        if current_loss < self.best_loss:
            delta = self.best_loss - current_loss
            self.best_pose = current_pose
            self.best_loss = current_loss
        if delta < self.converge_threshold:
            self.converge_count += 1
        else:
            self.converge_count = 0

        return self.converge_count > self.converge_patience


def init_objects_at_pose(pose, mesh, device):
    # Creates pyrender objects from the mesh at the specified pose
    objects = []
    material = pyredner.Material()
    for verts, faces in zip(mesh.verts_list(), mesh.faces_list()):
        verts = verts.to(device) - pose.unsqueeze(0)
        faces = faces.to(device)
        objects.append(pyredner.Object(verts, faces.int(), material))
    vertices = [obj.vertices.clone() for obj in objects]

    return objects, vertices


def init_camera_at_origin(config):
    # Creates a pyrender camera at the origin
    origin = torch.Tensor([0.0, 0.0, 0.0])
    # look at x axis equivalent to an offset x'
    look_at = torch.Tensor([1, 0, 0])
    up = torch.Tensor([0.0, 0.0, 1.0])
    camera = pyredner.Camera(
        position=origin,
        look_at=look_at,
        up=up,
        camera_type=pyredner.camera_type.panorama,
        resolution=config.POSE_REFINE.RENDER_SIZE,
    )
    return camera


def init_optimiser(config, params):
    optimiser = torch.optim.Adam(params, lr=config.POSE_REFINE.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=config.POSE_REFINE.SCHEDULER_PATIENCE,
        threshold=config.POSE_REFINE.SCHEDULER_THRESHOLD,
        factor=config.POSE_REFINE.SCHEDULER_DECAY,
    )
    return optimiser, scheduler


def render_at_pose(camera, objects, vertices, pose):
    for i in range(len(objects)):
        objects[i].vertices = vertices[i] - pose * 1000

    scene = pyredner.Scene(camera=camera, objects=objects)
    img = pyredner.render_g_buffer(scene, [pyredner.channels.depth], device=pose.device)
    img = img.flip(dims=[1])
    return img.squeeze(2)

