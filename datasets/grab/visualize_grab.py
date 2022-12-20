""" Visualizes whole-body from GRAB in MeshViewer """

import sys

sys.path.append(".")
sys.path.append("..")
import argparse
import glob
import os

import numpy as np
import smplx
import torch
from tqdm import tqdm

from training_tools.objectmodel import ObjectModel
from tools.mesh_viewer import Mesh, MeshViewer
from training_tools.utils import DotDict, euler, params2torch, parse_npz, to_cpu, colors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_sequences(cfg):
    """Create visualization of the whole body in MeshViewer."""
    grab_path = cfg.grab_path
    all_seqs = glob.glob(grab_path + "/*/*/*.npz")
    mv = MeshViewer(offscreen=False)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
    camera_pose[:3, 3] = np.array([-0.5, -4.0, 1.5])
    mv.update_camera_pose(camera_pose)

    choice = np.random.choice(len(all_seqs), 10, replace=False)
    for i in tqdm(choice):
        vis_sequence(cfg, all_seqs[i], mv)
    mv.close_viewer()


def vis_sequence(cfg, sequence, mv):
    """Visualize given sequence of events, according to the configuration."""
    grab_path = cfg.grab_path
    seq_data = parse_npz(sequence)
    n_comps = seq_data["n_comps"]
    gender = seq_data["gender"]

    T = seq_data.n_frames

    sbj_mesh = os.path.join(grab_path, ".", seq_data.body.vtemp)
    sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

    sbj_m = smplx.create(
        model_path=cfg.model_path,
        model_type="smplx",
        gender=gender,
        num_pca_comps=n_comps,
        v_template=sbj_vtemp,
        batch_size=T,
    )

    sbj_parms = params2torch(seq_data.body.params)
    verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)
    joints_sbj = to_cpu(sbj_m(**sbj_parms).joints)

    obj_mesh = os.path.join(grab_path, ".", seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)

    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp, batch_size=T)

    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)

    table_mesh = os.path.join(grab_path, ".", seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh)
    table_vtemp = np.array(table_mesh.vertices)
    table_m = ObjectModel(v_template=table_vtemp, batch_size=T)

    table_parms = params2torch(seq_data.table.params)
    verts_table = to_cpu(table_m(**table_parms).vertices)

    skip_frame = 4
    for frame in range(0, T, skip_frame):
        o_mesh = Mesh(
            vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors["yellow"]
        )
        o_mesh.set_vertex_colors(
            vc=colors["red"], vertex_ids=seq_data["contact"]["object"][frame] > 0
        )

        s_mesh = Mesh(
            vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors["pink"], smooth=True
        )
        s_mesh.set_vertex_colors(
            vc=colors["red"], vertex_ids=seq_data["contact"]["body"][frame] > 0
        )
        s_mesh_wf = Mesh(
            vertices=verts_sbj[frame],
            faces=sbj_m.faces,
            vc=colors["grey"],
            wireframe=True,
        )

        j_mesh = Mesh(vertices=joints_sbj[frame], vc=colors["green"], smooth=True)
        t_mesh = Mesh(
            vertices=verts_table[frame], faces=table_mesh.faces, vc=colors["white"]
        )
        mv.set_static_meshes([o_mesh, j_mesh, s_mesh, s_mesh_wf, t_mesh])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAB-visualize")
    parser.add_argument(
        "--grab-path",
        default="_SOURCE_DATA/GRAB/GRAB-data/",
        type=str,
        help="The path to the downloaded grab data",
    )
    parser.add_argument(
        "--model-path",
        default="_BODY_MODELS/models/",
        type=str,
        help="The path to the folder containing smpl models",
    )

    args = parser.parse_args()
    grab_path = args.grab_path
    model_path = args.model_path

    cfg = {
        "grab_path": grab_path,
        "model_path": model_path,
    }
    cfg = DotDict(cfg)
    visualize_sequences(cfg)
