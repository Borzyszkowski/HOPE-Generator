""" Preprocessing script for GNet """

import glob
import os
import shutil
import sys
from datetime import datetime

sys.path.append("")
sys.path.append("..")

import numpy as np
import smplx
import torch
from bps_torch.bps import bps_torch
from psbody.mesh import Mesh
import _pickle as pickle

from tqdm import tqdm

from training_tools.cfg_parser import Config
from training_tools.objectmodel import ObjectModel
from training_tools.utils import (aa2rotmat, append2dict, makelogger,
                                  makepath, params2torch, parse_npz,
                                  prepare_params, rotate, rotmat2aa, rotmul,
                                  to_cpu, to_tensor, torch2np)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTENTS = ["lift", "pass", "offhand", "use", "all"]


class GNetDataSet(object):
    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        self.cwd = os.path.dirname(sys.argv[0])
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, "gnet_preprocessing.log")
            self.logger = makelogger(log_dir=log_dir, mode="a").info
        else:
            self.logger = logger
        self.logger("Starting data preprocessing !")

        # assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger(
            "intent:%s --> processing %s sequences!" % (self.intent, self.intent)
        )

        if cfg.splits is None:
            self.splits = {"test": 0.1, "val": 0.05, "train": 0.85}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits

        self.all_seqs = glob.glob(os.path.join(self.grab_path, "*/*/*.npz"))

        ### to be filled
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {"test": [], "val": [], "train": []}

        ### group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger("Total sequences: %d" % len(self.all_seqs))
        self.logger("Selected sequences: %d" % len(self.selected_seqs))
        self.logger(
            "Number of sequences in each data split : train: %d , test: %d , val: %d"
            % (
                len(self.split_seqs["train"]),
                len(self.split_seqs["test"]),
                len(self.split_seqs["val"]),
            )
        )
        ### process the data
        self.data_preprocessing(cfg)

    def data_preprocessing(self, cfg):

        self.obj_info = {}
        self.sbj_info = {}

        bps_path = makepath(os.path.join(cfg.out_path, "bps.pt"), isfile=True)
        bps_orig_path = f"{self.cwd}/../configs/bps.pt"

        self.bps_torch = bps_torch()
        self.bps = torch.load(bps_orig_path)
        shutil.copy2(bps_orig_path, bps_path)
        self.logger(f"loading bps from {bps_orig_path}")

        verts_ids = to_tensor(
            np.load(f"{self.cwd}/../consts/verts_ids_0512.npy"), dtype=torch.long
        )

        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(
            sys.argv[0],
            os.path.join(
                self.out_path,
                os.path.basename(sys.argv[0]).replace(
                    ".py", "_%s.py" % datetime.strftime(stime, "%Y%m%d_%H%M")
                ),
            ),
        )

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        for split in self.split_seqs.keys():
            # split = 'train'
            outfname = makepath(
                os.path.join(cfg.out_path, split, "GNet_data.npy"), isfile=True
            )

            if os.path.exists(outfname):
                self.logger("Results for %s split already exist." % (split))
                continue
            else:
                self.logger("Processing data for %s split." % (split))

            frame_names = []
            n_frames = -1

            GNet_data = {
                "transl": [],
                "fullpose": [],
                "fullpose_rotmat": [],
                "verts": [],
                "verts_obj": [],
                "transl_obj": [],
                "global_orient_obj": [],
                "global_orient_rotmat_obj": [],
                "verts2obj": [],
                "bps_obj_glob": [],
            }

            for sequence in tqdm(self.split_seqs[split]):

                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                sbj_id = seq_data.sbj_id

                n_comps = seq_data.n_comps
                gender = seq_data.gender

                frame_mask = self.filter_contact_frames(seq_data)

                # total selectd frames
                T = frame_mask.sum()
                if T < 1:
                    continue  # if no frame is selected continue to the next sequence

                ##### motion data preparation
                bs = T
                sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)
                obj_info = self.load_obj_verts(obj_name, seq_data, cfg.n_verts_sample)

                with torch.no_grad():
                    sbj_m = smplx.create(
                        model_path=cfg.model_path,
                        model_type="smplx",
                        gender=gender,
                        num_pca_comps=n_comps,
                        v_template=sbj_vtemp,
                        batch_size=bs,
                    )

                    obj_m = ObjectModel(v_template=obj_info["verts"], batch_size=bs)

                    root_offset = smplx.lbs.vertices2joints(
                        sbj_m.J_regressor, sbj_m.v_template.view(1, -1, 3)
                    )[0, 0]

                    rel_offset = seq_data.object.params.transl[frame_mask]
                    rel_offset[:, 2] -= rel_offset[:, 2]

                    ##### batch motion data selection
                    sbj_params = prepare_params(
                        seq_data.body.params, frame_mask, rel_offset
                    )
                    obj_params = prepare_params(
                        seq_data.object.params, frame_mask, rel_offset
                    )

                    sbj_params_orig = params2torch(sbj_params)
                    obj_params_orig = params2torch(obj_params)

                    # transformation from vicon to smplx coordinate frame
                    R_v2s = torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
                    ).reshape(1, 3, 3)

                    motion_sbj, motion_obj, rel_trans = glob2rel(
                        sbj_params_orig,
                        obj_params_orig,
                        R_v2s.transpose(1, 2),
                        root_offset,
                    )

                    sbj_output = sbj_m(**motion_sbj)
                    verts_sbj = sbj_output.vertices

                    obj_out = obj_m(**motion_obj)
                    verts_obj = obj_out.vertices

                    obj_in = {k + "_obj": v for k, v in motion_obj.items()}

                    append2dict(GNet_data, motion_sbj)
                    append2dict(GNet_data, obj_in)

                    GNet_data["verts"].append(to_cpu(verts_sbj[:, verts_ids]))

                    verts2obj = self.bps_torch.encode(
                        x=verts_obj,
                        feature_type=["deltas"],
                        custom_basis=verts_sbj[:, verts_ids],
                    )["deltas"]

                    GNet_data["verts2obj"].append(to_cpu(verts2obj))

                    obj_bps = self.bps["obj"] + motion_obj["transl"].reshape(T, 1, 3)

                    bps_obj = self.bps_torch.encode(
                        x=verts_obj, feature_type=["deltas"], custom_basis=obj_bps
                    )["deltas"]

                    GNet_data["bps_obj_glob"].append(to_cpu(bps_obj))

                    frame_names.extend(
                        [
                            "%s_%s" % (sequence.split(".")[0], fId)
                            for fId in np.arange(T)
                        ]
                    )

            self.logger("Processing for %s split finished" % split)
            self.logger(
                "Total number of frames for %s split is:%d" % (split, len(frame_names))
            )

            out_data = [GNet_data]
            out_data_name = ["GNet_data"]

            for idx, _ in enumerate(out_data):
                data_name = out_data_name[idx]
                out_data[idx] = torch2np(out_data[idx])
                outfname = makepath(
                    os.path.join(self.out_path, split, "%s.npy" % data_name),
                    isfile=True,
                )

                pickle.dump(out_data[idx], open(outfname, "wb"), protocol=4)

            np.savez(
                os.path.join(self.out_path, split, "frame_names.npz"),
                frame_names=frame_names,
            )

            np.save(os.path.join(self.out_path, "obj_info.npy"), self.obj_info)
            np.save(os.path.join(self.out_path, "sbj_info.npy"), self.sbj_info)

    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split("/")[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split("_")[0]

            # filter data based on the motion intent

            if "all" in self.intent:
                pass
            elif "use" in self.intent and any(
                intnt in action_name for intnt in INTENTS[:3]
            ):
                continue
            elif all([item not in action_name for item in self.intent]):
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits["test"]:
                self.split_seqs["test"].append(sequence)
            elif object_name in self.splits["val"]:
                self.split_seqs["val"].append(sequence)
            else:
                self.split_seqs["train"].append(sequence)
                if object_name not in self.splits["train"]:
                    self.splits["train"].append(object_name)

    def filter_contact_frames(self, seq_data):

        table_height = seq_data.object.params.transl[0, 2]
        table_xy = seq_data.object.params.transl[0, :2]
        obj_height = seq_data.object.params.transl[:, 2]
        obj_xy = seq_data.object.params.transl[:, :2]

        contact_array = seq_data.contact.object
        fil2 = np.logical_or(
            (obj_height > table_height + 0.005), (obj_height < table_height - 0.005)
        )
        fil21 = np.logical_and(
            (obj_height > table_height - 0.15), (obj_height < table_height + 0.15)
        )

        fil22 = np.sqrt(np.power(obj_xy - table_xy, 2).sum(-1)) < 0.10

        include_fil = np.isin(contact_array, cfg.include_joints).any(axis=1)
        exclude_fil = ~np.isin(contact_array, cfg.exclude_joints).any(axis=1)
        fil3 = np.logical_and(include_fil, exclude_fil)
        in_contact_frames = fil2 * fil21 * fil22 * fil3

        return in_contact_frames

    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=2048):

        mesh_path = os.path.join(self.grab_path, seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.v)
            faces_obj = np.array(obj_mesh.f)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(
                    verts_obj.shape[0], n_verts_sample, replace=False
                )
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {
                "verts": verts_obj,
                "faces": faces_obj,
                "verts_sample_id": verts_sample_id,
                "verts_sample": verts_sampled,
                "obj_mesh_file": mesh_path,
            }

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, seq_data.body.vtemp)
        betas_path = mesh_path.replace(".ply", "_betas.npy")

        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]["vtemp"]
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).v)
            sbj_betas = np.load(betas_path)
            self.sbj_info[sbj_id] = {
                "vtemp": sbj_vtemp,
                "gender": seq_data.gender,
                "betas": sbj_betas,
            }
        return sbj_vtemp


def glob2rel(motion_sbj, motion_obj, R, root_offset, rel_trans=None):
    fpose_sbj_rotmat = aa2rotmat(motion_sbj["fullpose"])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj["transl"] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj["transl"], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj["global_orient"])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.clone()
        rel_trans[:, 1] -= rel_trans[:, 1]

    motion_sbj["transl"] = to_tensor(trans_sbj_rel)
    motion_sbj["global_orient"] = rotmat2aa(
        to_tensor(global_orient_sbj_rel).squeeze()
    ).squeeze()
    motion_sbj["global_orient_rotmat"] = to_tensor(global_orient_sbj_rel)
    motion_sbj["fullpose"][:, :3] = motion_sbj["global_orient"]
    motion_sbj["fullpose_rotmat"] = fpose_sbj_rotmat

    motion_obj["transl"] = to_tensor(trans_obj_rel)
    motion_obj["global_orient"] = rotmat2aa(
        to_tensor(global_orient_obj_rel).squeeze()
    ).squeeze()
    motion_obj["global_orient_rotmat"] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNet-data-GRAB")

    parser.add_argument(
        "--grab-path",
        default="_SOURCE_DATA/GRAB/GRAB-data/",
        type=str,
        help="The path to the folder that contains GRAB data",
    )
    parser.add_argument(
        "--smplx-path",
        default="_BODY_MODELS/models/",
        type=str,
        help="The path to the folder containing SMPL-X model downloaded from the website",
    )
    parser.add_argument(
        "--out-path",
        default="_DATA/GNet_data/",
        type=str,
        help="The output path to save the preprocessed data",
    )

    cmd_args = parser.parse_args()

    grab_path = cmd_args.grab_path
    model_path = cmd_args.smplx_path
    out_path = cmd_args.out_path

    # split the dataset based on the objects
    grab_splits = {
        "test": ["mug", "camera", "binoculars", "apple", "toothpaste"],
        "val": ["fryingpan", "toothbrush", "elephant", "hand"],
        "train": [],
    }

    cfg = {
        "intent": ["all"],  # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        "save_contact": False,  # if True, will add the contact info to the saved data
        # motion fps (default is 120.)
        "fps": 30.0,
        "past": 10,  # number of past frames to include
        "future": 10,  # number of future frames to include
        ### splits
        "splits": grab_splits,
        ###IO path
        "grab_path": grab_path,
        "out_path": out_path,
        ### number of vertices samples for each object
        "n_verts_sample": 2048,
        ### body and hand model path
        "model_path": model_path,
        ### include/exclude joints
        "include_joints": list(range(41, 53)),
        # 'required_joints' : [16],  # mouth
        "required_joints": list(range(53, 56)),  # thumb
        "exclude_joints": list(range(26, 41)),
        ### bps info
        "r_obj": 0.15,
        "n_obj": 1024,
        "r_sbj": 1.5,
        "n_sbj": 1024,
        "g_size": 20,
        "h_sbj": 2.0,
        "r_rh": 0.2,
        "n_rh": 1024,
        "r_hd": 0.15,
        "n_hd": 2048,
        ### interpolaton params
        "interp_frames": 60,
    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, "../configs/grab_preprocessing_cfg.yaml")
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)

    makepath(cfg.out_path)
    cfg.write_cfg(write_path=cfg.out_path + "/grab_preprocessing_cfg.yaml")

    log_dir = os.path.join(cfg.out_path, "grab_processing.log")
    logger = makelogger(log_dir=log_dir, mode="a").info

    GNetDataSet(cfg, logger)
