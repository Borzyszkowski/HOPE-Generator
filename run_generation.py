""" This script generates dynamic whole-body grasps for the selected dataset """

import argparse
import os
import sys

sys.path.append(".")
sys.path.append("..")
import time

import numpy as np
import smplx
import torch
from bps_torch.bps import bps_torch
from loguru import logger
from omegaconf import OmegaConf
from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.colors import name_to_rgb
from smplx import SMPLXLayer

from data_preparation.mnet_dataloader import LoadData, build_dataloader
from datasets.oakink.oikit.oi_shape.oi_shape import OakInkShape
from models.cvae import gnet_model
from models.mlp import mnet_model
from models.model_utils import parms_6D2full
from models.motion_module import motion_module
from training_tools.objectmodel import ObjectModel
from training_tools.utils import (LOGGER_DEFAULT_FORMAT, d62rotmat, makepath,
                                  rotate, rotmat2aa, rotmul, smplx_loc2glob,
                                  to_cpu, to_tensor)
from training_tools.vis_tools import get_ground, sp_animation

cdir = os.path.dirname(sys.argv[0])


class HopeGenerator:
    def __init__(self, cfg_motion, cfg_static):

        self.dtype = torch.float32
        self.cfg = cfg_motion
        torch.manual_seed(cfg_motion.seed)

        # Initialize the logger
        makepath(cfg_motion.work_dir, isfile=False)
        logger_path = makepath(
            os.path.join(cfg_motion.work_dir, "V00_GNet_MNet.log"), isfile=True
        )
        logger.add(logger_path, backtrace=True, diagnose=True)
        logger.add(
            lambda x: x,
            level=cfg_motion.logger_level.upper(),
            colorize=True,
            format=LOGGER_DEFAULT_FORMAT,
        )
        self.logger = logger.info

        # Use GPU with CUDA
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        self.data_info = {}
        self.load_data()
        self.body_model_cfg = cfg_motion.body_model
        self.predict_offsets = cfg_motion.get("predict_offsets", False)
        self.logger(f"Predict offsets: {self.predict_offsets}")
        self.use_exp = cfg_motion.get("use_exp", 0)
        self.logger(f"Use exp function on distances: {self.use_exp}")

        # Initialize the body model
        model_path = os.path.join(
            self.body_model_cfg.get("model_path", "data/models"), "smplx"
        )
        self.body_model = SMPLXLayer(
            model_path=model_path,
            gender="neutral",
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)
        self.female_model = SMPLXLayer(
            model_path=model_path,
            gender="female",
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)
        self.male_model = SMPLXLayer(
            model_path=model_path,
            gender="male",
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        # Initialize the object model
        self.object_model = ObjectModel().to(self.device)

        # Create the network
        self.n_out_frames = self.cfg.network.n_out_frames
        self.network_motion = mnet_model(**cfg_motion.network.mnet_model).to(
            self.device
        )
        self.network_static = gnet_model(**cfg_static.network.gnet_model).to(
            self.device
        )

        # Restore the pre-trained models
        self.cfg_static = cfg_static
        self.network_static.cfg = cfg_static
        self.network_motion.cfg = cfg_motion
        self.bps_torch = bps_torch()

        self.network_motion.load_state_dict(
            torch.load(cfg_motion.best_model, map_location=self.device), strict=False
        )
        self.logger(
            "Restored motion grasp trained model from %s" % cfg_motion.best_model
        )

        self.network_static.load_state_dict(
            torch.load(cfg_static.best_model, map_location=self.device), strict=False
        )
        self.logger(
            "Restored static grasp trained model from %s" % cfg_static.best_model
        )

        # Setup the training losses
        loss_cfg = self.cfg.get("losses", {})
        self.verts_ids = to_tensor(
            np.load(f"{cdir}/{self.cfg.datasets.verts_sampled}"), dtype=torch.long
        )
        self.rhand_idx = torch.from_numpy(np.load(f"{cdir}/{loss_cfg.rh2smplx_idx}"))
        self.rh_ids_sampled = torch.tensor(
            np.where([id in self.rhand_idx for id in self.verts_ids])[0]
        ).to(torch.long)

    def load_data(self, ds_name="test"):
        """
        Loads data for the specified split (defaults to 'test') and stores it in the instance's attributes.
        """
        self.logger("Base dataset_dir is %s" % self.cfg.datasets.dataset_dir)

        # Loads the test data
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name] = {}
        self.data_info[ds_name]["frame_names"] = ds_test.frame_names
        self.data_info[ds_name]["frame_sbjs"] = ds_test.frame_sbjs
        self.data_info[ds_name]["frame_objs"] = ds_test.frame_objs

        # Computes the staring chunk of data as well as body and object info
        ch_start = (
            np.array(
                [
                    int(name.split("_")[-1])
                    for name in self.data_info[ds_name]["frame_names"][:, 10]
                ]
            )
            == 0
        )
        self.data_info[ds_name]["chunk_starts"] = ch_start
        self.data_info["body_vtmp"] = ds_test.sbj_vtemp
        self.data_info["body_betas"] = ds_test.sbj_betas
        self.data_info["obj_verts"] = ds_test.obj_verts
        self.data_info["obj_info"] = ds_test.obj_info
        self.data_info["sbj_info"] = ds_test.sbj_info

        # Builds the dataloader
        self.ds_test = build_dataloader(ds_test, split="test", cfg=self.cfg.datasets)
        self.bps = ds_test.bps

    def forward(self, x):
        """
        Perform forward pass of the motion prediction model on a batch of input data.

        Args:
            x: A dictionary containing the input data for the model.
        Returns:
            A dictionary containing the outputs of the model.
        """

        bs = x["transl"].shape[0]
        pf = self.cfg.network.previous_frames

        dec_x = {}
        dec_x["fullpose"] = x["fullpose_rotmat"][:, 11 - pf : 11, :, :2, :]
        dec_x["transl"] = x["transl"][:, 11 - pf : 11]
        dec_x["betas"] = x["betas"]

        verts2last = (
            x["verts"][:, 10:11, self.rh_ids_sampled]
            - x["verts"][:, -1:, self.rh_ids_sampled]
        )

        if self.use_exp == 0 or self.use_exp != -1:
            dec_x["vel"] = torch.exp(
                -self.use_exp * x["velocity"][:, 10:11].norm(dim=-1)
            )
            dec_x["verts_to_last_dist"] = torch.exp(
                -self.use_exp * verts2last.norm(dim=-1)
            )
        else:
            dec_x["vel"] = x["velocity"][:, 10:11].norm(dim=-1)
            dec_x["verts_to_last_dist"] = verts2last.norm(dim=-1)

        dec_x["vel"] = x["velocity"][:, 10:11]
        dec_x["verts"] = x["verts"][:, 10:11]
        dec_x["verts_to_rh"] = verts2last
        dec_x["bps_rh"] = x["bps_rh_glob"]
        dec_x = torch.cat(
            [v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1
        )

        pose, trans, dist, rh2last = self.network_motion(dec_x)

        if self.predict_offsets:
            pose_rotmat = d62rotmat(pose).reshape(bs, self.n_out_frames, -1, 3, 3)
            pose = torch.matmul(pose_rotmat, x["fullpose_rotmat"][:, 10:11])
            trans = trans + torch.repeat_interleave(
                x["transl"][:, 10:11], self.n_out_frames, dim=1
            ).reshape(trans.shape)

        pose = pose.reshape(bs * self.n_out_frames, -1)
        trans = trans.reshape(bs * self.n_out_frames, -1)
        d62rot = pose.shape[-1] == 330
        body_params = parms_6D2full(pose, trans, d62rot=d62rot)

        results = {}
        results["body_params"] = body_params
        results["dist"] = dist
        results["rh2last"] = rh2last
        return results

    def infer(self, x):
        """
        Perform inference on a batch of input data using the static and motion prediction models.

        Args:
            x: A dictionary containing the input data for the model.
        Returns:
            A dictionary containing the outputs of the model.
        """

        bs = x["transl"].shape[0]
        dec_x = {}
        dec_x["betas"] = x["betas"]
        dec_x["transl_obj"] = x["transl_obj"]
        dec_x["bps_obj"] = x["bps_obj_glob"].reshape(1, -1, 3).norm(dim=-1)

        z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros(
                [1, self.cfg_static.network.gnet_model.latentD], requires_grad=False
            )
            .to(self.device)
            .type(self.dtype),
            scale=torch.ones(
                [1, self.cfg_static.network.gnet_model.latentD], requires_grad=False
            )
            .to(self.device)
            .type(self.dtype),
        )

        z_enc_s = z_enc.rsample()
        dec_x["z"] = z_enc_s

        dec_x = torch.cat(
            [v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1
        )

        net_output = self.network_static.decode(dec_x)

        pose, trans = net_output["pose"], net_output["trans"]

        rnet_in, cnet_output, m_refnet_params, f_refnet_params = self.prepare_rnet(
            x, pose, trans
        )

        results = {}
        results["z_enc"] = {"mean": z_enc.mean, "std": z_enc.scale}

        cnet_output.update(net_output)
        results["cnet"] = cnet_output
        results["cnet_f"] = f_refnet_params
        results["cnet_m"] = m_refnet_params

        return results

    def prepare_rnet(self, batch, pose, trans):
        """
        Prepare the input and output for the motion model (refnet).

        Args:
            batch: A dictionary containing the input data for the model.
            pose: A tensor of shape (batch_size, 330) containing the predicted pose for the input sequence.
            trans: A tensor of shape (batch_size, 3) containing the predicted translation for the input sequence.

        Returns:
            A tuple containing:
            - rnet_in: A tensor of shape (batch_size, n_out_frames, 6, 330) with input for the motion model.
            - cnet_output: A dictionary containing the output of the static model.
        """

        d62rot = pose.shape[-1] == 330
        bparams = parms_6D2full(pose, trans, d62rot=d62rot)

        genders = batch["gender"]
        males = genders == 1
        females = ~males

        v_template = batch["sbj_vtemp"].to(self.device)

        FN = sum(females)
        MN = sum(males)

        f_refnet_params = {}
        m_refnet_params = {}
        cnet_output = {}
        refnet_in = {}

        R_rh_glob = smplx_loc2glob(bparams["fullpose_rotmat"])[:, 21]
        rh_bps = rotate(self.bps["rh"].to(self.device), R_rh_glob)

        if FN > 0:
            f_params = {k: v[females] for k, v in bparams.items()}
            f_params["v_template"] = v_template[females]
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices

            cnet_output["f_verts_full"] = f_verts
            cnet_output["f_params"] = f_params

            f_refnet_params["f_verts2obj"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][females],
                feature_type=["deltas"],
                custom_basis=f_verts[:, self.verts_ids],
            )["deltas"]
            f_refnet_params["f_rh2obj"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][females],
                feature_type=["deltas"],
                custom_basis=f_verts[:, self.rhand_idx],
            )["deltas"]

            f_rh_bps = rh_bps[females] + f_output.joints[:, 43:44]

            f_refnet_params["f_bps_obj_rh"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][females],
                feature_type=["deltas"],
                custom_basis=f_rh_bps,
            )["deltas"]

            refnet_in["f_refnet_in"] = torch.cat(
                [
                    f_params["fullpose_rotmat"][:, :, :2, :]
                    .reshape(FN, -1)
                    .to(self.device),
                    f_params["transl"].reshape(FN, -1).to(self.device),
                ]
                + [v.reshape(FN, -1).to(self.device) for v in f_refnet_params.values()],
                dim=1,
            )

        if MN > 0:
            m_params = {k: v[males] for k, v in bparams.items()}
            m_params["v_template"] = v_template[males]
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            cnet_output["m_verts_full"] = m_verts
            cnet_output["m_params"] = m_params

            m_refnet_params["m_verts2obj"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][males],
                feature_type=["deltas"],
                custom_basis=m_verts[:, self.verts_ids],
            )["deltas"]
            m_refnet_params["m_rh2obj"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][males],
                feature_type=["deltas"],
                custom_basis=m_verts[:, self.rhand_idx],
            )["deltas"]

            m_rh_bps = rh_bps[males] + m_output.joints[:, 43:44]

            m_refnet_params["m_bps_obj_rh"] = self.bps_torch.encode(
                x=batch["verts_obj"][:, -1][males],
                feature_type=["deltas"],
                custom_basis=m_rh_bps,
            )["deltas"]

            refnet_in["m_refnet_in"] = torch.cat(
                [
                    m_params["fullpose_rotmat"][:, :, :2, :]
                    .reshape(MN, -1)
                    .to(self.device),
                    m_params["transl"].reshape(MN, -1).to(self.device),
                ]
                + [v.reshape(MN, -1).to(self.device) for v in m_refnet_params.values()],
                dim=1,
            )

        refnet_in = torch.cat([v for v in refnet_in.values()], dim=0)
        return refnet_in, cnet_output, m_refnet_params, f_refnet_params

    def run_generation(self, dataset_choice):
        """
        Run the generation process for the specified dataset.

        Args:
        dataset_choice (str): The dataset to generate data for. Can be either 'OakInk' or 'GRAB'.

        Raises:
        Exception: If the given dataset name is not supported.
        """

        self.logger(f"Generating data for the {dataset_choice} dataset")

        self.network_motion.eval()
        self.network_static.eval()
        device = self.device

        ds_name = "test"
        data = self.ds_test

        base_movie_path = os.path.join(self.cfg.results_base_dir, self.cfg.expr_ID)

        chunk_starts = self.data_info[ds_name]["chunk_starts"]

        for batch_id, batch in enumerate(data):

            if not chunk_starts[batch_id]:
                continue

            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            gender = batch["gender"].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            sbj_m.v_template = batch["sbj_vtemp"].to(sbj_m.v_template.device)

            name = (
                self.data_info[ds_name]["frame_names"][batch["idx"].to(torch.long)][0][
                    :-2
                ].split("/s")
            )[-1]

            if dataset_choice == "OakInk":
                oi_shape = OakInkShape(
                    category="teapot", intent_mode="use", data_split="test"
                )
                for oid, obj in oi_shape.obj_warehouse.items():
                    self.logger(f"Generation for the object {oid}")
                    obj_verts = obj["verts"]
                    obj_faces = obj["faces"]
                    obj_m = ObjectModel(v_template=obj_verts).to(device)
                    obj_mesh = Mesh(v=obj_verts, f=obj_faces)
                    sequence_name = "s" + name[0] + f"_{oid}_" + name.split("_")[-1]
                    self.generate(
                        batch, sbj_m, obj_m, obj_mesh, sequence_name, base_movie_path
                    )

            elif dataset_choice == "GRAB":
                obj_name = (
                    self.data_info[ds_name]["frame_names"][batch["idx"].to(torch.long)][
                        0
                    ]
                    .split("/")[-1]
                    .split("_")[0]
                )
                obj_path = os.path.join(
                    self.cfg.datasets.grab_path,
                    "tools/object_meshes/contact_meshes",
                    f"{obj_name}.ply",
                )

                obj_mesh = Mesh(filename=obj_path)
                obj_verts = torch.from_numpy(obj_mesh.v)

                obj_m = ObjectModel(v_template=obj_verts).to(device)
                obj_m.faces = obj_mesh.f
                sequence_name = "s" + self.data_info[ds_name]["frame_names"][
                    batch["idx"].to(torch.long)
                ][0][:-2].split("/s")[-1].replace("/", "_")
                self.generate(
                    batch, sbj_m, obj_m, obj_mesh, sequence_name, base_movie_path
                )

            else:
                raise Exception(
                    f"The given dataset name {dataset_choice} is not supported"
                )

    def generate(self, batch, sbj_m, obj_m, obj_mesh, sequence_name, base_movie_path):
        """
        Generate and save the motion and static meshes for the given batch.

        Args:
        batch (dict): A dictionary containing the data for the current batch.
        sbj_m (ObjectModel): The subject model.
        obj_m (ObjectModel): The object model.
        obj_mesh (Mesh): The mesh of the object.
        sequence_name (str): The name of the sequence to be generated.
        base_movie_path (str): The base path to save the generated data.

        Attributes:
        visualize (bool): Whether to visualize the generated meshes.
        save_meshes (bool): Whether to save the generated meshes.
        num_samples (int): The number of samples to generate.
        """

        visualize = False
        save_meshes = True
        num_samples = 1

        if visualize:
            mvs = MeshViewers()
        else:
            mvs = None

        mov_count = 1
        motion_path = os.path.join(
            base_movie_path,
            "static_and_motion_" + str(mov_count),
            sequence_name + "_motion.html",
        )
        grasp_path = os.path.join(
            base_movie_path,
            "static_and_motion_" + str(mov_count),
            sequence_name + "_grasp.html",
        )
        motion_meshes_path = os.path.join(
            base_movie_path,
            "static_and_motion_" + str(mov_count),
            sequence_name + "_motion_meshes",
        )
        static_meshes_path = os.path.join(
            base_movie_path,
            "static_and_motion_" + str(mov_count),
            sequence_name + "_static_meshes",
        )

        while os.path.exists(motion_path):
            mov_count += 1
            motion_path = os.path.join(
                base_movie_path,
                "static_and_motion_" + str(mov_count),
                sequence_name + "_motion.html",
            )
            grasp_path = os.path.join(
                base_movie_path,
                "static_and_motion_" + str(mov_count),
                sequence_name + "_grasp.html",
            )
            motion_meshes_path = os.path.join(
                base_movie_path,
                "static_and_motion_" + str(mov_count),
                sequence_name + "_motion_meshes",
            )
            static_meshes_path = os.path.join(
                base_movie_path,
                "static_and_motion_" + str(mov_count),
                sequence_name + "_static_meshes",
            )

        if save_meshes:
            makepath(motion_meshes_path)
            makepath(static_meshes_path)

        """ Run grasping optimization (GNet) """
        from training_tools.gnet_optim import GNetOptim as FitSmplxStatic

        fit_smplx_static = FitSmplxStatic(
            sbj_model=sbj_m, obj_model=obj_m, cfg=self.cfg, verbose=True
        )

        grnd_mesh, cage, axis_l = get_ground()
        sp_anim_static = sp_animation()
        static_grasp_results = []
        batch_static = {}

        for k, v in batch.items():
            if v.ndim > 1:
                if v.shape[1] == 22:
                    batch_static[k] = v.clone()[:, -1]
                else:
                    batch_static[k] = v.clone()
            else:
                batch_static[k] = v.clone()

        rel_transl = batch_static["transl_obj"].clone()
        rel_transl[:, 1] -= rel_transl[:, 1]
        batch_static["transl_obj"] -= rel_transl

        # Generate static grasping
        for i in range(num_samples):
            print(f"{sequence_name} -- {i}/{num_samples - 1} frames")
            net_output = self.infer(batch_static)

            optim_output = fit_smplx_static.fitting(batch_static, net_output)

            static_grasp_results.append(optim_output)

            sbj_cnet = Mesh(
                v=to_cpu(optim_output["cnet_verts"][0]),
                f=sbj_m.faces,
                vc=name_to_rgb["pink"],
            )
            sbj_opt = Mesh(
                v=to_cpu(optim_output["opt_verts"][0]),
                f=sbj_m.faces,
                vc=name_to_rgb["green"],
            )
            obj_i = Mesh(
                to_cpu(fit_smplx_static.obj_verts[0]),
                f=obj_mesh.f,
                vc=name_to_rgb["yellow"],
            )

            if visualize:
                mvs[0][0].set_static_meshes([sbj_cnet, sbj_opt, obj_i])
                time.sleep(1)

            if save_meshes:
                sbj_cnet.write_ply(static_meshes_path + f"/{i:04d}_sbj_coarse.ply")
                sbj_opt.write_ply(static_meshes_path + f"/{i:04d}_sbj_refine.ply")
                obj_i.write_ply(static_meshes_path + f"/{i:04d}_obj.ply")

            sp_anim_static.add_frame(
                [sbj_cnet, sbj_opt, obj_i, grnd_mesh],
                ["coarse_grasp", "refined_grasp", "object", "ground_mesh"],
            )

        # Take one of the samples and update the batch based on it
        final_grasp = static_grasp_results[0]

        # Transformation from vicon to smplx coordinate frame
        R_v2s = (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            .reshape(1, 3, 3)
            .to(self.device)
        )

        rel_rot = batch["rel_rot"]

        R_rot = torch.matmul(rel_rot, R_v2s)
        root_offset = smplx.lbs.vertices2joints(
            sbj_m.J_regressor, sbj_m.v_template[0].view(1, -1, 3)
        )[:, 0]

        fpose_sbj_rotmat = final_grasp["fullpose_rotmat"].clone()
        global_orient_sbj_rel = rotmul(R_rot, fpose_sbj_rotmat[:, 0])
        fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

        trans_sbj_rel = (
            rotate((final_grasp["transl"] + root_offset), R_rot)
            - root_offset
            + rel_transl
        )

        batch["transl"][:, -1] = trans_sbj_rel
        batch["fullpose"][:, -1] = rotmat2aa(fpose_sbj_rotmat).reshape(1, -1)
        batch["fullpose_rotmat"][:, -1] = fpose_sbj_rotmat

        grasp_sbj_params = parms_6D2full(fpose_sbj_rotmat, trans_sbj_rel, d62rot=False)

        grasp_sbj_output = sbj_m(**grasp_sbj_params)
        grasp_verts_sampled = grasp_sbj_output.vertices[:, self.verts_ids]

        batch["verts"][:, -1] = grasp_verts_sampled
        input_data = {k: batch[k].to(self.device) for k in batch.keys()}
        grasping_motion = motion_module(
            input_data, sbj_model=sbj_m, obj_model=obj_m, cfg=self.cfg
        )

        grasping_motion.bps = self.bps
        grasping_motion.mvs = mvs

        input_data = grasping_motion.get_current_params()

        """ Run motion optimization (MNet) """
        from training_tools.mnet_optim import MNetOpt as FitSmplxMotion

        fit_smplx_motion = FitSmplxMotion(
            sbj_model=sbj_m, obj_model=obj_m, cfg=self.cfg
        )

        fit_smplx_motion.stop = False
        fit_smplx_motion.mvs = mvs

        # Generate grasping motion
        while grasping_motion.num_iters < 10:
            net_output = self.forward(input_data)

            fit_results = fit_smplx_motion.fitting(input_data, net_output)
            grasping_motion(fit_results)

            if fit_smplx_motion.stop:
                break

            input_data = grasping_motion.get_current_params()
            min_dist2obj = (
                input_data["verts2obj"][:, 10].reshape(-1, 3).norm(dim=-1).min()
            )
            min_dist_offset = net_output["rh2last"].reshape(-1, 3).norm(dim=-1).max()

            if min_dist2obj < 0.003 and min_dist_offset < 0.2:
                break

        # Store subject and object parameters
        sbj_params = {k: v.clone() for k, v in grasping_motion.sbj_params.items()}
        obj_params = {k: v.clone() for k, v in grasping_motion.obj_params.items()}

        # Get subject output mesh
        sbj_output_glob = sbj_m(**sbj_params)
        verts_sbj_glob = sbj_output_glob.vertices

        # Get object output mesh
        obj_out_glob = obj_m(**obj_params)
        verts_obj_glob = obj_out_glob.vertices
        grnd_mesh, cage, axis_l = get_ground()

        # Saves the generated static and motion meshes
        sp_anim_motion = sp_animation()
        for i in range(grasping_motion.n_frames - 1):
            sbj_i = Mesh(
                v=to_cpu(verts_sbj_glob[i]), f=sbj_m.faces, vc=name_to_rgb["pink"]
            )
            obj_i = Mesh(
                v=to_cpu(verts_obj_glob[i]), f=obj_mesh.f, vc=name_to_rgb["yellow"]
            )

            if visualize:
                mvs[0][0].set_static_meshes([sbj_i, obj_i, grnd_mesh])
                mvs[0][0].set_static_lines([grasping_motion.axis_l])

            if save_meshes:
                sbj_i.write_ply(motion_meshes_path + f"/{i:04d}_sbj.ply")
                obj_i.write_ply(motion_meshes_path + f"/{i:04d}_obj.ply")

            sp_anim_motion.add_frame(
                [sbj_i, obj_i, grnd_mesh], ["sbj_mesh", "obj_mesh", "ground_mesh"]
            )

        # Saves the animations for generated meshes
        if save_meshes:
            sp_anim_motion.save_animation(motion_path)
            sp_anim_static.save_animation(grasp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOPE-Generator")
    parser.add_argument(
        "--work-dir",
        default="_RESULTS/HOPE-Generator",
        type=str,
        help="The path to the folder to save results",
    )
    parser.add_argument(
        "--grab-path",
        default="_SOURCE_DATA/GRAB/GRAB-data/",
        type=str,
        help="The path to the folder that contains GRAB data",
    )
    parser.add_argument(
        "--preprocessed-data-path",
        default="_DATA/",
        type=str,
        help="The path to the folder containing preprocessed data for GNet and MNet",
    )
    parser.add_argument(
        "--smplx-path",
        default="_BODY_MODELS/models/",
        type=str,
        help="The path to the folder containing SMPLX models",
    )
    parser.add_argument(
        "--dataset-choice",
        default="GRAB",
        type=str,
        choices=["OakInk", "GRAB"],
        help="The choice of dataset for which the grasps should be generated",
    )

    # Parses the given arguments
    cmd_args = parser.parse_args()
    dataset_choice = cmd_args.dataset_choice

    # Restores the pre-trained models
    best_gnet = f"{cdir}/models/GNet_model.pt"
    best_mnet = f"{cdir}/models/MNet_model.pt"

    # Configuration for MNet (motion generation)
    cfg_path_motion = f"{cdir}/configs/MNet_orig.yaml"
    cfg_motion = OmegaConf.load(cfg_path_motion)
    cfg_motion.batch_size = 1
    cfg_motion.best_model = best_mnet

    cfg_motion.datasets.grab_path = cmd_args.grab_path
    cfg_motion.datasets.source_grab_path = cmd_args.grab_path
    cfg_motion.datasets.dataset_dir = os.path.join(
        cmd_args.preprocessed_data_path, "MNet_data"
    )
    cfg_motion.datasets.preprocessed_data_path = cmd_args.preprocessed_data_path

    cfg_motion.output_folder = cmd_args.work_dir
    cfg_motion.results_base_dir = os.path.join(cfg_motion.output_folder, "results")
    cfg_motion.work_dir = os.path.join(cfg_motion.output_folder, "HOPEGEN_test")
    cfg_motion.body_model.model_path = cmd_args.smplx_path

    # Configuration for GNet (grasp generation)
    cfg_path_static = f"{cdir}/configs/GNet_orig.yaml"
    cfg_static = OmegaConf.load(cfg_path_static)
    cfg_static.batch_size = 1
    cfg_static.best_model = best_gnet

    cfg_static.datasets.dataset_dir = os.path.join(
        cmd_args.preprocessed_data_path, "GNet_data"
    )
    cfg_static.datasets.preprocessed_data_path = cmd_args.preprocessed_data_path

    cfg_static.output_folder = cmd_args.work_dir
    cfg_static.results_base_dir = os.path.join(cfg_static.output_folder, "results")
    cfg_static.work_dir = os.path.join(cfg_static.output_folder, "HOPEGEN_test")
    cfg_static.body_model.model_path = cmd_args.smplx_path

    # Initialize and run generation (inference), given motion and static configuration
    generator = HopeGenerator(cfg_motion, cfg_static)
    generator.run_generation(dataset_choice)
