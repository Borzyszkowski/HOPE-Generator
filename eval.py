import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import trimesh

import multiprocessing as mp


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def get_all_static_meshes(results_path):
    """
    Returns the list of all generated static meshes

    Input:
        - results_path: str
            path to the results folder
    Output:
        - static_motions: list
            list of all static motions
    """
    obj_path = "0000_obj.ply"
    sbj_path = "0000_sbj_refine.ply"
    all_folders = os.listdir(results_path)
    meshes = []
    for folder in all_folders:
        if folder.endswith("static_meshes"):
            # join all the paths
            obj = os.path.join(results_path, folder, obj_path)
            sbj = os.path.join(results_path, folder, sbj_path)
            meshes.append((obj, sbj))
    return meshes


def mesh_intersection(i, obj_path, sbj_path):
    """
    Computes the intersection of two meshes

    Input:
        - obj_path: str
            path to the object mesh
        - sbj_path: str
            path to the subject mesh
    Output:
        - volume: float, contact: int (0 or 1)
    """
    obj_mesh = trimesh.load(obj_path)
    sbj_mesh = trimesh.load(sbj_path)
    intersection = obj_mesh.intersection(sbj_mesh)
    print(i, "done\n")
    if isinstance(intersection, trimesh.Scene): 
        # no intersection, check for contact on the surface
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object("Subject", sbj_mesh)
        is_contact = collision_manager.in_collision_single(obj_mesh)
        return 0, int(is_contact)
    return np.abs(intersection.volume), 1


def get_statistics(results_path):
    """
    Computes the statistics of the generated meshes

    Input:
        - results_path: str
            path to the results folder
    Output:
        - stats: dict
            dictionary of the statistics
    """
    # get all the static meshes
    meshes = get_all_static_meshes(results_path)
    factor = 1e6 # convert to cm^3
    # compute the intersection
    # volumes = []
    # contacts = []
    st_time = time.time()
    # for obj_path, sbj_path in tqdm(meshes):
    #     vol, contact = mesh_intersection(obj_path, sbj_path)
    #     volumes.append(vol * factor)
    #     contacts.append(contact)

    # do multiprocessing
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(mesh_intersection, args=(i, obj_path, sbj_path))\
                    for i, (obj_path, sbj_path) in enumerate(meshes)]
    results = [p.get() for p in results]
    volumes = [res[0] * factor for res in results]
    contacts = [res[1] for res in results]
    pool.close()
    pool.join()

    print("Time taken: ", time.time() - st_time)
    # compute the statistics
    stats = {}
    stats["mean_volume"] = np.mean(volumes)
    stats["std_volume"] = np.std(volumes)
    stats["contact_ratio"] = np.mean(contacts)

    with open('paths.txt','w+') as f:
        for obj_path, sbj_path in meshes:
            f.write(obj_path+'\n'+sbj_path+'\n')
    with open('volumes.txt','w+') as f:
        for vol in volumes:
            f.write(str(vol)+'\n')
    with open('contacts.txt','w+') as f:
        for contact in contacts:
            f.write(str(contact)+'\n')
    return stats



def v2v(x, y, mean=True):
    dist = np.linalg.norm(x - y, axis=-1) * 1000 # convert to mm
    if mean:
        return dist.mean()
    else:
        return dist


def get_verts_ids(path):    
    verts_feet: str = f'{path}/consts/feet_verts_ids_0512.npy'
    rh2smplx_ids: str = f'{path}/consts/rhand_smplx_ids.npy'
    feet_verts = np.load(verts_feet).astype(np.int8)
    hand_verts = np.load(rh2smplx_ids).astype(np.int8)
    return hand_verts, feet_verts


def get_v2v(dir_path_root, hand_ids, feet_ids, plot=False, plot_part=0):
    static = dir_path_root + '_static_meshes/'
    motion = dir_path_root + '_motion_meshes/'
    refined_gt_path = static + "0000_sbj_refine.ply"
    motion_path = motion + "0050_sbj.ply"
    
    refined_gt = trimesh.load(refined_gt_path).vertices
    refined_gt_hand = refined_gt[hand_ids]
    refined_gt_feet = refined_gt[feet_ids]

    motion_mesh = trimesh.load(motion_path).vertices
    # translate motion mesh to gt
    offset = refined_gt[0] - motion_mesh[0]
    # offset = refined_gt.mean(axis=0) - motion_mesh.mean(axis=0)
    motion_mesh = motion_mesh + offset
    motion_hand = motion_mesh[hand_ids]
    motion_feet = motion_mesh[feet_ids]

    if plot:
        # scatter plot gt with blue, motion with red
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if plot_part == 0:
            ax.scatter(refined_gt[:, 0], refined_gt[:, 1], refined_gt[:, 2], c='b', marker='o')
            ax.scatter(motion_mesh[:, 0], motion_mesh[:, 1], motion_mesh[:, 2], c='r', marker='o')
        elif plot_part == 1:
            ax.scatter(refined_gt_hand[:, 0], refined_gt_hand[:, 1], refined_gt_hand[:, 2], c='b', marker='o')
            ax.scatter(motion_hand[:, 0], motion_hand[:, 1], motion_hand[:, 2], c='r', marker='o')
        else:
            ax.scatter(refined_gt_feet[:, 0], refined_gt_feet[:, 1], refined_gt_feet[:, 2], c='b', marker='o')
            ax.scatter(motion_feet[:, 0], motion_feet[:, 1], motion_feet[:, 2], c='r', marker='o')
        plt.show()

    v2v_hand = v2v(refined_gt_hand, motion_hand, mean=True)
    v2v_feet = v2v(refined_gt_feet, motion_feet, mean=True)
    v2v_body = v2v(refined_gt, motion_mesh, mean=True)

    return v2v_hand, v2v_feet, v2v_body


def get_v2v_stats(root_paths, hand_ids, feet_ids, plot=False):
    v2v_hand = []
    v2v_feet = []
    v2v_body = []
    for root_path in root_paths:
        v2v_hand_, v2v_feet_, v2v_body_ = get_v2v(root_path, hand_ids, feet_ids)
        v2v_hand.append(v2v_hand_)
        v2v_feet.append(v2v_feet_)
        v2v_body.append(v2v_body_)
    
    v2v_mean_hand = np.mean(v2v_hand)
    v2v_std_hand = np.std(v2v_hand)
    v2v_mean_feet = np.mean(v2v_feet)
    v2v_std_feet = np.std(v2v_feet)
    v2v_mean_body = np.mean(v2v_body)
    v2v_std_body = np.std(v2v_body)
    print("Mean v2v hand: ", v2v_mean_hand)
    print("Std v2v hand: ", v2v_std_hand)
    print("Mean v2v feet: ", v2v_mean_feet)
    print("Std v2v feet: ", v2v_std_feet)
    print("Mean v2v body: ", v2v_mean_body)
    print("Std v2v body: ", v2v_std_body)

    stats = {}
    stats['v2v_mean_hand'] = v2v_mean_hand
    stats['v2v_std_hand'] = v2v_std_hand
    stats['v2v_mean_feet'] = v2v_mean_feet
    stats['v2v_std_feet'] = v2v_std_feet
    stats['v2v_mean_body'] = v2v_mean_body
    stats['v2v_std_body'] = v2v_std_body

    with open('root_paths.txt','w+') as f:
        for path in root_paths:
            f.write(path+'\n')
    with open('v2v_hands.txt','w+') as f:
        for v in v2v_hand:
            f.write(str(v)+'\n')
    with open('v2v_feet.txt','w+') as f:
        for v in v2v_feet:
            f.write(str(v)+'\n')
    with open('v2v_body.txt','w+') as f:
        for v in v2v_body:
            f.write(str(v)+'\n')
    
    if plot:
        plt.hist(v2v_hand, bins=20)
        plt.title("v2v hand")
        plt.show()
        plt.hist(v2v_feet, bins=20)
        plt.title("v2v feet")
        plt.show()
        plt.hist(v2v_body, bins=20)
        plt.title("v2v body")
        plt.show()
    return stats


def get_all_root_paths(path):
    root_paths = []
    for dir_name in os.listdir(path):
        root_paths.append(path + dir_name[:-14])
    return list(set(root_paths))


if __name__ == "__main__":
    results_path = "./_RESULTS/Downloaded/objects_meshes/"
    repo_path = "./"
    stats = get_statistics(results_path)
    for key, value in stats.items():
        print(key, " : ",value)

    root_paths = get_all_root_paths(results_path)
    hand_ids, feet_ids = get_verts_ids(repo_path)

    stats = get_v2v_stats(root_paths, hand_ids, feet_ids, False)
    for key, value in stats.items():
        print(key, " : ",value)
