from tools.utils import d62rotmat

def full2bone(pose,trans):
    global_orient = pose[:, 0:1]
    body_pose = pose[:, 1:22]
    jaw_pose  = pose[:, 22:23]
    leye_pose = pose[:, 23:24]
    reye_pose = pose[:, 24:25]
    left_hand_pose = pose[:, 25:40]
    right_hand_pose = pose[:, 40:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms


def parms_6D2full(pose,trans, d62rot=True):

    bs = trans.shape[0]

    if d62rot:
        pose = d62rotmat(pose)
    pose = pose.reshape([bs, -1, 3, 3])

    body_parms = full2bone(pose,trans)
    body_parms['fullpose_rotmat'] = pose

    return body_parms
