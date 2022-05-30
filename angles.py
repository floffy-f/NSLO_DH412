import numpy as np
from plot_pose import ARTIC_NAMES

def make_vec(keypoints, pt1, pt2):
    """
    returns vector pt1->pt2
    """
    slicer = np.zeros(13)
    slicer[ARTIC_NAMES[pt1]] = -1
    slicer[ARTIC_NAMES[pt2]] = 1
    # i: articulation
    # j: apsara n°
    # k: 2D pt coordinate
    return np.einsum("i,jik->jk", slicer, keypoints)

def angle(vec1, vec2):
    """
    Compute angle between vectors 1 and 2
    vec1 and vec2 are lists of vectors (in np format)
    """
    prod = np.einsum("ij,ij->i", vec1, vec2)
    nrm = np.linalg.norm(vec1, axis=1)\
        * np.linalg.norm(vec2, axis=1)
    return np.arccos(prod / nrm)

def get_angles(keypoints):
    angle_knee1 = angle(make_vec(keypoints, "knee 1", "foot 1"),
                        make_vec(keypoints, "knee 1", "belly"))
    angle_knee2 = angle(make_vec(keypoints, "knee 2", "foot 2"),
                        make_vec(keypoints, "knee 2", "belly"))
    angle_elbow1 = angle(make_vec(keypoints, "elbow 1", "hand 1"),
                         make_vec(keypoints, "elbow 1", "shoulder 1"))
    angle_elbow2 = angle(make_vec(keypoints, "elbow 2", "hand 2"),
                         make_vec(keypoints, "elbow 2", "shoulder 2"))
    angle_shoulder1 = angle(make_vec(keypoints, "shoulder 1", "elbow 1"),
                            make_vec(keypoints, "shoulder 1", "belly"))
    angle_shoulder2 = angle(make_vec(keypoints, "shoulder 2", "elbow 2"),
                            make_vec(keypoints, "shoulder 2", "belly"))
    return np.stack([angle_knee1,
                     angle_knee2,
                     angle_elbow1,
                     angle_elbow2,
                     angle_shoulder1,
                     angle_shoulder2
                    ], axis=1)