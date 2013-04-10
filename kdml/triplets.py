import numpy as np
import random
import itertools, collections
from msmbuilder.utils import lru_cache
from msmbuilder.clustering import concatenate_trajectories
from msmbuilder.geometry import dihedral as dihedralcalc
from msmbuilder import metrics
from kdml.wrmsd import WRMSD
import logging
logger = logging.getLogger()


def extract_recipcontact(project, close, stride, far=None):
    A,B,C = triplets(project, close, stride, far)
    m = metrics.ContinuousContact(contacts='all', scheme='CA')
    pA, pB, pC = map(m.prepare_trajectory, [A, B, C])

    # reciprocate the maps
    pA, pB, pC = np.reciprocal(pA), np.reciprocal(pB), np.reciprocal(pC)

    return pA - pB, pA - pC


def extract_rmsd(project, close, stride, atomindices, far=None):
    A,B,C = triplets(project, close, stride, far)

    w = WRMSD(atomindices=atomindices)
    pA, pB, pC = map(w.prepare_trajectory, [A,B,C])

    AtoB = w.corresponding_deviations(pA, pB)
    AtoC = w.corresponding_deviations(pA, pC)

    return AtoB, AtoC


def extract_dihedral(project, close, stride, types=None, indices=None, far=None):

    if indices is None:
        indices = dihedralcalc.get_indices(project.empty_traj(), types)

    # get trajectories for A, B, C
    A,B,C = triplets(project, close, stride, far)

    metric = metrics.Dihedral(indices=indices)
    A = metric.prepare_trajectory(A)
    B = metric.prepare_trajectory(B)
    C = metric.prepare_trajectory(C)

    return A-B, A-C


def extract_drmsd(project, close, stride, indices, far=None):
    A,B,C = triplets(project, close, stride, far)
    atom_pairs =  np.array(list(itertools.combinations(indices, 2)))
    metric = metrics.AtomPairs(atom_pairs=atom_pairs)

    A = metric.prepare_trajectory(A)
    B = metric.prepare_trajectory(B)
    C = metric.prepare_trajectory(C)

    return A-B, A-C, atom_pairs


def extract_2d(project, close, stride, far):
    A, B, C = triplets(project, close, stride, far)

    return A['XYZList'] - B['XYZList'], A['XYZList'] - C['XYZList']


def triplets(project, close, stride, far=None):
    """Get three trajectories which represent triplets of close/far pairs

    Trajectories A, B, C constructued such that frame A[i] is close to frame
    B[i] and far from frame C[i].

    Paramters
    ---------
    project : msmbuilder.Project
        MSMBuilder Project object
    close : int
        number of frames between A[i] and B[i] -- to be considered "close"
    stride : int
        Frequency to get from the dataset
    far : {int, None}
        If `far` is none, the point `C` is sampled randomly from the dataset.
        If `far` is an int, the point `C` is taken to be `far` frames after
        point A

    Returns
    -------
    A : msmbuilder.Trajectory
    B : msmbuilder.Trajectory
    C : msmbuilder.Trajectory
    """
    cumsum = np.cumsum(project.traj_lengths)
    def split(longindex):
        traj = np.where(longindex <= cumsum)[0][0]
        if traj == 0:
            frame = longindex
        else:
            frame = longindex - cumsum[traj-1]
        return traj, frame

    # load the trajs with caching of the two most recently loaded trajectories?
    @lru_cache(maxsize=2)
    def load_traj(traj_index):
        return project.load_traj(traj_index)

    a_indices = np.arange(0, cumsum[-1], stride)
    b_indices = a_indices + close
    if far is not None:
        c_indices = a_indices + far
    else:
        c_indices = np.random.permutation(len(a_indices))
        np.sort(c_indices)

    ind_to_keep = np.where(c_indices < cumsum[-1])
    c_indices = c_indices[ind_to_keep]
    b_indices = b_indices[ind_to_keep]
    a_indices = a_indices[ind_to_keep]


    a_frames, b_frames, c_frames = [], [], []
    for i, a in enumerate(a_indices):
        b, c = b_indices[i], c_indices[i]
        itraj_a, iframe_a = split(a)
        itraj_b, iframe_b = split(b)
        itraj_c, iframe_c = split(c)

        #print 't', itraj_a, itraj_b, itraj_c
        logger.debug('f %d %d %d', iframe_a, iframe_b, iframe_c)

        if a < b < c < cumsum[-1] and far is not None and itraj_a == itraj_b == itraj_c:
            traj = load_traj(split(a)[0])
            if iframe_a < len(traj) and iframe_b < len(traj) and iframe_c < len(traj):
                frame_a = traj['XYZList'][iframe_a]
                frame_b = traj['XYZList'][iframe_b]
                frame_c = traj['XYZList'][iframe_c]

                a_frames.append(frame_a)
                b_frames.append(frame_b)
                c_frames.append(frame_c)


        elif a < b < c < cumsum[-1] and far is None and itraj_a == itraj_b:
            traj_a, traj_b, traj_c = load_traj(itraj_a), load_traj(itraj_b), load_traj(itraj_c)
            if iframe_a < len(traj_a) and iframe_b < len(traj_b) and iframe_c < len(traj_c):

                a_frames.append(traj_a['XYZList'][iframe_a])
                b_frames.append(traj_b['XYZList'][iframe_b])
                c_frames.append(traj_c['XYZList'][iframe_c])

    A = project.empty_traj()
    B = project.empty_traj()
    C = project.empty_traj()
    A['XYZList'] = np.array(a_frames)
    B['XYZList'] = np.array(b_frames)
    C['XYZList'] = np.array(c_frames)

    if far is None:
        # if far is none, what we really want is to have randomly selected
        # shuffled frames
        np.random.shuffle(C['XYZList'])

    return A, B, C
