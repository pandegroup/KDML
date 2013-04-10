#!/usr/bin/env python

import os, sys
import cPickle as pickle
from argparse import ArgumentParser
import numpy as np

from msmbuilder.arglib import add_argument, die_if_path_exists
from msmbuilder import Project, io, metrics
from kdml import triplets, lmmdm
from kdml.wrmsd import WRMSD
import toy_systems

def main():
    parser = ArgumentParser(os.path.split(__file__)[1], description='''
    Kinetically discriminitory metric learning. This is a method to build
    a kinetically informed distance metric for clustering molecular dynamics
    trajectories.''')
    sparser_group = parser.add_subparsers(dest='subparser')

    #########################    #########################    ##################
    ##    EXTRACT SUBCOMMAND
    #########################    #########################    ##################
    extract_subparser = sparser_group.add_parser('extract',
        description='''Extract training data for KDML metric learning
        from trajectories. This involves pulling triplets of structures from
        an MSMBuilder project''')
    extract_group = extract_subparser.add_subparsers(dest='extract_method')
    rmsd = extract_group.add_parser('rmsd', description='''Use RMSD. For each triplet,
        we find the optimal rotation/translation to align them, and then report the distance
        between two structures as the distance between atom[i] in frame A and atom[i] in frame
        B for alll atoms i.''')
    add_argument(rmsd, 'atomindices', default='AtomIndices.dat',
        help='Which atoms to use for RMSD.')

    dihedral = extract_group.add_parser('dihedral', description='''Use Dihedral. Here,
        the dihedrals for each frame are extracted, and then the difference between two
        frames is just the difference in the dihedrals''')
    mutexgroup = dihedral.add_mutually_exclusive_group(required=True)
    mutexgroup.add_argument('-t', '--types', help='''Which dihedral angles to use. Can be either phi, psi, chi
        or any combination (written as phi/psi with the / as a deliminter)''')
    mutexgroup.add_argument('-i', '--indices', help='''For more customizability, you may pass
        a path to a file on disk containing an N x 4 array of atom indices where each row
        gives the indices of 4 atoms to be considered a dihedral angle''')

    drmsd = extract_group.add_parser('drmsd', description='''Use dRMSD pairwise distances.
        Each frame is represented by a set of atom-atom pairwise distances, and then the
        distance between two frames is the difference in these pairwise distances''')
    add_argument(drmsd, 'indices', default='AtomIndices.dat', help='''Which atoms to take pairwise
        distances between? Pass the path to a file on disk. If the file is a 1D array
        like AtomIndices.dat, then we take all the N choose 2 pairwise distances between those
        atoms to represent the frames by. Otherwise, you can pass the path to a N x 2
        2D array where each row gives the indices of 2 atoms whos distance should be considered''')

    recipcontact = extract_group.add_parser('recipcontact', description='''reciprocal
        contact map (CA)''')


    for p in [rmsd, dihedral, drmsd, recipcontact]:
        add_argument(p, 'far', help=('Number of timestep to consider "far", if '
            '<0, the "far" point will be a conf selected randomly from the dataset'),
            default=-1, type=int)
        add_argument(p, 'close', help='Number of timesteps to consider "close"')
        add_argument(p, 'stride', help='frequency to select triplets from the dataset')
        add_argument(p, 'project_info', help='MSMBuilder Project File', default='ProjectInfo.h5')
        add_argument(p, 'output', help='Location to save output to', default='triplets.h5')


    #########################    #########################    ##################
    ##      LEARN SUBCOMMAND
    #########################    #########################    ##################
    learn_subparser = sparser_group.add_parser('learn', description='learn a distance metric')
    learn_group = learn_subparser.add_subparsers(dest='learn_method')
    diagonal = learn_group.add_parser('diagonal', description='''Learn a diagonal (weighted euclidean)
        distance metric. This is faster and easier, but less flexibile''')
    dense = learn_group.add_parser('dense', description='''Learn a full rank Mahalaobis distace metric''')

    for p in [diagonal, dense]:
        g = p.add_argument_group('required arguments')
        add_argument(g, 'triplets', help='Path to triplets file (produced by extract subcommand)')
        add_argument(p, 'alpha', help='weight on margin in objective function', default=0.5)
        add_argument(p, 'matrix', help='output file for metric matrix (can be loaded with numpy.load())', default='metric.npy')
        add_argument(p, 'metric', help='output pickled metric object for MSMBuilder', default='metric.pickl')



    add_argument(dense, 'initialize', help='starting metric to use as seed',
        default='diagonal')
    add_argument(dense, 'epsilon', help='convergence threshold', default=1e-3)
    add_argument(dense, 'outer_iterations', default=100)
    add_argument(dense, 'inner_iterations', default=10)

    #########################    #########################    ##################
    ##     Parse and dispatch
    #########################    #########################    ##################
    args = parser.parse_args()
    if args.subparser == 'extract':
        main_extract(args)
    elif args.subparser == 'learn':
        main_learn(args)
    else:
        raise RuntimError('not supposed to happen')

def main_extract(args):
    "main method for the extract subcommand"
    project = Project.load_from(args.project_info)
    close = int(args.close)
    stride = int(args.stride)
    if args.far < 0:
        far = None
    else:
        far = args.far

    die_if_path_exists(args.output)

    if args.extract_method == 'rmsd':
        atomindices = np.loadtxt(args.atomindices, dtype=int)
        AtoB, AtoC = triplets.extract_rmsd(project, close, stride, atomindices, far)

    elif args.extract_method == 'dihedral':
        if 'types' in args:
            AtoB, AtoC = triplets.extract_dihedral(project, close, stride, types=args.types, far=far)
        else:
            indices = np.loadtxt(args.indices, dtype=int)
            AtoB, AtoC = triplets.extract_dihedral(project, close, stride, indices=indices, far=far)

    elif args.extract_method == 'recipcontact':
        AtoB, AtoC = triplets.extract_recipcontact(project, close, stride, far=far)

    elif args.extract_method == 'drmsd':
        indices = np.loadtxt(args.indices, dtype=int)
        AtoB, AtoC, atom_pairs = triplets.extract_drmsd(project, close, stride, indices=indices, far=far)
        io.saveh(args.output, atom_pairs=atom_pairs)
    else:
        raise NotImplementedError("Sorry, we don't have that metric")

    #Serializer({'AtoB': AtoB, 'AtoC': AtoC, 'metric': args.extract_method}).SaveToHDF(args.output)
    io.saveh(args.output, AtoB=AtoB, AtoC=AtoC, metric=np.array(list(args.extract_method)))
    print 'Saved triplets to {}'.format(args.output)


def main_learn(args):
    "main method for the learn subcommand"
    s = io.loadh(args.triplets)
    metric_string = ''.join(s['metric'])

    if args.learn_method == 'diagonal':

        # type conversion
        alpha = float(args.alpha)

        rho, weights = lmmdm.optimize_diagonal(s['AtoB'], s['AtoC'], alpha, loss='huber')

        if metric_string == 'dihedral':
            metric = metrics.Dihedral(metric='seuclidean', V=weights)
        elif metric_string == 'drmsd':
            metric = metrics.AtomPairs(metric='seuclidean', V=weights, atom_pairs=s['atom_pairs'])
        elif metric_string == 'rmsd':
            metric = WRMSD(metric='seuclidean', V=weights)
        elif metric_string == 'recipcontact':
            metric = metrics.ContinuousContact(contacts='all', scheme='CA',
                metric='seuclidean', V=weights)
        else:
            raise NotImplementedError('Sorry')


        # save to disk
        pickle.dump(metric, open(args.metric, 'w'))
        print 'Saved metric pickle to {}'.format(args.metric)
        np.save(args.matrix, [weights, rho])
        print 'Saved weights as flat text to {}'.format(args.matrix)

    elif args.learn_method == 'dense':
        initialize = args.initialize
        if not args.initialize in ['euclidean', 'diagonal']:
            try:
                initialize = np.load(initialize)
            except IOError as e:
                print >> sys.stderr, '''-i --initialize must be either "euclidean",
                    "diagonal", or the path to a flat text matrix'''
                print >> sys.stderr, e
                sys.exit(1)

        # type conversion
        alpha, epsilon = map(float, [args.alpha, args.epsilon])
        outer_iterations, inner_iterations = map(int,
            [args.outer_iterations, args.inner_iterations])

        rho, weights = lmmdm.optimize_diagonal(s['AtoB'], s['AtoC'], alpha, loss='huber')
        rho, metric_matrix = lmmdm.optimize_dense(s['AtoB'], s['AtoC'], alpha, rho, np.diag(weights),
            loss='huber', epsilon=1e-5, max_outer_iterations=outer_iterations,
            max_inner_iterations=inner_iterations)

        if metric_string == 'dihedral':
            metric = metrics.Dihedral(metric='mahalanobis', VI=metric_matrix)
        elif metric_string == 'drmsd':
            metric = metrics.AtomPairs(metric='mahalanobis', VI=metric_matrix, atom_pairs=s['atom_pairs'])
        elif metric_string == 'rmsd':
            metric = WRMSD(metric='mahalanobis', VI=metric_matrix)
        elif metric_string == 'recipcontact':
            metric = metrics.ContinuousContact(contacts='all', scheme='CA', metric='mahalanobis',
                VI=metrix_matrix)
        else:
            raise NotImplementedError('Sorry')

        # save to disk
        pickle.dump(metric, open(args.metric, 'w'))
        print 'Saved metric pickle to {}'.format(args.metric)
        np.save(args.matrix, [metric_matrix, rho])
        print 'Saved weights, rho to {}'.format(args.matrix)


if __name__ == '__main__':
    main()
