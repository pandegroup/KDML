from msmbuilder import metrics
import numpy as np
from roberttools import _WRMSD

class WRMSD(metrics.RMSD):
    """
    Weighted RMSD distance metric
    
    To compute the distance between two frames, the frames are first aligned
    (rotation and translation) to minimize the standard RMSD, and the collection
    of atomic displacements is calculated (i.e. distance from atom[i] in frame A
    to atom[i] in frame B). Then, using these displacements, a weighted RMSD is
    calculated. For metric='seuclidean', we sum these together (in quadrature)
    with a set of weights -- if the weights are all equal you recover the standard
    RMSD. [to make them numerically equal the weights should be equal AND sum to 1]
    
    If metric='mahalanobis', the displacements, we instead calculate the
    WRMSD distance from the deviations as sqrt(deviation.T * VI * deviation)
    -- an inner product with the mahalanobis matrix VI.
    
    Note that the alignment is done UNWEIGHTED.
    """
    def __init__(self, atomindices=None, metric='euclidean', V=None, VI=None, omp_parallel=True):
        super(WRMSD, self).__init__(atomindices, omp_parallel=omp_parallel)
        
        if not metric in ['euclidean', 'seuclidean', 'mahalanobis']:
            raise ValueError()
        if metric == 'seuclidean' and V is None:
            raise ValueError('V with seuclidean')
        if metric == 'mahalanobis' and VI is None:
            raise ValueError('VI with mahalanobis')
        # check that V is 1D
        # check that VI is square
            
        self.metric = metric
        self.V = np.array(V)
        self.VI = np.array(VI)
    
    def __repr__(self):
        return "<WRMSD(atom_indices={}, metric={})>".format(str(self.atomindices), self.metric)

    def _collect_deviations(self, deviations):
        if self.metric == 'euclidean':
            sdeviations = np.square(deviations)
            return np.sqrt(np.sum(sdeviations, axis=1))
            
        elif self.metric == 'seuclidean':
            sdeviations = np.square(deviations)
            return np.sqrt(np.tensordot(sdeviations, self.V, axes=(1,0)))
            
        elif self.metric == 'mahalanobis':
            deviations = np.sqrt(np.sum(deviations * np.dot(deviations, self.VI), axis=1))
            
            # fastest check for existance of nans
            # http://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
            if np.isnan(np.sum(deviations)):
                raise ValueError('VI is not positive definite')
            return deviations
        
        else:
            raise RuntimeError('Should not happen')
        
    def one_to_all(self, ptraj1, ptraj2, index1):
        """Compute the distance from a single frame of one trajectory to each
        frame of another
        
        Parameters
        ----------
        ptraj1 : prepared_trajectory
        ptraj2 : prepared_trajectory
        index1 : int
        """
        
        deviations = self.one_to_all_deviations(ptraj1, ptraj2, index1)
        return self._collect_deviations(deviations)

    def one_to_many(self, ptraj1, ptraj2, index1, indices2):
        """Compute the distance from a single frame of one trajectory to a set
        of frames in another
        
        Parameters
        ----------
        ptraj1 : prepared_trajectory
        ptraj2 : prepared_trajectory
        index1 : int
        indices2 : array_like of ints
        """
        
        deviations = self.one_to_many_deviations(ptraj1, ptraj2, index1, indices2)
        return self._collect_deviations(deviations)
        
    def one_to_all_deviations(self, ptraj1, ptraj2, index1):
        """Compute the deviations -- distances between each atom in frame1 to
        its counterpart in frame2 -- for all frame2 in the ptraj2
        
        Parameters
        ----------
        ptraj1 : prepared_trajectory
        ptraj2 : prepared_trajectory
        index1 : int
        
        Returns
        -------
        deviations : np.ndarray
            2D array of shape `len(ptraj2)` x n_atoms giving the displacement
            of each atom from ptraj1[index1] to each its counterpart in each frame
            of index2
        """
        
        if not ptraj1.NumAtoms == ptraj2.NumAtoms:
            raise ValueError()
        if not ptraj1.NumAtomsWithPadding == ptraj2.NumAtomsWithPadding:
            raise ValueError()

        deviations = np.zeros((len(ptraj2), ptraj1.NumAtoms), dtype=np.float32)
        _WRMSD.one_to_all(ptraj1.NumAtoms, ptraj1.NumAtomsWithPadding,
                          ptraj1.NumAtomsWithPadding, self.omp_parallel,
                          ptraj1.G[index1], ptraj1.XYZData[index1],
                          ptraj2.XYZData, ptraj2.G, deviations)
                 
        return deviations
    
    def one_to_many_deviations(self, ptraj1, ptraj2, index1, indices2):
        """Compute the deviations -- distances between each atom in frame1 to
        its counterpart in frame2 -- for a set of frames in ptraj2
        
        Parameters
        ----------
        ptraj1 : prepared_trajectory
        ptraj2 : prepared_trajectory
        index1 : int
        indices2 : array_like of ints
        
        Returns
        -------
        deviations : np.ndarray
            2D array of shape `len(indices2)` x n_atoms giving the displacement
            of each atom from ptraj1[index1] to each its counterpart in the frames
            ptraj2[indices2]
        """
        
        if not ptraj1.NumAtoms == ptraj2.NumAtoms:
            raise ValueError()
        if not ptraj1.NumAtomsWithPadding == ptraj2.NumAtomsWithPadding:
            raise ValueError()
        if not index1 == np.int(index1):
            raise TypeError()
        indices2 = np.array(indices2, dtype=np.int32)
        if not indices2.flags.contiguous:
            raise ValueError()
        if not (np.max(indices2) < len(ptraj2) and np.min(indices2) >= 0):
            raise ValueError()
        
        deviations = np.zeros((len(indices2), ptraj1.NumAtoms), dtype=np.float32)
        
        _WRMSD.one_to_many(ptraj1.NumAtoms, ptraj1.NumAtomsWithPadding,
                              ptraj1.NumAtomsWithPadding, self.omp_parallel,
                              ptraj1.G[index1], ptraj1.XYZData[index1],
                              ptraj2.XYZData, ptraj2.G, indices2, deviations)
        
        return deviations
        
    def corresponding_deviations(self, ptraj1, ptraj2):
        """Compute the deviations between corresponding frames in two trajectories
        
        ptraj1 and ptraj2 must be the same length
        
        Parameters
        ----------
        ptraj1 : prepared_trajectory
        ptraj2 : prepared_trajectory
        
        Returns
        -------
        deviations : np.ndarray
            2D array of shape `len(ptraj1)` x `n_atoms` giving the displacement
            of each atom from `ptraj1[i]` to `ptraj2[i]` for each i up to the
            lengthof the trajectories
        """
        
        if not ptraj1.NumAtoms == ptraj2.NumAtoms:
            raise ValueError()
        if not ptraj1.NumAtomsWithPadding == ptraj2.NumAtomsWithPadding:
            raise ValueError()
        if not len(ptraj1) == len(ptraj2):
            raise ValueError
            
        deviations = np.zeros((len(ptraj1), ptraj1.NumAtoms), dtype=np.float32)        
        _WRMSD.corresponding(
                ptraj1.NumAtoms, ptraj1.NumAtomsWithPadding,
                ptraj1.NumAtomsWithPadding, self.omp_parallel, ptraj1.XYZData,
                ptraj2.XYZData, ptraj1.G, ptraj2.G, deviations)
        
        return deviations
