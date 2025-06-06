__author__ = 'Brian Iwana'

# https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247491743&idx=1&sn=2bceb18eded5a72d1b773748c437d84a&chksm=96c7cd5aa1b0444c28b53bb64ab79087a7b35ab4182a375b31cb8f0c82bddaa51b6709f41478&scene=178&cur_album_id=1364202321906941952&search_click_id=#rd

import math
import sys

import numpy as np

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1


# Core DTW
def _traceback(DTW, slope_constraint):
    i, j = np.array(DTW.shape) - 1
    p, q = [i-1], [j-1]
    
    if slope_constraint == "asymmetric":
        while (i > 1):
            tb = np.argmin((DTW[i-1, j], DTW[i-1, j-1], DTW[i-1, j-2]))

            if (tb == 0):
                i = i - 1
            elif (tb == 1):
                i = i - 1
                j = j - 1
            elif (tb == 2):
                i = i - 1
                j = j - 2

            p.insert(0, i-1)
            q.insert(0, j-1)
    elif slope_constraint == "symmetric":
        while (i > 1 or j > 1):
            tb = np.argmin((DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1]))

            if (tb == 0):
                i = i - 1
                j = j - 1
            elif (tb == 1):
                i = i - 1
            elif (tb == 2):
                j = j - 1

            p.insert(0, i-1)
            q.insert(0, j-1)
    else:
        sys.exit("Unknown slope constraint %s"%slope_constraint)
        
    return (np.array(p), np.array(q))


def _cummulative_matrix(cost, slope_constraint, window):
    p = cost.shape[0]
    s = cost.shape[1]
    
    # Note: DTW is one larger than cost and the original patterns
    DTW = np.full((p+1, s+1), np.inf)

    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p+1):
            if i <= window+1:
                DTW[i,1] = cost[i-1,0] + min(DTW[i-1,0], DTW[i-1,1])
            for j in range(max(2, i-window), min(s, i+window)+1):
                DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j-2], DTW[i-1,j-1], DTW[i-1,j])
    elif slope_constraint == "symmetric":
        for i in range(1, p+1):
            for j in range(max(1, i-window), min(s, i+window)+1):
                DTW[i,j] = cost[i-1,j-1] + min(DTW[i-1,j-1], DTW[i,j-1], DTW[i-1,j])
    else:
        sys.exit("Unknown slope constraint %s"%slope_constraint)
        
    return DTW


def dtw(prototype, sample, return_flag = RETURN_VALUE, slope_constraint="asymmetric", window=None):
    """ Computes the DTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"
    
    if window is None:
        window = s
    
    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i-window)
        end = min(s, i+window)+1
        cost[i,start:end]=np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cummulative_matrix(cost, slope_constraint, window)
        
    if return_flag == RETURN_ALL:
        return DTW[-1,-1], cost, DTW[1:,1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1,-1]


def shape_dtw(prototype, sample, return_flag = RETURN_VALUE, slope_constraint="asymmetric", window=None, descr_ratio=0.05):
    """ Computes the shapeDTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    # shapeDTW
    # https://www.sciencedirect.com/science/article/pii/S0031320317303710
    
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"
    
    if window is None:
        window = s
        
    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)
    
    # padding
    p_pad_front = (np.ceil(p_feature_len / 2.)).astype(int)
    p_pad_back = (np.floor(p_feature_len / 2.)).astype(int)
    s_pad_front = (np.ceil(s_feature_len / 2.)).astype(int)
    s_pad_back = (np.floor(s_feature_len / 2.)).astype(int)
    
    prototype_pad = np.pad(prototype, ((p_pad_front, p_pad_back), (0, 0)), mode="edge") 
    sample_pad = np.pad(sample, ((s_pad_front, s_pad_back), (0, 0)), mode="edge") 
    p_p = prototype_pad.shape[0]
    s_p = sample_pad.shape[0]
        
    cost = np.full((p, s), np.inf)
    for i in range(p):
        for j in range(max(0, i-window), min(s, i+window)):
            cost[i, j] = np.linalg.norm(sample_pad[j:j+s_feature_len] - prototype_pad[i:i+p_feature_len])
            
    DTW = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)
    
    if return_flag == RETURN_ALL:
        return DTW[-1,-1], cost, DTW[1:,1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1,-1]
    
# Draw helpers
def draw_graph2d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    #cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[0]-0.5))

    #dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0]+1, path[1]+1, 'y')
    plt.xlim((-0.5, DTW.shape[0]-0.5))
    plt.ylim((-0.5, DTW.shape[0]-0.5))

    #prototype
    plt.subplot(2, 3, 4)
    plt.plot(prototype[:,0], prototype[:,1], 'b-o')

    #connection
    plt.subplot(2, 3, 5)
    for i in range(0,path[0].shape[0]):
        plt.plot([prototype[path[0][i],0], sample[path[1][i],0]],[prototype[path[0][i],1], sample[path[1][i],1]], 'y-')
    plt.plot(sample[:,0], sample[:,1], 'g-o')
    plt.plot(prototype[:,0], prototype[:,1], 'b-o')

    #sample
    plt.subplot(2, 3, 6)
    plt.plot(sample[:,0], sample[:,1], 'g-o')

    plt.tight_layout()
    plt.show()


def draw_graph1d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
   # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    p_steps = np.arange(prototype.shape[0])
    s_steps = np.arange(sample.shape[0])

    #cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0]-0.5))
    plt.ylim((-0.5, cost.shape[0]-0.5))

    #dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0]+1, path[1]+1, 'y')
    plt.xlim((-0.5, DTW.shape[0]-0.5))
    plt.ylim((-0.5, DTW.shape[0]-0.5))

    #prototype
    plt.subplot(2, 3, 4)
    plt.plot(p_steps, prototype[:,0], 'b-o')

    #connection
    plt.subplot(2, 3, 5)
    for i in range(0,path[0].shape[0]):
        plt.plot([path[0][i], path[1][i]],[prototype[path[0][i],0], sample[path[1][i],0]], 'y-')
    plt.plot(p_steps, sample[:,0], 'g-o')
    plt.plot(s_steps, prototype[:,0], 'b-o')

    #sample
    plt.subplot(2, 3, 6)
    plt.plot(s_steps, sample[:,0], 'g-o')

    plt.tight_layout()
    plt.show()