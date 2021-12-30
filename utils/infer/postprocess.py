import numpy as np
from scipy.ndimage import maximum_filter


def local_maxima(img: np.ndarray, threshold: float = 0.5):
    '''detect local maxima of predicted heatmap'''
    dim = img.ndim
    footprint = np.ones([3]*dim)
    footprint[tuple([1]*dim)] = 0
    maxima_filtered = np.clip(maximum_filter(
        img, footprint=footprint, mode='constant', cval=np.inf), 0, 1-1e-6)
    maxima_indices = np.array(np.where(img > maxima_filtered))
    maxima_values = img[tuple([maxima_indices[i] for i in range(dim)])]
    maxima_indices = maxima_indices.T

    values_greater = maxima_values > threshold
    coords = maxima_indices[values_greater]
    return coords


def process_coords(coords: np.ndarray, threshold: float = 12.5):
    '''local area could exist several local maxima,so the coordinations need to be processed.
    coords: [N,3],coordinations of local maxima
    threshold: distance between 2 vertebrae should be greater than 12.5mm.
    '''
    final_coords = []
    tmp = []  # record local maximas
    def dist(x, y): return np.sqrt(np.power(x-y, 2).sum())
    prev = coords[0]
    tmp.append(prev.reshape(1, 3))
    for i in range(1, len(coords)):
        if dist(coords[i], prev) * 2 < 12.5:
            tmp.append(coords[i])
            prev = coords[i]
            continue
        pt = np.round(np.mean(np.vstack(tmp), 0)).astype(int)
        final_coords.append(pt)
        prev = coords[i]
        tmp = []
        tmp.append(prev.reshape(1, 3))
    last_pt = np.round(np.mean(np.vstack(tmp), 0)).astype(int)
    final_coords.append(last_pt)
    return np.array(final_coords)


def one_class_dist(landmark, coords, threshold=12.5):
    '''calculate distance between predicted coordination and groundtruth landmark'''
    i = 0
    j = 0
    dist_total = 0.
    cnt = 0
    def dist(x, y): return np.sqrt(np.power(x-y, 2).sum())
    while(i < len(landmark) and j < len(coords)):
        gt = landmark[i]
        pt = coords[j]
        if dist(gt, pt)*2 < threshold:
            dist_total += dist(gt, pt)*2
            cnt += 1
            i += 1
            j += 1
            continue
        if coords[j, 0] < landmark[i, 0]:
            j += 1
        else:
            i += 1
    return dist_total, cnt


def multi_channel_coords(data: np.ndarray, threshold: float = 0.1):
    '''Localize the coordination of multi-channel heatmap
    data:[C,Z,Y,X]:
    threshold: threshold to decide if this class exists or not.
    '''
    indices = []
    num = data.shape[0]
    for i in range(num):
        tmp = np.array(np.where(data[i] == data[i].max())).T
        if len(tmp) > 1:
            tmp = np.mean(tmp, 0).round().astype(int)
        else:
            tmp = tmp.reshape(-1)
        indices.append(tmp)
    # indices of the biggest value in every channel.[C,3]
    indices = np.array(indices)
    maxima_values = [
        data[j][tuple([indices[j, i] for i in range(3)])] for j in range(num)]
    maxima_values = np.array(maxima_values)  # values of those voxel
    # classes of detected vertebrae
    classes = np.where(maxima_values > threshold)
    return indices, classes


def multi_class_dist(landmark, indices, classes):
    '''calculate distance between predicted coordination and groundtruth landmark'''
    dist_total = 0.
    cnt = 0
    def dist(x, y): return np.sqrt(np.power(x-y, 2).sum())
    for i in range(landmark.shape[0]):
        c = landmark[i, 0]  # class of groundtruth landmark
        if c in classes:
            dist_total += dist(landmark[i, 1:], indices[c])
            cnt += 1
    return dist_total, cnt
