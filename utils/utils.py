import scipy


def read_keypoints_descriptors(path):
    data = scipy.io.loadmat(path)
    p = data['p']  # (2,N) numpy array, where N is the total number of keypoints
    d = data['d']  # (128,N) numpy array

    return d, p
