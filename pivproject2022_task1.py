import sys
from utils.utils import split_video_to_frames, get_keypoints_ORB


def project():
    split_video_to_frames(template_dir)


def test():
    for i in range(1, 1000):
        img_nr = str(i)
        n_zeros = (4 - len(img_nr)) * '0'
        img_path = "frames/rgb_" + n_zeros + img_nr + ".jpg"
        k, s = get_keypoints_ORB(img_path)
        print(img_path)
        print(k)
        print(s)



if __name__ == "__main__":
    args = sys.argv
    template_dir = args[1]
    input_dir = args[2]
    output_dir = args[3]

    test()




