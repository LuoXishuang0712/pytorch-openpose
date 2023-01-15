
from dynamic_draw import dynamic_draw
from src.body import Body

import cv2
import numpy as np
import argparse
import tqdm

import time

draw = False

body_estimation = Body('./model/body_pose_model.pth', device="cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--video_path")
parser.add_argument("--data_path")
# parser.add_argument("--type")
args = parser.parse_args()

video_path = args.video_path
output_file = args.data_path
# output_video = output_file[:-4] + ".mp4" if output_file.endswith(".npy") else output_file + ".mp4"
output_file = output_file if output_file.endswith(
    ".npy") else output_file + ".npy"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
data = [[], [], []]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if draw:
    drawer = dynamic_draw(height, width)

def data_standard(array, max):
    # map [0, max] to [-1, 1]
    return (array - (max / 2)) / (max / 2)


def ske_map(mapping, raw_data):
    """ ## args:  
            mapping: list, looks like [(target, raw),], target is always int and raw can be int(directly mapping) or string(calculate)
            raw_data: np.array, the raw data to be maped
        ## about raw at str
            % and ,: specify the ratio about raw skeletons in the target, usage: "1%10,2%30,3%40,4%20"
            &: simply mean the data, usage: "1&2" (as same as "1%50,2%50")
    """

    def ske_calc(division, skeletons):
        assert len(division) == len(skeletons)
        ans = np.zeros_like(skeletons[0])
        for i in range(len(division)):
            ans[0] += (skeletons[i][0]*division[i])/100
            ans[1] += (skeletons[i][1]*division[i])/100
        ans[2] = -1
        return ans

    ans = np.zeros((1, 25, 3))
    for item in mapping:
        assert len(item) == 2
        assert isinstance(item[0], int)

        if isinstance(item[1], str):
            if '%' in item[1]:
                division = []
                skeletons = []
                for item_ in item[1].split(","):
                    division.append(int(item_.split('%')[1]))
                    skeletons.append(raw_data[0][int(item_.split('%')[0])])
                assert sum(division) == 100
                ans[0][item[0]] = ske_calc(division, skeletons)
            elif '&' in item[1]:
                ans[0][item[0]] = ske_calc([50, 50], [
                    raw_data[0][int(item[1].split("&")[0])],
                    raw_data[0][int(item[1].split("&")[1])]
                ])
            else:
                raise ValueError("map format error")
        elif isinstance(item[1], int):
            ans[0][item[0]] = raw_data[0][item[1]]
        else:
            raise ValueError("Raw type should be int or str")
    return ans


def trans_starndard_openpose(candidate, subset):
    output = np.zeros((len(subset), 18, 3))
    skeleton_points = 18
    ske_maps = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                (8, "8&11"), (9, 8), (10, 9), (11, 10), (12, 11), (13, 12),
                (14, 13), (15, 14), (16, 15), (17, 16), (18, 17)]
    for i in range(skeleton_points):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            output[n][i][0], output[n][i][1], output[n][i][2] = \
                candidate[index][0:3]
    return ske_map(ske_maps, output)

if __name__ == "__main__":
    start = time.time()
    for _ in tqdm.tqdm(range(frame_count)):
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            break

        candidate, subset = body_estimation(frame)

        if candidate is None or subset is None:
            continue

        sdata = trans_starndard_openpose(candidate, subset)

        if draw:
            drawer.motion_update(sdata)

        frame_shape = frame.shape
        if len(sdata) > 0:
            sdata = sdata[0].T
            data[0].append(data_standard(
                sdata[0], frame_shape[1]).reshape((25, 1)))
            data[1].append(data_standard(
                sdata[1], frame_shape[0]).reshape((25, 1)))
            data[2].append(sdata[2].reshape((25, 1)))
        

    for _ in range(350 - len(data[0])):
        data[0].append(np.zeros((25, 1)))
        data[1].append(np.zeros((25, 1)))
        data[2].append(np.zeros((25, 1)))

    final_data = np.array([
        np.array(data[0]),
        np.array(data[1]),
        np.array(data[2])
    ], dtype='float32')
    final_data.resize((3, 350, 25, 1))

    np.save(output_file, final_data)

    stop = time.time()
    print("delta: %d" % (stop - start))
