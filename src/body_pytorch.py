import cv2
import numpy as np
import math
import time
# from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

import sys

sys.path.append("./")

from src import util
from src.model import bodypose_model

def gaussian_filter(mat, sigma, device='cpu', numpy=True):  # replace scipy.ndimage.filters.gaussian_filter
    assert not (numpy and device != 'cpu'), 'when set gpu as calculation device, numpy should be False'
    raw_shape = mat.shape
    out = transforms.GaussianBlur((5, 5), sigma=(sigma, sigma))(torch.from_numpy(mat[None, :]) if not isinstance(mat, torch.Tensor) else mat[None, :])
    if numpy:
        return out.numpy().reshape(*raw_shape)
    else:
        return out[0]

def tensor_delete(mat : torch.Tensor, index) -> torch.Tensor:  # replace np.delete
    if isinstance(index, int):
        return tensor_delete_i(mat, index)
    elif isinstance(index, list):
        index.sort(reverse=True)
        for indext in index:
            mat = tensor_delete_i(mat, indext)
        return mat
    else:
        print("get wrong arg type:(%s, %s), except(torch.Tensor, [int|list])" % 
            (type(mat), type(index)))
        return mat

def tensor_delete_i(mat : torch.Tensor, index : int) -> torch.Tensor:  # for tensor_delete, delete a line on the dim0
    return torch.cat((mat[:index], mat[index+1:]), dim=0)

def resize_as_cv(mat: torch.Tensor, size, fx = None, fy = None, interpolation = transforms.InterpolationMode.BICUBIC, device = 'cpu'):  # let torchvision.transforms.Resize work as cv2.resize
    ret = None
    if fx is not None and fy is not None:
        size = list(mat.shape)
        size[0] *= fx
        size[1] *= fy
        ret = torch.zeros(size).to(device)
    else:
        ret = torch.zeros([size[0], size[1], mat.shape[2]]).to(device)
    resizer = transforms.Resize([*size[:2]], interpolation=interpolation).to(device)
    for index in range(mat.shape[2]):
        ret[:,:,index: index + 1] = \
            resizer(mat[:,:,index: index + 1].reshape((1, mat.shape[0], mat.shape[1]))) \
                .reshape((ret.shape[0], ret.shape[1], 1))
    return ret

def logical_and_reduce_axis_zero(mat: torch.Tensor):
    it = iter(mat)
    ret = next(it)
    while True:
        try:
            ret = torch.logical_and(ret, next(it))
        except StopIteration as ign:
            break
    return ret

class Body(object):
    def __init__(self, model_path, device="cpu"):
        self.model = bodypose_model()
        self.device = device
        if device != "cpu":
            self.model = self.model.to(device)
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        # if self.device != "cpu":
        #     oriImg = torch.tensor(oriImg).to(self.device)
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 19)).to(self.device)
        paf_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 38)).to(self.device)

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            # im = torch.transpose(torch.float32(imageToTest_padded[:, :, :, None]), (3, 2, 0, 1)) / 256 - 0.5
            im = torch.transpose(torch.from_numpy(imageToTest_padded[:, :, :, None]).to(torch.float32), 0, 2).transpose(0, 1).transpose(0, 3) / 256 - 0.5
            #                                                         [0, 1, 2, 3] [2, 1, 0, 3] [1, 2, 0, 3] [3, 2, 0, 1]
            # im = np.ascontiguousarray(im)  # 改变数组在内存中的空间位置，提高计算速度，这里使用pytorch在显存计算

            data = im.float()
            if self.device != "cpu":
                data = data.to(self.device)
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            # Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            # Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            # heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
            # heatmap = torch.transpose(torch.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            heatmap = torch.transpose(torch.squeeze(Mconv7_stage6_L2), 0, 1).transpose(1, 2).to(self.device)  # output 1 is heatmaps
            #                                                [0, 1, 2] [1, 0, 2] [1, 2, 0]
            # heatmap = heatmap.cpu().numpy()
            heatmap = resize_as_cv(heatmap, (0, 0), fx=stride, fy=stride, interpolation=transforms.InterpolationMode.BICUBIC, device=self.device)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = resize_as_cv(heatmap, (oriImg.shape[0], oriImg.shape[1]), interpolation=transforms.InterpolationMode.BICUBIC, device=self.device)
            # heatmap = torch.from_numpy(heatmap).to(self.device)

            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            # paf = torch.transpose(torch.squeeze(Mconv7_stage6_L1), (1, 2, 0))  # output 0 is PAFs
            paf = torch.transpose(torch.squeeze(Mconv7_stage6_L1), 0, 1).transpose(1, 2)  # output 0 is PAFs
            #                                           [0, 1, 2] [1, 0, 2] [1, 2, 0]
            # paf = paf.cpu().numpy()  # TODO redo algorithm, move to pytorch
            paf = resize_as_cv(paf, (0, 0), fx=stride, fy=stride, interpolation=transforms.InterpolationMode.BICUBIC, device=self.device)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = resize_as_cv(paf, (oriImg.shape[0], oriImg.shape[1]), interpolation=transforms.InterpolationMode.BICUBIC, device=self.device)
            # paf = torch.from_numpy(paf).to(self.device)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += + paf / len(multiplier)

        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3, device=self.device, numpy=False)  # lead to the low speed
            # one_heatmap = one_heatmap.to("cpu")

            map_left = torch.zeros(one_heatmap.shape).to(self.device)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = torch.zeros(one_heatmap.shape).to(self.device)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = torch.zeros(one_heatmap.shape).to(self.device)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = torch.zeros(one_heatmap.shape).to(self.device)
            map_down[:, :-1] = one_heatmap[:, 1:]
            
            peaks_binary = logical_and_reduce_axis_zero(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            # peaks_binary = torch.from_numpy(peaks_binary)
            try:
                peaks = list(zip(
                    torch.nonzero(peaks_binary)[0][1].reshape(1, 1), 
                    torch.nonzero(peaks_binary)[0][0].reshape(1, 1)
                    ))  # note reverse
            except IndexError:
                peaks = []
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = torch.subtract(
                            torch.Tensor([i for i in candB[j][:2]]).to(self.device), 
                            torch.Tensor([i for i in candA[i][:2]]).to(self.device)
                            )
                        norm = torch.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = torch.max(torch.Tensor([0.001]).to(self.device), norm)
                        vec = torch.divide(vec, norm)

                        startend = list(zip(torch.linspace(candA[i][0].item(), candB[j][0].item(), steps=mid_num), \
                                            torch.linspace(candA[i][1].item(), candB[j][1].item(), steps=mid_num)))

                        vec_x = torch.tensor([score_mid[int(startend[I][1].round()), int(startend[I][0].round()), 0] \
                                          for I in range(len(startend))]).to(self.device)
                        vec_y = torch.tensor([score_mid[int(startend[I][1].round()), int(startend[I][0].round()), 1] \
                                          for I in range(len(startend))]).to(self.device)

                        score_midpts = torch.multiply(vec_x, vec[0]) + torch.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(torch.nonzero(score_midpts > thre2).T[0]) > 0.8 * len(score_midpts)
                        criterion2 = (score_with_dist_prior > 0).item()
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = torch.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = torch.vstack([connection, torch.tensor([candA[i][3], candB[j][3], s, i, j])])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # can trans to GPU -

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * torch.ones((0, 20))
        candidate = torch.tensor([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = torch.tensor(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].to(torch.long), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).to(torch.int) + (subset[j2] >= 0).to(torch.int))[:-2]
                        if len(torch.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            # subset = np.delete(subset, j2, 0)
                            subset = torch.cat((subset[:j2], subset[j2+1:]), dim=0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].to(torch.long), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * torch.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].to(torch.long), 2]) + connection_all[k][i][2]
                        subset = torch.vstack([subset, row])

        # can trans to GPU -

        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        # subset = np.delete(subset, deleteIdx, axis=0)
        subset = tensor_delete(subset, deleteIdx)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate.numpy(), subset.numpy()

if __name__ == "__main__":
    body_estimation = Body('./model/body_pose_model.pth')

    test_image = './imgs/ski.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    plt.imsave("./ski.jpg", canvas[:, :, [2, 1, 0]])
    # plt.show()
