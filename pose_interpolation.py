import numpy as np
import utils
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def vec2sM(vec):
    return np.array([
            [0,-vec[2],vec[1], vec[3]],
            [vec[2],0,-vec[0], vec[4]],
            [-vec[1],vec[0],0, vec[5]],
            [0, 0, 0, 0]])

def sM2vec(sM):
    return np.array([sM[2][1],sM[0][2],sM[1][0], sM[0][3], sM[1][3], sM[2][3]])

def zerobase(R, T):
    R0_inv = np.linalg.inv(R[0])
    T0 = T[0].copy()
    for i in range(R.shape[0]):
        R[i] = np.matmul(R0_inv, R[i])
        T[i] = np.matmul(R0_inv, T[i] - T0)
    scale = np.linalg.norm(T[1])
    T /= scale
    return R, T

def poseInterpolate(timestamp_list1, path):
    timestamp_list2, R, T = utils.readPose(path)
    i, j = 0, 0
    R_interpo = []
    T_interpo = []
    eular = []
    time_list = []
    while (i<len(timestamp_list1) and j<len(timestamp_list2)):
        if timestamp_list1[i] < timestamp_list2[0]:
            i += 1
            continue
        if timestamp_list2[j] > timestamp_list1[i]:
            time_list.append(timestamp_list1[i])
            t1 = timestamp_list2[j-1]
            t2 = timestamp_list2[j]
            R1 = R[j-1].copy()
            T1 = T[j-1].copy()
            Transform1 = np.eye(4)
            Transform1[0:3, 0:3] = R1.copy()
            Transform1[0:3, 3] = T1.squeeze()
            R2 = R[j].copy()
            T2 = T[j].copy()
            Transform2 = np.eye(4)
            Transform2[0:3, 0:3] = R2.copy()
            Transform2[0:3, 3] = T2.squeeze()
            delta_trans = sM2vec(logm(np.matmul(np.linalg.inv(Transform1), Transform2))) / (t2-t1) * (timestamp_list1[i]-t1)
            Trans_inter = np.matmul(Transform1, expm(vec2sM(delta_trans)))

            R_interpo.append(Trans_inter[0:3, 0:3])
            T_interpo.append(Trans_inter[0:3, 3])
            eular.append(utils.rotationMatrixToEulerAngles(Trans_inter[0:3, 0:3]))
            i += 1
        j += 1

    R_interpo = np.asarray(R_interpo)
    T_interpo = np.asarray(T_interpo)
    # R_interpo, T_interpo = zerobase(R_interpo, T_interpo)
    np.save(path + 'img_Rotations.npy', R_interpo)
    R_interpo = Rotation.from_matrix(R_interpo).as_quat()

    eular = np.asarray(eular).squeeze()
    time_list = np.asarray(time_list)
    np.save(path + 'times_list.npy', time_list)
    print(time_list.shape)

    pose_gt = open(path + 'pose_GT.txt', mode='w')
    for i in range(eular.shape[0]):
        timestamp = str(time_list[i])
        px = str(T_interpo[i, 0])
        py = str(T_interpo[i, 1])
        pz = str(T_interpo[i, 2])
        qx = str(R_interpo[i, 0])
        qy = str(R_interpo[i, 1])
        qz = str(R_interpo[i, 2])
        qw = str(R_interpo[i, 3])
        pose_gt.write(timestamp + ' ' + px + ' ' + py + ' ' + pz + ' ' + qx + ' ' + qy + ' ' + qz + ' ' + qw + '\n')
    pose_gt.close()

    time_list_txt = open(path + 'times_list.txt', mode='w')
    for i in range(time_list.shape[0]):
        time_list_txt.write(str(time_list[i]) + '\n')
    time_list_txt.close()

    plt.plot(eular[:, 0])
    plt.plot(eular[:, 1])
    plt.plot(eular[:, 2])
    plt.show()

if __name__ == '__main__':
    PATH = 'dataset_path'
    timestamp_list, _ = utils.readImgRef(PATH)
    # poseInterpolate(timestamp_list, PATH)
    time_list_txt = open(PATH + 'times_list.txt', mode='w')
    for i in range(len(timestamp_list)):
        time_list_txt.write(str(timestamp_list[i]) + '\n')
    time_list_txt.close()