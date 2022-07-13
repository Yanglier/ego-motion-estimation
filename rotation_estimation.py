import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import utils

def detectTrackPoints(img):
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    points = cv.goodFeaturesToTrack(img, mask=None, **feature_params)
    # points = cv.cornerSubPix(img, points, (5, 5), (-1, -1), criteria)

    # orb = cv.ORB_create()
    # kps = orb.detect(img, None)
    # points = [kp.pt for kp in kps]
    # points = np.asarray(points, np.float32).reshape(-1, 1, 2)

    return points

def OFRoEs_Img(path):
    color = np.random.randint(0, 255, (500, 3))
    mask_flow = np.zeros((180, 240, 3), dtype=np.uint8)
    lk_params = dict(winSize=(15, 15), maxLevel=8)

    R_gt = np.load(path + 'img_Rotations.npy')
    K, distCoeffs = utils.readCameraParas(path)
    img_path = os.listdir(path + 'images_Pro')
    img_path.sort()
    print(img_path)

    R21_pre = np.eye(3)
    R0 = R_gt[0].copy()
    R_estimate = [np.linalg.inv(R_gt[0])]
    track_num = []
    errors_ape = [0]
    errors_rpe = []

    eular_gt = [utils.rotationMatrixToEulerAngles(R0)]
    eular_estimate = [utils.rotationMatrixToEulerAngles(R0)]

    I1 = cv.imread(path + 'images_Pro/' + img_path[0], 0)
    p1 = detectTrackPoints(I1)
    for i in range(1, len(img_path)):
        RJ_gt = R_gt[i-1].copy()
        RI_gt = R_gt[i].copy()
        R12_gt = np.matmul(np.linalg.inv(RJ_gt), RI_gt)
        eular_gt.append(utils.rotationMatrixToEulerAngles(RI_gt))

        I2 = cv.imread(path + 'images_Pro/' + img_path[i], 0)
        p2, st, err = cv.calcOpticalFlowPyrLK(I1, I2, p1, None, **lk_params)
        if  len(p1) == 0: break
        ps1 = p1[st == 1]
        ps2 = p2[st == 1]

        print('Img %d and Img %d, good track points: %d' %(i-1, i, np.sum(st)))
        track_num.append(np.sum(st))
        frame = np.expand_dims(I2, -1).repeat(3, -1)
        for j in range(len(ps2)):
            mask_flow = cv.line(mask_flow, tuple(ps1[j]), tuple(ps2[j]), color[j].tolist(), 2)
            frame = cv.circle(frame, tuple(ps2[j]), 2, color[j].tolist(), -1)
        img = cv.add(frame, mask_flow)
        cv.imshow('frame', img)
        cv.waitKey(10)

        if ps1.shape[0] >= 4:
            points1 = cv.undistortPoints(ps1, K, distCoeffs).squeeze()
            points2 = cv.undistortPoints(ps2, K, distCoeffs).squeeze()
            E, mask = cv.findEssentialMat(points1, points2, np.eye(3), cv.RANSAC, prob=0.99, threshold=0.002)
            tmp = type(E)
            if tmp == np.ndarray and E.shape == (3, 3):
                num, R21, T, mask = cv.recoverPose(E, points1, points2, np.eye(3), mask=mask)
                print(str(num) + '/' + str(mask.shape[0]))
                if num/mask.shape[0] > 0:
                    r_tmp, _ = cv.Rodrigues(R21)
                    if np.linalg.norm(r_tmp) < 0.3:
                        RI = np.matmul(R21, R_estimate[i - 1])
                        # R21_pre = R21
                    else:
                        RI = R_estimate[i - 1]
                        R21 = np.eye(3)
                        # RI = np.matmul(R21_pre, R_estimate[i - 1])
                else:
                    RI = R_estimate[i - 1]
                    R21 = np.eye(3)
                    # RI = np.matmul(R21_pre, R_estimate[i - 1])
            else:
                RI = R_estimate[i - 1]
                R21 = np.eye(3)
                # RI = np.matmul(R21_pre, R_estimate[i - 1])
        else:
            RI = R_estimate[i - 1]
            R21 = np.eye(3)
            # RI = np.matmul(R21_pre, R_estimate[i - 1])

        R_estimate.append(RI)
        eular_estimate.append(utils.rotationMatrixToEulerAngles(RI.transpose()))

        R_error = np.matmul(RI, RI_gt)
        r_error, _ = cv.Rodrigues(R_error)
        errors_ape.append(np.linalg.norm(r_error))
        R_error_rpe = np.matmul(R21, R12_gt)
        r_error_rpe, _ = cv.Rodrigues(R_error_rpe)
        errors_rpe.append(np.linalg.norm(r_error_rpe))

        I1 = I2
        if len(ps2) < 500:
            p1 = detectTrackPoints(I1)
        else:
            p1 = np.expand_dims(ps2, 1)
        if i > 600:
            break

    print('Average track num: ', sum(track_num) / len(track_num))
    eular_gt = np.asarray(eular_gt)
    eular_estimate = np.asarray(eular_estimate)
    print('APE: ', sum(errors_ape) / len(errors_ape))
    print('RPE: ', sum(errors_rpe) / len(errors_rpe))
    np.save('boxes_ape/ape.npy', np.asarray(errors_ape))

    x = np.load(path + 'times_list.npy')
    x = x[:eular_estimate.shape[0]]

    plt.grid(linestyle="--")
    plt.plot(x, eular_estimate[:, 0], 'b-')
    plt.plot(x, eular_gt[:, 0], 'b--')
    plt.plot(x, eular_estimate[:, 1], 'g-')
    plt.plot(x, eular_gt[:, 1], 'g--')
    plt.plot(x, eular_estimate[:, 2], 'r-')
    plt.plot(x, eular_gt[:, 2], 'r--')

    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 12}
    plt.ylim([-260, 100])
    plt.xlabel('time/second', {'family': 'Times New Roman', 'weight': 'bold', 'size': 16.3})
    plt.ylabel('Euler angles/degree', {'family': 'Times New Roman', 'weight': 'bold', 'size': 16.3})
    plt.xticks(fontproperties = font)
    plt.yticks(fontproperties = font)
    legend = plt.legend(['X_estimate', 'X_groundtruth', 'Y_estimate', 'Y_groundtruth', 'Z_estimate', 'Z_groundtruth'],
                        prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10}, ncol=3, loc='upper right')
    plt.savefig('eular_plot.svg')
    plt.show()

if __name__ == '__main__':
    PATH = 'dataset_path'
    OFRoEs_Img(PATH)
    