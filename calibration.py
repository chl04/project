import cv2
import numpy as np

def zed_camera_calibration(camera_calibration, camera_mode, full_width, height):

    try:
        left = camera_calibration['LEFT_CAM_'+camera_mode];
        right = camera_calibration['RIGHT_CAM_'+camera_mode];
        common = camera_calibration['STEREO'];
    except:
        print("Error - specified config file does not contain valid ZED config.");
        exit(1);


    Lfx = float(left['fx']);
    Lfy = float(left['fy']);
    Lcx = float(left['cx']);
    Lcy = float(left['cy']);
    Lk1 = float(left['k1']);
    Lk2 = float(left['k2']);
    Lk3 = 0;
    Lp1 = float(left['p1']);
    Lp2 = float(left['p2']);


    Rfx = float(right['fx']);
    Rfy = float(right['fy']);
    Rcx = float(right['cx']);
    Rcy = float(right['cy']);
    Rk1 = float(right['k1']);
    Rk2 = float(right['k2']);
    Rk3 = 0;
    Rp1 = float(right['p1']);
    Rp2 = float(right['p2']);


    K_CameraMatrix_left = np.array([[Lfx, 0, Lcx],[ 0, Lfy, Lcy],[0, 0, 1]]);
    K_CameraMatrix_right = np.array([[Rfx, 0, Rcx],[ 0, Rfy, Rcy],[0, 0, 1]]);

    distCoeffsR = np.array([[Rk1], [Rk2], [Rk3], [Rp1], [Rp2]]);
    distCoeffsL = np.array([[Lk1], [Lk2], [Lk3], [Lp1], [Lp2]]);

    Baseline = float(common['Baseline']);
    CV = float(common['CV_'+camera_mode]);
    RX = float(common['RX_'+camera_mode]);
    RZ = float(common['RZ_'+camera_mode]);

    R_xyz_vector = np.array([[RX], [CV], [RZ]]);
    R,_ = cv2.Rodrigues(R_xyz_vector);

    T = np.array([[Baseline],[0],[0]]);

    Q = np.array([  [1, 0, 0, -1*Lcx],
                    [0, 1, 0, -1*Lcy],
                    [0, 0, 0, Lfx],
                    [0, 0, -1.0/Baseline, ((Lcx - Rcx) / Baseline)]]
                );

    return Lfx, Lfy, Lcx, Lcy, Baseline, K_CameraMatrix_left, K_CameraMatrix_right, R, T, Q;
