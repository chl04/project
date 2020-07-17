import cv2
import argparse
import sys
import math
import csv
import requests
import configparser
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import numpy as np
import shapely.geometry as geom
from calibration import *


def read_data(data_L, data_R):

    Lx_pixel=np.zeros(len(data_L)+1)
    Ly_pixel=np.zeros(len(data_L)+1)
    Rx_pixel=np.zeros(len(data_L)+1)
    Ry_pixel=np.zeros(len(data_L)+1)
    
    for i in range(1, len(data_L)):
        x1=int(data_L[i][2])
        x2=int(data_R[i][2])
        y1=int(data_L[i][3])
        y2=int(data_R[i][3])
        # Visibility=0
        if (data_L[i][1]=='0' or data_R[i][1]=='0'):
            Lx_pixel[i-1]=np.nan
            Ly_pixel[i-1]=np.nan
            Rx_pixel[i-1]=np.nan
            Ry_pixel[i-1]=np.nan
        else:
            Ly_pixel[i-1]=y1
            Lx_pixel[i-1]=x1
            Ry_pixel[i-1]=y2
            Rx_pixel[i-1]=x2
    return Lx_pixel, Ly_pixel, Rx_pixel, Ry_pixel

def estimate_hits(M, deg, window, Lx_pixel, Ly_pixel):
    #true_hits=[0,54,100,138,169,212]
    fit_error=np.zeros(M)
    for i in range(0,M,1):
        f_start = i
        f_end = i + window
        fitx = Lx_pixel[f_start:f_end]
        fity = Ly_pixel[f_start:f_end]
        idx = np.isfinite(fitx) & np.isfinite(fity)
        f1, err, _, _, _ = np.polyfit(fitx[idx], fity[idx], deg, full=True)
        p1 = np.poly1d(f1)
        if err.size==0:
            fit_error[i]=fit_error[i-1]
        else: fit_error[i]=err[0]
    hit=np.asarray(argrelextrema(fit_error, np.greater_equal, order=15)).flatten()
    hit=hit+10
    hit[0]=0
    return hit
    
def fit_2d(M, N, deg, X, Y, hit, sample_num):
    
    curve_data=np.zeros((M, sample_num, 2))
    points=np.zeros((M,100,2))
    points_new=np.zeros((M,100,2))
    
    for i in range(0, hit.size, 1):
        
        f_start = hit[i]
        if i==hit.size-1:
            f_end = N
        else:
            f_end = hit[i+1]
        fitx = X[f_start:f_end]
        fity = Y[f_start:f_end]
        idx = np.isfinite(fitx) & np.isfinite(fity)
        f1, err, _, _, _ = np.polyfit(fitx[idx], fity[idx], deg, full=True)
        p1 = np.poly1d(f1)

        x_sorted = np.sort(X[f_start:f_end])
        yvals = p1(x_sorted)
        
        points[hit[i],0:fitx.size,0]=fitx
        points[hit[i],0:fitx.size,1]=fity
        dataX = np.linspace(np.nanmin(fitx[idx]), np.nanmax(fitx[idx]),200)
        dataY = p1(dataX)
        curve_data[hit[i],:,0]=dataX
        curve_data[hit[i],:,1]=dataY
        
    
        coords = curve_data[hit[i],:,:]
        line = geom.LineString(coords)
        for j in range(0,100,1):
            point = geom.Point(points[hit[i],j,0], points[hit[i],j,1])

            if point.x==0 and point.y==0:
                break
            if np.isnan(points[hit[i],j,0]):
                points_new[hit[i],j,0]=np.nan
                points_new[hit[i],j,1]=np.nan
            else:
                point_on_line = line.interpolate(line.project(point))

                points_new[hit[i],j,0]=(point_on_line.x)
                points_new[hit[i],j,1]=(point_on_line.y)
    return points_new

def calc_depth(x1, x2, params):

    f, B = params;

    if (x1 != x2):
        depth = f * (B / (x1-x2));
    else:
        depth = 0

    return depth
    
def data_3d(L_points, R_points, hit, fx, fy, B, cx, cy):

    x_meter=[]
    y_meter=[]
    z_meter=[]
    
    for i in range(0, hit.size, 1):
        for j in range(0,100,1):
            x1=L_points[hit[i],j,0]
            x2=R_points[hit[i],j,0]
            y1=L_points[hit[i],j,1]
            if x1==0 and y1==0:
                break
            if np.isnan(x1):
                x_meter.append(None)
                z_meter.append(None)
                y_meter.append(None)
            else:
                depth = calc_depth(x1, x2, (fx, B))
                wX, wY, wZ = get_coordinate(fx, fy, cx, cy, x1, y1, depth/1000)
                x_meter.append(wX)
                z_meter.append(wY)
                y_meter.append(wZ)
    x = np.array(x_meter, dtype=float)
    y = np.array(y_meter, dtype=float)
    z = np.array(z_meter, dtype=float)
    return x, y, z
    
"""
def get_sec(time_str):

    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)       
"""

def get_coordinate(fx, fy, cx, cy, u, v, D):
    Z = D
    X = Z / fx * (u - cx)
    Y = Z / fy * (v - cy)
    return X, Y, Z
    
def fit_3d(x, y, z, hit, N, deg):

    x_new=np.zeros(N)
    y_new=np.zeros(N)
    z_new=np.zeros(N)
    curve_x=np.zeros(N)
    curve_y=np.zeros(N)
    curve_z=np.zeros(N)
    
    for i in range(0, hit.size, 1):
        f_start=hit[i]
        if i==hit.size-1:
            f_end = N
        else:
            f_end = hit[i+1]
        fitx1 = x[f_start:f_end]
        fity1 = y[f_start:f_end]
        fitz1 = z[f_start:f_end]
        idx = np.isfinite(fitx1) & np.isfinite(fity1) & np.isfinite(fitz1)
        
        yx= np.polyfit(fity1[idx], fitx1[idx], deg)
        pYX = np.poly1d(yx)
        yz= np.polyfit(fity1[idx], fitz1[idx], deg)
        pYZ = np.poly1d(yz)

        y_sorted = np.sort(fity1)
        x_pYX = pYX(y_sorted)
        z_pYZ = pYZ(y_sorted)
  
        fitx1_new, fity1_new, fitz1_new, pYX_new, pYZ_new = get_cleaned_fit(fitx1, fity1, fitz1, pYX, pYZ, deg)
        y_sorted_new = np.sort(fity1_new)
        x_pYX_new = pYX_new(y_sorted_new)
        z_pYZ_new = pYZ_new(y_sorted_new)

        x_new[f_start:f_end]=fitx1_new
        y_new[f_start:f_end]=fity1_new
        z_new[f_start:f_end]=fitz1_new
        curve_x[f_start:f_end]=x_pYX_new
        curve_y[f_start:f_end]=y_sorted_new
        curve_z[f_start:f_end]=z_pYZ_new
    return x_new, y_new, z_new, curve_x, curve_y, curve_z

def get_cleaned_fit(x, y, z, polyYX, polyYZ, deg):
    array_yz = abs(z - polyYZ(y))
    max_accept_deviation = 0.8
    mask_yz = array_yz >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yz)[0]))
    
    for i in rows_to_del:
        x[i]=np.nan
        y[i]=np.nan
        z[i]=np.nan
    
    array_yx = abs(x - polyYX(y))
    mask_yx = array_yx >= max_accept_deviation
    rows_to_del = np.asarray(tuple(te for te in np.where(mask_yx)[0]))
    
    for i in rows_to_del:
        x[i]=np.nan
        y[i]=np.nan
        z[i]=np.nan
    
    idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    cYX = np.polyfit(y[idx], x[idx], deg)
    pYX = np.poly1d(cYX)
    cYZ= np.polyfit(y[idx], z[idx], deg)
    pYZ = np.poly1d(cYZ)
    return x, y, z, pYX, pYZ

def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2,num-1:num])
    return lines

def plot_result(x_new, y_new, z_new, curve_x, curve_y, curve_z, N, hit):

    camPoints = np.vstack((x_new, y_new, z_new))
    curves = np.vstack((curve_x, curve_y, curve_z))
    
    data = [camPoints]
    data2 = [curves]
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    lines = [ax.plot(data[0][0,0:1], data[0][1,0:1], data[0][2,0:1], 'o')[0]]
    
    # set the axes properties
    ax.set_xlim3d([-4, 4])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 15])
    ax.set_ylabel('Depth')
    
    ax.set_zlim3d([np.nanmin(z_new), np.nanmax(z_new)])
    ax.set_zlabel('Y')
    plt.gca().invert_zaxis()
    ax.set_title('')

    ax.scatter3D(x_new,y_new,z_new)
    
    for i in range(0, hit.size, 1):
        f_start=hit[i]
        if i==hit.size-1:
            f_end = N
        else:
            f_end = hit[i+1]
        ax.plot(curve_x[f_start:f_end], curve_y[f_start:f_end], curve_z[f_start:f_end])
    
    plt.show()
    # animation
    #ani = animation.FuncAnimation(fig, update_lines, N, fargs=(data, lines), interval=30, repeat=False, blit=False)
    #ani.save('matplotsave.mp4', fps=30)
    

def main():
    cam_calibration = configparser.ConfigParser();
    cam_calibration.read("SN19889.conf");
    camera_mode = "HD";
    width = 1280
    height = 720
    N = 240
    deg = 3
    window = 15
    M = N - window
    sample_num = 200
    
    """camera parameters"""
    fx, fy, cx, cy, B, Kl, Kr, R, T, Q = zed_camera_calibration(cam_calibration, camera_mode, width, height)
    
    """read csv data"""
    Lx_pixel, Ly_pixel, Rx_pixel, Ry_pixel = read_data(data_L, data_R)
    
    """estimate hits"""
    hit = estimate_hits(M, deg, window, Lx_pixel, Ly_pixel)
    
    """2d curve fit, move points on curve"""
    L_points = fit_2d(M, N, deg, Lx_pixel, Ly_pixel, hit, sample_num)
    R_points = fit_2d(M, N, deg, Rx_pixel, Ry_pixel, hit, sample_num)
   
    """3d data in meters"""
    x, y, z = data_3d(L_points, R_points, hit, fx, fy, B, cx, cy)
    
    """3d curve fit"""
    x_new, y_new, z_new, curve_x, curve_y, curve_z = fit_3d(x, y, z, hit, N, deg)
    
    """plot"""
    plot_result(x_new, y_new, z_new, curve_x, curve_y, curve_z, N, hit)




with open('L_predict.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data_L = list(reader)

with open('R_predict.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data_R = list(reader)


if __name__=="__main__":
    main()
