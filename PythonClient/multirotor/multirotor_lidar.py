# Python client example to get Lidar data from a car
#

import setup_path 
import airsim

import os
import sys
import math
import time
import argparse
import pprint
import numpy as np
import colorsys

import open3d as o3d

import ast

from pynput import keyboard

# Makes the car drive and get Lidar data
class LidarTest:

    def __init__(self, save_pcd, save_dir):

        # connect to the AirSim simulator
        # self.client = airsim.CarClient()
        self.client = airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.enableApiControl(True)
        # self.car_controls = airsim.CarControls()
        self.save_pcd = save_pcd
        self.save_dir = save_dir

        filename = cwd = os.path.dirname(os.path.realpath(__file__)) + "/../../docs/seg_rgbs.txt"
        self.color_map = {}
        with open(filename) as f:
            for line in f:
                (idx, rgb) = line.split("\t")
                rgb = ast.literal_eval(rgb)
                self.color_map[int(idx)] = np.array(rgb, dtype=np.dtype('i4'))
        print("Finish reading the colour map from ", filename)

        # objects = self.client.simListSceneObjects()
        # success = self.client.simSetSegmentationObjectID("road_[\w]*", 21);
        # for object in objects:
        #     print(object, " ",self.client.simGetSegmentationObjectID(object))
        # print( objects)


    def execute(self):
        # Collect events until released
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        try:
            pose = self.client.simGetVehiclePose()
            # yaw goes from -pi/2 to pi/2 clockwise, 0 is the same direction with x, and pi/2 same direction of y
            roll, pitch, yaw = self.quaternion_to_euler_angle_vectorized2(pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val)

            if key.char == 'w':
                pose.position.x_val += 1 * np.cos(yaw)
                pose.position.y_val += 1 * np.sin(yaw)
                self.client.simSetVehiclePose(pose, False)
                # self.client.moveByVelocityBodyFrameAsync(10, 0, 0, 10.0, vehicle_name = 'Cam1')
            elif key.char == 's':
                pose.position.x_val -= 1 * np.cos(yaw)
                pose.position.y_val -= 1 * np.sin(yaw)
                self.client.simSetVehiclePose(pose, False)
            elif key.char == 'a':
                pose.position.x_val += 1 * np.sin(yaw)
                pose.position.y_val -= 1 * np.cos(yaw)
                self.client.simSetVehiclePose(pose, False)
            elif key.char == 'd':
                pose.position.x_val -= 1 * np.sin(yaw)
                pose.position.y_val += 1 * np.cos(yaw)
                self.client.simSetVehiclePose(pose, False)
            elif key.char == 'q':
                pose.position.z_val += 0.5
                self.client.simSetVehiclePose(pose, False)
            elif key.char == 'e':
                pose.position.z_val -= 0.5
                self.client.simSetVehiclePose(pose, False)
            elif key.char == 'o':
                pose.orientation = pose.orientation + airsim.to_quaternion(0.0, 0.0, 1.7)
                self.client.simSetVehiclePose(pose, False)
                # self.client.moveByRollPitchYawZAsync(0.5, 0, 0, pose.position.z_val, 1.0)
            elif key.char == 'p':
                pose.orientation = pose.orientation + airsim.to_quaternion(0.0, 0.0, -1.7)
                self.client.simSetVehiclePose(pose, False)
                # self.client.moveByRollPitchYawZAsync(-0.5, 0, 0, pose.position.z_val, 1.0)

            elif key.char == 'l':
                lidarData = self.client.getLidarData();
                if (len(lidarData.point_cloud) < 3):
                    print("\tNo points received from Lidar data")
                else:
                    print("\t time_stamp: %d number_of_points: %d" % (lidarData.time_stamp, len(lidarData.point_cloud) / 3))
                    print("\t\tlidar position: %s" % (pprint.pformat(lidarData.pose.position)))
                    print("\t\tlidar orientation: %s" % (pprint.pformat(lidarData.pose.orientation)))
                    if self.save_pcd: 
                        self.write_lidarData_to_disk(lidarData)
                    

        except AttributeError:
            print('special key pressed: {0}'.format(key))

    def on_release(self, key):
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        elif key == keyboard.Key.up: 
            print("UP key pressed")


    def quaternion_to_euler_angle_vectorized2(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)

        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)

        return X, Y, Z

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    def define_segmentation_color(self, obj_id):

        minimum = min(obj_id)
        maximum = max(obj_id)
        red_color_offset = 30
        m_range = maximum - minimum + red_color_offset
        print("all object ids are: ", np.unique(obj_id))

        interp_color = (obj_id - minimum) / m_range #HSV hue values are 0-1

        print("all interp_color are: ", np.unique(interp_color))

        rgb = np.zeros(shape=(interp_color.shape[0],3))
        for idx, j in enumerate(interp_color):
            rgb[idx, :] = np.array(colorsys.hsv_to_rgb(j, 1, 0.8))
        return rgb

    def lookup_segmentation_from_colormap(self, obj_id):

        # print("all object ids are: ", np.unique(obj_id))

        rgb = np.zeros(shape=(obj_id.shape[0],3))
        for idx, j in enumerate(obj_id):
            rgb[idx, :] = self.color_map[j] / 255
        return rgb


    def write_lidarData_to_disk(self, lidarData):
        
        points = self.parse_lidarData(lidarData)
        obj_ids = np.array(lidarData.segmentation, dtype=np.dtype('i4'))
        rgb_color = self.lookup_segmentation_from_colormap(obj_ids)
        print("The number of lidar points is ", points.shape[0])
        
        time_stamp_str = str(int(lidarData.time_stamp // 1e9)) + "_" + str(int(lidarData.time_stamp % 1e9))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_color)
        save_path = self.save_dir + "/" + time_stamp_str + ".pcd"
        o3d.io.write_point_cloud(save_path, pcd)
        
        txt_save_path = self.save_dir + "/" + time_stamp_str + ".txt"
        np.savetxt(txt_save_path, np.hstack((points, obj_ids.reshape(obj_ids.shape[0],1))), fmt='%1.3f')
        
        pose_save_path = self.save_dir + "/" + time_stamp_str + "_pose.txt"
        pose_vector = np.hstack((lidarData.pose.position.to_numpy_array(), 
                                lidarData.pose.orientation.to_numpy_array())).reshape(1,-1)

        np.savetxt(pose_save_path, pose_vector, header="x y z qx qy qz qw")
        print("finish writing txt and pcd file to ", save_path)

    def stop(self):

        airsim.wait_key('Press any key to reset to original state')

        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")

# main
if __name__ == "__main__":
    args = sys.argv
    args.pop(0)

    arg_parser = argparse.ArgumentParser("Lidar.py makes car move and gets Lidar data")

    arg_parser.add_argument('--save-pcd', type=bool, help="save Lidar data to disk", default=False)
    arg_parser.add_argument('--save-dir', type=str, help="save Lidar data to disk", default="/tmp/airsim_lidar")

    args = arg_parser.parse_args(args)

    if args.save_pcd and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    lidarTest = LidarTest(args.save_pcd, args.save_dir)
    try:
        lidarTest.execute()
    finally:
        lidarTest.stop()



