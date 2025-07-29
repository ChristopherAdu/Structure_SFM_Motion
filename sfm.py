#!/usr/bin/env python3

import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import operator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go 
import gtsam
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2, NonlinearFactorGraph, SfmTrack, noiseModel,
                PinholeCameraCal3_S2, Point2, Point3, Pose3, PriorFactorPoint3, PriorFactorPose3, Values)




class SfmHelpers:
    def __init__(self,path, single_img=False, viz=False):
        self.path = path
        self.k = None
        self.viz = viz
        self.single_img = single_img

    def viz_images(self, imgs, titles=None):
        if not self.viz:
            return

        if self.single_img:  # to see the changes on the single image 
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
            ax.set_title(titles[0] if titles else "Image 1", fontsize=10)
            ax.axis('off')
        else:    # to see the changes on all images
            fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))

            if len(imgs) == 1:
                axs = [axs]

            for i in range(len(imgs)):
                axs[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
                axs[i].set_title(titles[i] if titles else f"Image {i+1}", fontsize=8)
                axs[i].axis('off')
            plt.tight_layout()


    def pre_process(self, img, create_mask=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if create_mask:
            # Captures reddish-brown Buddha colors
            lower = np.array([105, 10, 10])

            # Up to white background
            upper = np.array([255, 255, 255])
            
            thresh = cv2.inRange(img, lower, upper)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            mask = 255 - morph  # Inverts the mask correctly
            
            enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        
        return enhanced

    def get_images(self):
        imgs = [cv2.imread(os.path.join(self.path, file)) for file in sorted(os.listdir(self.path))]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs = [clahe.apply(cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) for img in imgs]
        self.viz_images(imgs)

        return imgs
    
    def set_intrinsics(self, imgs):     # to set the intrinsics of the camera
        if len(imgs[0].shape) == 2:
            self.height, self.width = imgs[0].shape
        else:
            self.height, self.width, _ = imgs[0].shape

        self.k = np.array([[1500, 0, self.width/2], [0, 1500, self.height/2], [0, 0, 1]])

        if self.viz:
            print(f"----Intrinsic matrix K set to----")
            print(self.k)

    def match_marker(self, imgd1, imgd2, img1=None, img2=None, kp1=None, kp2=None,
                 min_pts=50, max_ratio=0.75, relax_step=0.03, show_keypoints=False): 
        """
        Filters SIFT matches using Lowe's ratio test with adaptive relaxation.

        Parameters: 
        -   imgd1, imgd2: discriptors from the image1 and image 2
        -   img1 , img2: passing images if want to display keypoints
        -   kp1, kp2: passing the keypoints from the sift
        -   min_pts: Minimum number of good matches needed
        -   max_ratio:  starting ratio for the lowe's test
        -   relax_step: step size to relax the threshold if not enough matches

        Returns;
        -   amatches
        -   good_matches:  list of cv2.Dmatch objects filtered by ratio

        """
        if imgd1 is None or imgd2 is None:
            print(f"[Error] Descriptor is None check if SIFT feature extraction failed")
            return [],[]
    
            # Optional: Visualize keypoints before matching
        if show_keypoints and kp1 is not None and kp2 is not None:
            vis1 = cv2.drawKeypoints(img1, kp1, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            vis2 = cv2.drawKeypoints(img2, kp2, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            self.viz_images([vis1, vis2], titles=["Image 1 Keypoints", "Image 2 Keypoints"])

        # using FLANN matcher instead of BFMatcher for better matches 
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(imgd1, imgd2, k=2)
        
        # Filtering  out matches where we don't have 2 nearest neighbors
        matches = [m for m in matches if len(m) == 2]

        ratio = max_ratio
        good_matches = []

        while len(good_matches)<min_pts and ratio <= 0.95:
            good_matches.clear()
            for m,n in matches:
                if m.distance < ratio *n.distance:
                    good_matches.append(m)
                
            ratio += relax_step

        good_matches = sorted(good_matches, key=operator.attrgetter('distance'))
        
        return matches, good_matches
    
    def non_max_suppression(self, keypoints, descriptors):
        """
        parameters: 
        -   keypoints
        -   descriptors

        returns: nms_mask id's and filtered keypoints and descriptors 
        """
        binary_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        response = np.array([kp.response for kp in keypoints])
        mask = np.flip(np.argsort(response))  # highest response first

        points_list = np.rint([point.pt for point in keypoints])
        points_list = points_list[mask].astype(int)  # keeping keypoints with high strength and then int

        nms_mask = []

        for point, index in zip(points_list, mask):
            if binary_mask[point[1], point[0]] == 0:
                nms_mask.append(index)
                cv2.circle(binary_mask, tuple(point), 1, 255, -1)
        
        keypoints = np.array(keypoints)[nms_mask]
        descriptors = np.array(descriptors)[nms_mask]

        return nms_mask, keypoints, descriptors
    
    def essential_matrix(self, src_pts, dst_pts):

        """
        parameters: 
        -   matched points

        return: 
        - E:    Essential matrix
        -   matrix
        -   Translation vector
        -   RANSAC inlier mask
        """

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inlier_src_pts = src_pts[mask.ravel() == 1]
        inlier_dst_pts = dst_pts[mask.ravel() == 1]

        # _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.k)

        if self.viz:
            print("Essential Matrix E:\n", E)
        
        
        return E, mask, inlier_src_pts, inlier_dst_pts
    
    def posesFromE(self, E, img1_inlier_pts, img2_inlier_pts):
        """
        Manually decomposes E to (R, t) using SVD + cheirality check
        """
        U, _, Vt = np.linalg.svd(E)
        Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        R1 = U @ Y @ Vt
        R2 = U @ Y.T @ Vt
        t1 = U[:, 2]
        t2 = -U[:, 2]

        Rt_list = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

        for i, (R, t) in enumerate(Rt_list):
            if np.linalg.det(R) < 0:
                Rt_list[i] = (-R, -t)

        num_pos_pts = []
        for R, t in Rt_list:
            P1 = self.k @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.k @ np.hstack((R, t.reshape(3, 1)))

            count = 0
            for pt1, pt2 in zip(img1_inlier_pts, img2_inlier_pts):
                X = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
                X /= X[3]
                X = X[:3]
                if X[2] > 0 and (R[2] @ (X - t.reshape(3, 1)))[0] > 0:
                    count += 1
            num_pos_pts.append(count)

        best_idx = np.argmax(num_pos_pts)
        R, t = Rt_list[best_idx]

        Tr = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([[0, 0, 0, 1]])))

        p1 = self.k @ np.hstack((np.eye(3), np.zeros((3,1))))
        p2 = self.k @ np.hstack((R, t.reshape(3,1)))

        return R, t
    
    # Additional helper function to add to your SfmHelpers class:
    def triangulate_pts(self, pose_c1, pose_c2, filtered_src_pts, filtered_dst_pts):
        """
        Simplified triangulation like ideal code - only filter by positive depth
        """
        # Triangulate points
        pts_4d_hom = cv2.triangulatePoints(pose_c1, pose_c2, filtered_src_pts.T, filtered_dst_pts.T)
        
        print(f"pts_3d shape before masking = {pts_4d_hom.shape}")

        # Inhomogenize points
        pts_3d = pts_4d_hom/pts_4d_hom[3,:]
        pts_3d = pts_3d[:3,:]

        # ONLY filter by positive depth - like ideal code
        mask = pts_3d[2, :] > 0
        pts_3d = pts_3d[:, mask]
        img1_pts = filtered_src_pts[mask]
        img2_pts = filtered_dst_pts[mask]
        
        print(f"pts_3d shape after masking = {pts_3d.shape}")
        
        return pts_3d.T, img1_pts, img2_pts
        
    # def triangulate_pts(self, pose_c1, pose_c2, filtered_src_pts, filtered_dst_pts):
    #     # filtered_src_pts, filtered_dst_pts: shape (N, 2)

    #     pts_4d_hom = cv2.triangulatePoints(pose_c1, pose_c2, filtered_src_pts.T, filtered_dst_pts.T)
    #     pts_3d = pts_4d_hom[:3, :] / pts_4d_hom[3, :]

    #     # pts_3d shape: (3, N), transpose to (N, 3)
    #     pts_3d = pts_3d.T

    #     print(f"shape of pts_3d before masking = {pts_3d.shape}")

    #     # Create mask based on positive depth
    #     mask = pts_3d[:, 2] > 0

    #     # Apply mask to points in (N, 2) format
    #     img1_pts = filtered_src_pts[mask]
    #     img2_pts = filtered_dst_pts[mask]
    #     pts_3d = pts_3d[mask]

    #     print(f"shape of pts_3d after masking = {pts_3d.shape}")
    #     print(f"\nTriangulated {pts_3d.shape[0]} 3D points")


    #     return pts_3d, img1_pts, img2_pts
    
    def poses_from_pnp(self, pts_3d, pts_2d, k):

        # rvec, tvec: Pose of the 3D points in camera coordinate system
        success, rvec, t, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, self.k, None)
         
        R, _ = cv2.Rodrigues(rvec)

        Tr = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0, 0, 0, 1])))

        return R, t, Tr 
    
class TrackManager:
    def __init__(self):

        self.tracks = {}
        self.camera_poses = []
        self.gtsam_camera_poses = []

    def add_points(self, pts_3d, img1_pts, img2_pts, pose_idx1, pose_idx2):
        
        for p3d, img1_pt, img2_pt in zip(pts_3d, img1_pts, img2_pts):
            img1_pair = (pose_idx1, tuple(img1_pt))
            img2_pair = (pose_idx2, tuple(img2_pt))
            self.tracks[tuple(p3d)] = [img1_pair, img2_pair]
            
            track_found = False
            for track_key in self.tracks:
                if img1_pair in self.tracks[track_key] or img2_pair in self.tracks[track_key]:
                    print("Track found. Printing current and existing 3d point for comparison")
                    print(tuple(p3d))
                    print(track_key)

                    if img1_pair not in self.tracks[track_key]:
                        self.tracks[track_key].append(img1_pair)
                    if img2_pair not in self.tracks[track_key]:
                        self.tracks[track_key].append(img2_pair)
                    track_found = True

                    break
            # if there are no existing tracks found create a new one    
            if not track_found:
                self.tracks[tuple(p3d)] = [img1_pair,img2_pair]

        print(f"Number of tracks: {len(self.tracks)}")
        return self.tracks
    
    def common_pts(self, img1_pts, img2_pts):
        """
        parameters: 
        -    img1_pts, img2_pts

        return:
        -   3d points and their corresponding 2d points in img2 for pnp
        """

        match_3d = []
        match_2d = []

        for i, point in enumerate(img1_pts):
            point_tuple = tuple(point)
            
            for track, measurement in self.tracks.items():
                # Loop over the measurement list to see if point_tuple exists
                found = False
                for m in measurement:
                    if point_tuple == m[1]:
                        found = True
                        break

                # If found, add corresponding 3D and 2D points
                if found:
                    match_3d.append(np.array(track))
                    match_2d.append(np.array(img2_pts[i]))
                    break  # Break outer loop to avoid checking other tracks for same point
                    
        return np.array(match_3d), np.array(match_2d)
    
    
    # def plot_scene(self):
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     pts_3d = np.array([np.array(p) for p in self.tracks.keys()], dtype=np.float64)

    #     ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], c='blue', s=3, label='Triangulated Points')

    #     for i, (R, t) in enumerate(self.camera_poses):
    #         C = (-R.T @ t).reshape(-1)
    #         ax.scatter(C[0], C[1], C[2], c='red', marker='o', s=40)
    #         ax.text(C[0], C[1], C[2], f'C{i}', color='black')

    #         cam_dir = (R.T @ np.array([0, 0, 1])).reshape(-1)
    #         ax.quiver(C[0], C[1], C[2], cam_dir[0], cam_dir[1], cam_dir[2],
    #                 length=0.05, color='green')

    #     ax.set_title('3D Scene with Triangulated Points and Camera Poses')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.legend()
    #     ax.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    #     ax.legend()
    #     ax.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    def addPoints_tracks(self, pts_3d, img1_pts, img2_pts, pose_idx1, pose_idx2):
        self.tracks = []
        for P3d, img1_pt, img2_pt in zip(pts_3d, img1_pts, img2_pts):
            p = Point3(np.array(P3d))
            track = SfmTrack(p)
            trackPoints = [tuple(t.p) for t in self.tracks]
            if tuple(track.p) not in trackPoints:                                        # Possible error in result since triangulated 3d point can be different
                track.addMeasurement(pose_idx1, Point2(np.array(img1_pt)))
                # print("Added new track")
                track.addMeasurement(pose_idx2, Point2(np.array(img2_pt)))
                # print("Added new track")
                self.tracks.append(track)
                # print("=======================================",tuple(track.p))
                # print(track.measurementMatrix())
                # print(track.measurement(0))
                # print("Added new track=============================================")
            
            else:
                track = self.tracks[trackPoints.index(tuple(track.p))]
                print("Track already exists", tuple(track.p))
                # print(track.measurementMatrix())
                # print(tuple(track.measurement(1)))
                m1 = (pose_idx1, tuple(img1_pt))
                m2 = (pose_idx2, tuple(img2_pt))
                im_pairs = []
                for i in  range(len(track.measurementMatrix())):
                    im_pairs.append((track.indexVector()[i], tuple(track.measurementMatrix()[i])))
                    print("IM PAIR: ", im_pairs[i])
                if m1 not in im_pairs:
                    track.addMeasurement(pose_idx1, Point2(np.array(img1_pt)))
                    print("Added measurement to existing track")
                    print(track.measurementMatrix())
                if m2 not in im_pairs:
                    track.addMeasurement(pose_idx2, Point2(np.array(img2_pt)))
                    print("Added measurement to existing track")
                    print(track.measurementMatrix())
        print(f"Number of tracks: {len(self.tracks)}")
            
        #for t in self.tracks:
        #    print(f"track.measurements = {[tuple(t.measurement(i)) for i in range(len(t.measurements))]}")

        return self.tracks
    
    def common_pts_tracks(self, img1_pts, img2_pts):
        '''
        Returns lists of 3d points and their corresponding
        2d points in image2 for PnP
        '''
        match_3d = []
        match_2d = []
    
        for i, point in enumerate(img1_pts):
            for track in self.tracks:
                if Point2(np.array(point)) in track.measurementMatrix():
                    match_3d.append(np.array(track.p))              # Possible error 
                    match_2d.append(np.array(img2_pts[i]))

        return np.array(match_3d), np.array(match_2d)
    
    def filter_points(self, perc):
        '''
        Filter top given percentile 3D points based on depth
        '''
        points_3d = np.array(list(self.tracks.keys()))
        depths = points_3d[:, 2]
        percentile = np.percentile(depths, perc)
        filtered_points_3d = points_3d[depths > percentile]

        for point in filtered_points_3d:
            del self.tracks[tuple(point)]


class Plotter:
    def __init__(self, points_3d, poses, filter):
        self.points_3d = np.array(points_3d)
        self.filter = filter
        self.poses = poses

        # filter the points which has any coordinate beyond 10,000
        filtered_points_3d = self.points_3d[(np.abs(self.points_3d) <= filter).all(axis = 1)]
        self.points_3d = filtered_points_3d

    def plot_3d_go(self):
        """
        using plotly to plot the 3d points and camera points in a 3d plot
        """

        total_colors = np.zeros((1,3))
        x = self.points_3d[:,0]
        y = self.points_3d[:,1]
        z = self.points_3d[:,2]

        fig = go.Figure()
        scatter = fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker = dict(size=1.2, color=total_colors/25)))

        # Plot camera poses

        for i in range(len(self.poses)):
            pose = self.poses[i]
            x, y, z = pose[0:3, 3]
            u, v, w = pose[0:3, 0:3] @ np.array([0, 0, 1])
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker = dict(size=3, color='red')))

        fig.show()

    def plot_3d_plt(self):
        '''
        Uses Matplotlib to plot the 3D points and camera poses in a 3D plot
        '''
        def plot_camera(ax, R, t, scale=0.01):
            # Define the camera pyramid vertices in camera coordinate system
            camera_body = np.array([
                [0, 0, 0],  # Camera center
                [1, 1, 2],  # Top-right corner
                [1, -1, 2],  # Bottom-right corner
                [-1, -1, 2],  # Bottom-left corner
                [-1, 1, 2]  # Top-left corner
            ]) * scale
            
            # Transform camera body to world coordinate system
            camera_body = (R @ camera_body.T).T + t
            
            # Define the six faces of the camera pyramid
            verts = [
                [camera_body[0], camera_body[1], camera_body[2]],  # Side 1
                [camera_body[0], camera_body[2], camera_body[3]],  # Side 2
                [camera_body[0], camera_body[3], camera_body[4]],  # Side 3
                [camera_body[0], camera_body[4], camera_body[1]],  # Side 4
                [camera_body[1], camera_body[2], camera_body[3], camera_body[4]]  # Base
            ]

            ax.add_collection3d(Poly3DCollection(verts, color='brown', alpha=0.5))

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D points
        ax.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='black', s=1)

        # Plot camera poses
        for pose in self.poses:
            # Extract the rotation and translation from the pose
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Plot the camera with orientation
            plot_camera(ax, R, t)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_3d_o3d(self, point_size=2, save=None):
        '''
        Uses Open3D to plot the 3D points and camera poses in a 3D plot
        '''

        # Create an Open3D point cloud object and negate y and z coordinates since image and world coordinates are different
        negated_points_3d = self.points_3d.copy()
        negated_points_3d[:, 1] = -negated_points_3d[:, 1]
        negated_points_3d[:, 2] = -negated_points_3d[:, 2]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(negated_points_3d)
        point_cloud.paint_uniform_color([0, 0, 0])  # Color all points black

        # Create a list to store camera spheres
        camera_spheres = []

        # Add camera poses as spheres and negate y and z coordinates
        for pose in self.poses:
            # Extract the translation from the pose
            t = pose[:3, 3]

            # Negate y and z coordinates in the translation vector
            t_negated = t.copy()
            t_negated[1] = -t_negated[1]
            t_negated[2] = -t_negated[2]

            # Create a sphere to represent the camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
            sphere.translate(t_negated)
            sphere.paint_uniform_color([1, 0, 0])  # Color the spheres red
            camera_spheres.append(sphere)

        # Create a visualizer and add geometry
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        for camera_sphere in camera_spheres:
            vis.add_geometry(camera_sphere)

        # Adjust the point size in the rendering options
        render_option = vis.get_render_option()
        render_option.point_size = point_size

        if save is not None:
            # Save the point cloud to a file
            o3d.io.write_point_cloud(save, point_cloud)
            print(f"Point cloud saved as {save}")

        # Run the visualizer
        vis.run()
        vis.destroy_window()
    
class GtsamOptimiser:
    def __init__(self, pc, k):
        self.pc = pc
        self.k = k  # Fixed: was self.K

        # Create factor graph and initialize estimates
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()  # Fixed: was values() - should be Values()

    def initialize_factor_graph(self):
        L = gtsam.symbol_shorthand.L
        X = gtsam.symbol_shorthand.X

        # Extract relevant values from K to construct gtsam camera calibration parameters
        gtsam_K = Cal3_S2(
            self.k[0, 0],  # fx - focal length x (Fixed: was self.K)
            self.k[1, 1],  # fy - focal length y 
            0.0,           # skew (usually 0)
            self.k[0, 2],  # cx - principal point x
            self.k[1, 2]   # cy - principal point y 
        )

        # How much do we trust pixel measurements?
        measurement_noise = noiseModel.Isotropic.Sigma(2, 10.0)  # ± 10 pixels uncertainty

        # Fix the first camera pose to prevent solution from drifting in space
        pose_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3,  # ±0.3 rad on roll,pitch,yaw    
                      0.1, 0.1, 0.1])  # ±0.1m on x,y,z 
        )
        
        # CONSTRAINT: "First camera is approximately here" 
        factor = PriorFactorPose3(
            X(0),  # First camera pose 
            Pose3(self.pc.gtsam_camera_poses[0]),  # Initial guess 
            pose_noise  # How certain we are 
        ) 
        self.graph.push_back(factor)

        # Add initial estimates for camera poses
        for i, pose in enumerate(self.pc.gtsam_camera_poses):
            self.initial_estimate.insert(X(i), Pose3(pose))

        # FOR EACH 3D LANDMARK that was tracked 
        i = 0
        for point, measurements in self.pc.tracks.items(): 
            # FOR EACH CAMERA that saw this landmark
            for j, measurement in enumerate([measurements[k][1] for k in range(len(measurements))]):
                # CREATE CONSTRAINT: "Landmark L(i) should project to pixel 'measurement' 
                # when viewed from camera X(camera_id)"
                factor = GenericProjectionFactorCal3_S2(
                    Point2(np.array(measurement)),  # OBSERVED pixel
                    measurement_noise,              # Trust level
                    X(measurements[j][0]),          # Camera pose 
                    L(i),                          # Landmark position
                    gtsam_K                        # Camera model
                )
                self.graph.push_back(factor)
            
            # Add initial estimate for this landmark
            self.initial_estimate.insert(L(i), Point3(np.array(point)))
            i += 1

        print(f"Added {i} landmarks to factor graph")

        # Add prior on the position of the first landmark to fix scale
        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        factor = PriorFactorPoint3(
            L(0), 
            Point3(np.array(list(self.pc.tracks.keys())[0])),  # Fixed: convert to Point3
            point_noise
        )
        self.graph.push_back(factor)  

        print(f"Total factors in graph: {self.graph.size()}")
        print(f"Total variables: {self.initial_estimate.size()}")

        return self.graph, self.initial_estimate, L, X

    def optimize(self):
        # Configure optimizer parameters
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        
        # Create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, 
            self.initial_estimate, 
            params
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        return result
    

def main():
    """
    Main function using existing method names from your SfmHelpers class
    """
    path_to_images = "buddha_images"  # Adjust path as needed
    
    sfm = SfmHelpers(path=path_to_images, single_img=False, viz=True)
    images = sfm.get_images()
    sfm.set_intrinsics(images)
    
    # Initialize exactly like ideal code
    track_manager = TrackManager()
    track_manager.camera_poses.append(np.eye(4))  # First camera pose is origin
    track_manager.gtsam_camera_poses.append(np.linalg.inv(track_manager.camera_poses[0]))

    print(f"Number of images = {len(images)}")
    print("\nStarting initial triangulation....")
    
    for i in range(len(images)-1):
        print(f"\nidx = {i}")
        
        # Feature detection using your existing methods
        if i == 0:
            nfeatures = 1000
        else:
            nfeatures = 5000
            
        # Extract SIFT features
        sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=3, contrastThreshold=0.04)
        kp1, des1 = sift.detectAndCompute(images[i], None)
        kp2, des2 = sift.detectAndCompute(images[i+1], None)
        
        # Apply Non-Max Suppression using your existing method
        _, kp1_nms, des1_nms = sfm.non_max_suppression(kp1, des1)
        _, kp2_nms, des2_nms = sfm.non_max_suppression(kp2, des2)
        
        # Match features using your existing method
        matches, good_matches = sfm.match_marker(
            des1_nms, des2_nms,
            img1=images[i], img2=images[i+1],
            kp1=kp1_nms, kp2=kp2_nms,
            min_pts=150 if i == 0 else 100,
            max_ratio=0.75,
            relax_step=0.02,
            show_keypoints=(i == 0)  # Only show for first pair
        )
        
        print(f"Number of good matches between images {i} and {i+1} = {len(good_matches)}")
        
        # Extract matched point coordinates
        src_pts = np.array([kp1_nms[m.queryIdx].pt for m in good_matches])
        dst_pts = np.array([kp2_nms[m.trainIdx].pt for m in good_matches])
        
        # Get essential matrix and inliers using your existing method
        E, mask, src_pts_inliers, dst_pts_inliers = sfm.essential_matrix(src_pts, dst_pts)

        if i == 0:
            # First pair - recover pose from essential matrix
            R, t = sfm.posesFromE(E, src_pts_inliers, dst_pts_inliers)
            TrMat = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0, 0, 0, 1])))
            track_manager.camera_poses.append(TrMat)
            track_manager.gtsam_camera_poses.append(np.linalg.inv(TrMat))

            # Build projection matrices
            P1 = np.array(sfm.k @ track_manager.camera_poses[i][:3,:])
            P2 = np.array(sfm.k @ track_manager.camera_poses[i+1][:3,:])

            # Triangulate using your existing method
            pts_3d, src_pts_valid, dst_pts_valid = sfm.triangulate_pts(P1, P2, src_pts_inliers, dst_pts_inliers)
            
            # Add points to tracks using your existing method
            track_manager.add_points(pts_3d, src_pts_valid, dst_pts_valid, i, i+1)

        else:
            # Subsequent pairs - use PnP
            common_3d, common_2d_dst = track_manager.common_pts(src_pts_inliers, dst_pts_inliers)
            print(f"Number of 3D and 2D common points: {common_3d.shape}, {common_2d_dst.shape}")

            if len(common_3d) >= 6:  # Need minimum points for PnP
                # Use your existing PnP method
                R_pnp, t_pnp, TrMat = sfm.poses_from_pnp(common_3d, common_2d_dst, sfm.k)
                track_manager.camera_poses.append(TrMat)
                track_manager.gtsam_camera_poses.append(np.linalg.inv(TrMat))

                # Build projection matrices
                P1 = sfm.k @ track_manager.camera_poses[i][:3,:]
                P2 = sfm.k @ track_manager.camera_poses[i+1][:3,:]

                # Triangulate new points
                pts_3d, src_pts_valid, dst_pts_valid = sfm.triangulate_pts(P1, P2, src_pts_inliers, dst_pts_inliers)
                track_manager.add_points(pts_3d, src_pts_valid, dst_pts_valid, i, i+1)
            else:
                print(f"Not enough common points for PnP ({len(common_3d)}), using identity")
                track_manager.camera_poses.append(np.eye(4))
                track_manager.gtsam_camera_poses.append(np.eye(4))

    # Filter and plot exactly like ideal code
    track_manager.filter_points(90)
    pts_3d_plot = np.array([point for point in track_manager.tracks.keys()])
    print(f"Number of 3D points in the point cloud = {len(pts_3d_plot)}")
    
    # # Simple matplotlib visualization
    # if len(pts_3d_plot) > 0:
    #     fig = plt.figure(figsize=(15, 5))
        
    #     # Plot 1: 3D view with cameras
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     ax1.scatter(pts_3d_plot[:, 0], pts_3d_plot[:, 1], pts_3d_plot[:, 2], c='black', s=1)
        
    #     # Plot camera positions with smaller markers
    #     for i, pose in enumerate(track_manager.camera_poses):
    #         if len(pose.shape) == 2 and pose.shape == (4, 4):
    #             # Extract translation from 4x4 matrix
    #             t = pose[:3, 3]
    #             ax1.scatter(t[0], t[1], t[2], c='red', s=20, marker='o')
    #             ax1.text(t[0], t[1], t[2], f'C{i}', fontsize=6, color='black')
        
    #     ax1.set_xlabel('X')
    #     ax1.set_ylabel('Y')
    #     ax1.set_zlabel('Z')
    #     ax1.set_title('3D Reconstruction with Cameras')

        
    # Simple matplotlib visualization
    if len(pts_3d_plot) > 0:
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: 3D view with cameras
        ax1 = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax1.scatter(pts_3d_plot[:, 0], pts_3d_plot[:, 1], pts_3d_plot[:, 2], c='black', s=1)

        # Plot camera positions with smaller markers
        for i, pose in enumerate(track_manager.camera_poses):
            if len(pose.shape) == 2 and pose.shape == (4, 4):
                t = pose[:3, 3]
                ax1.scatter(t[0], t[1], t[2], c='red', s=20, marker='o')
                ax1.text(t[0], t[1], t[2], f'C{i}', fontsize=6, color='black')

        # --- Remove axes and background ---
        ax1.set_axis_off()  # Removes axis lines and ticks
        ax1.grid(False)     # Removes grid
        ax1.set_title('')   # Optional: remove title
        fig.patch.set_facecolor('white')  # Optional: ensure white background

        
        
        # # Plot 2: Top view (XY) - Rotated 180 degrees clockwise
        # ax2 = fig.add_subplot(132)
        # # Rotate 180 degrees clockwise: (x,y) -> (-x,-y)
        # ax2.scatter(-pts_3d_plot[:, 0], -pts_3d_plot[:, 1], c='blue', s=1)
        # for i, pose in enumerate(track_manager.camera_poses):
        #     if len(pose.shape) == 2 and pose.shape == (4, 4):
        #         t = pose[:3, 3]
        #         ax2.scatter(-t[0], -t[1], c='red', s=15, marker='o')
        #         ax2.text(-t[0], -t[1], f'C{i}', fontsize=6)
        # ax2.set_xlabel('X (rotated)')
        # ax2.set_ylabel('Y (rotated)')
        # ax2.set_title('Top View (XY) - 180° Rotated')
        # ax2.grid(True)
        # ax2.axis('equal')
        
        # # Plot 3: Side view (XZ)
        # ax3 = fig.add_subplot(133)
        # ax3.scatter(pts_3d_plot[:, 0], pts_3d_plot[:, 2], c='blue', s=1)
        # for i, pose in enumerate(track_manager.camera_poses):
        #     if len(pose.shape) == 2 and pose.shape == (4, 4):
        #         t = pose[:3, 3]
        #         ax3.scatter(t[0], t[2], c='red', s=15, marker='o')
        #         ax3.text(t[0], t[2], f'C{i}', fontsize=6)
        # ax3.set_xlabel('X')
        # ax3.set_ylabel('Z (Depth)')
        # ax3.set_title('Side View (XZ)')
        # ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

        # Print some statistics
        print(f"Point cloud statistics:")
        print(f"X range: [{pts_3d_plot[:, 0].min():.3f}, {pts_3d_plot[:, 0].max():.3f}]")
        print(f"Y range: [{pts_3d_plot[:, 1].min():.3f}, {pts_3d_plot[:, 1].max():.3f}]")
        print(f"Z range: [{pts_3d_plot[:, 2].min():.3f}, {pts_3d_plot[:, 2].max():.3f}]")

        # GTSAM optimization
        optimizer = GtsamOptimiser(track_manager, sfm.k)
        graph, initial_estimate, L, X = optimizer.initialize_factor_graph()
        print('Initial Error = {}'.format(graph.error(initial_estimate)))

        result = optimizer.optimize()
        print('Final Error = {}'.format(graph.error(result)))

        # Extract optimized results
        optimized_pt_cloud = []
        i = 0
        for point in track_manager.tracks.keys():
            try:
                opt_point = result.atPoint3(L(i))
                optimized_pt_cloud.append([opt_point.x(), opt_point.y(), opt_point.z()])
            except:
                pass
            i+=1

        if optimized_pt_cloud:
            optimized_pt_cloud = np.array(optimized_pt_cloud)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(optimized_pt_cloud[:, 0], optimized_pt_cloud[:, 1], optimized_pt_cloud[:, 2], c='red', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Optimized 3D Reconstruction')
            plt.show()
    else:
        print("No 3D points to visualize!")

    return track_manager, optimizer, result if 'result' in locals() else None

if __name__ == "__main__":
    track_manager, optimizer, result = main()



    

    





  





        
    

    
    







# if __name__ == "__main__":
    
#     path_to_images = "/home/iso/Documents/MATLAB/GIT/Structure_SFM_Motion/buddha_images"

#     sfm = SfmHelpers(path=path_to_images, single_img=True, viz=True)

#     images = sfm.get_images()
#     sfm.set_intrinsics(images)


#     img1, img2 = images[0], images[1]

    

#     # Extract SIFT keypoints and descriptors
#     sift = cv2.SIFT_create()

#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     # 3. Non-Max Suppression
#     _, kp1_nms, des1_nms = sfm.non_max_suppression(kp1, des1)
#     _, kp2_nms, des2_nms = sfm.non_max_suppression(kp2, des2)



    

    
#     # 4. Match features and visualize keypoints (inside match_marker)
#     matches, good_matches = sfm.match_marker(des1_nms, des2_nms,
#                                             img1=img1, img2=img2,
#                                             kp1=kp1_nms, kp2=kp2_nms,
#                                             min_pts=150, max_ratio=0.8,
#                                             relax_step=0.02,
#                                             show_keypoints=True)

#     img1_nms_viz = cv2.drawKeypoints(img1, kp1_nms, None, color=(255, 0, 0),
#                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     img2_nms_viz = cv2.drawKeypoints(img2, kp2_nms, None, color=(0, 255, 0),
#                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#     # 5. Visualize Keypoints after NMS
#     img1_nms_viz = cv2.drawKeypoints(img1, kp1_nms, None, color=(255, 0, 0),
#                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     img2_nms_viz = cv2.drawKeypoints(img2, kp2_nms, None, color=(0, 255, 0),
#                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     sfm.viz_images([img1_nms_viz, img2_nms_viz], titles=["Image 1 (NMS)", "Image 2 (NMS)"])



#     # 6. Draw good matches
#     match_img = cv2.drawMatches(img1, kp1_nms, img2, kp2_nms, good_matches, None,
#                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     sfm.viz_images([match_img], titles=["Matched Features"])

#     # 7. Extract matched locations
#     matched_pts1 = np.float32([kp1_nms[m.queryIdx].pt for m in good_matches])
#     matched_pts2 = np.float32([kp2_nms[m.trainIdx].pt for m in good_matches])

#     # 8. Compute Essential Matrix and Recover Pose
#     #E, R, t, inlier_mask = sfm.essential_matrix(matched_pts1, matched_pts2)
#     E, inlier_mask = sfm.essential_matrix(matched_pts1, matched_pts2)
    
#     # Filter inlier points using the inlier_mask
#     matched_pts1_inliers = matched_pts1[inlier_mask.ravel() == 1]
#     matched_pts2_inliers = matched_pts2[inlier_mask.ravel() == 1]

#     # Use manual pose estimation
#     R, t= sfm.posesFromE(E, matched_pts1_inliers.T, matched_pts2_inliers.T)


    

#     print("\nManual posesFromE Output:")
#     print("Rotation R\n", R)
#     print("Translation t:\n", t)

#     # 9. Build projection matrices from recovered pose
#     pose_c1 = sfm.k @ np.hstack((np.eye(3), np.zeros((3, 1))))
#     pose_c2 = sfm.k @ np.hstack((R, t.reshape(3, 1)))

#     # 10. Transpose inlier points to shape (2, N)
#     # pts1_norm = matched_pts1_inliers.T.astype(np.float32)  # shape (2, N)
#     # pts2_norm = matched_pts2_inliers.T.astype(np.float32)  # shape (2, N)

#     # pass points in (N, 2)  
#     pts_3d, img1_valid, img2_valid = sfm.triangulate_pts(pose_c1, pose_c2, matched_pts1_inliers, matched_pts2_inliers)

    

#     # Convert filtered points (N, 2) to cv2.KeyPoint list
#     kp1_filtered = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=3) for pt in img1_valid]
#     kp2_filtered = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=3) for pt in img2_valid]

#     # Draw keypoints on original images (copy to avoid modifying originals)
#     img1_filtered_kp = cv2.drawKeypoints(img1.copy(), kp1_filtered, None, color=(0, 255, 0), flags=0)
#     img2_filtered_kp = cv2.drawKeypoints(img2.copy(), kp2_filtered, None, color=(0, 255, 0), flags=0)

#     # Use your existing viz function to display
#     sfm.viz_images([img1_filtered_kp, img2_filtered_kp], titles=["Image 1 Filtered", "Image 2 Filtered"])

#     print(f"Original SIFT keypoints: {len(kp1)}, {len(kp2)}")
#     print(f"After NMS: {len(kp1_nms)}, {len(kp2_nms)}")
#     print(f"Good matches: {len(good_matches)}")
#     print(f"After RANSAC (inliers): {len(matched_pts1_inliers)}")
#     print(f"After triangulation: {len(pts_3d)}")

#     R_pnp, t_pnp,T_pnp = sfm.poses_from_pnp(pts_3d, img2_valid, sfm.k)
#     print("PnP R: Rotates world points to align with camera\n", R_pnp)
#     print("PnP t: Translates world points to camera origin\n", t_pnp)
#     print("PnP Transformation:\n", T_pnp)

#     # 1. Initialize the tracker
#     track_manager = TrackManager()

#     # 2. Add triangulated points and keypoints to the tracker
#     track_manager.add_points(pts_3d, img1_valid, img2_valid, pose_idx1=0, pose_idx2=1)

#     # 3. Check the number of tracks
#     print(f"Tracks added: {len(track_manager.tracks)}")

#     track_manager.camera_poses.append((np.eye(3), np.zeros((3, 1))))
#     track_manager.camera_poses.append((R, t.reshape(3, 1)))
#     track_manager.camera_poses.append((R_pnp, t_pnp.reshape(3, 1)))

#     track_manager.plot_scene()


#     plt.show()