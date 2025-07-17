#!/usr/bin/env python3

import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import operator

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
        '''
            Load images and store into an array 
        '''
        print(f"----Loading the images from {self.path}----")
        
        imgs= [cv2.imread(os.path.join(self.path,file)) for file in sorted(os.listdir(self.path))]
        
        # comment out the pre_process out put to check the changes in the processed image
        imgs = [self.pre_process(img, create_mask = True) for img in imgs]

        self.viz_images(imgs)

        return imgs
    
    def set_intrinsics(self, imgs):     # to set the intrinsics of the camera
        if len(imgs[0].shape) == 2:
            self.height, self.width = imgs[0].shape
        else:
            self.height, self.width, _ = imgs[0].shape

        focal_length = 1.2 * max(self.width, self.height)  # heuristic if fx, fy not known

        self.k = np.array([
            [focal_length, 0, self.width / 2],
            [0, focal_length, self.height / 2],
            [0, 0, 1]
        ])

        if self.viz:
            print(f"----Intrinsic matrix K set to----")
            print(self.k)

    def match_marker(self, imgd1, imgd2, img1=None, img2=None, kp1=None, kp2=None,
                 min_pts=50, max_ratio=0.6, relax_step=0.05, show_keypoints=False): 
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

        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(imgd1,imgd2,k=2)

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
                cv2.circle(binary_mask, tuple(point), 2, 255, -1)
        
        keypoints = np.array(keypoints)[nms_mask]
        descriptors = np.array(descriptors)[nms_mask]

        return nms_mask, keypoints, descriptors
    
    def essential_matrix(self, matched_pts1, matched_pts2):

        """
        parameters: 
        -   matched points

        return: 
        - E:    Essential matrix
        -   matrix
        -   Translation vector
        -   RANSAC inlier mask
        """

        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, cameraMatrix=self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # _, R, t, _ = cv2.recoverPose(E, matched_pts1, matched_pts2, self.k)

        if self.viz:
            print("Essential Matrix E:\n", E)
        
        
        return E, mask
    
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
            for pt1, pt2 in zip(img1_inlier_pts.T, img2_inlier_pts.T):
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
    
    def triangulate_pts(self, pose_c1, pose_c2, filtered_src_pts, filtered_dst_pts):
        # filtered_src_pts, filtered_dst_pts: shape (N, 2)

        pts_4d_hom = cv2.triangulatePoints(pose_c1, pose_c2, filtered_src_pts.T, filtered_dst_pts.T)
        pts_3d = pts_4d_hom[:3, :] / pts_4d_hom[3, :]

        # pts_3d shape: (3, N), transpose to (N, 3)
        pts_3d = pts_3d.T

        print(f"shape of pts_3d before masking = {pts_3d.shape}")

        # Create mask based on positive depth
        mask = pts_3d[:, 2] > 0

        # Apply mask to points in (N, 2) format
        img1_pts = filtered_src_pts[mask]
        img2_pts = filtered_dst_pts[mask]
        pts_3d = pts_3d[mask]

        print(f"shape of pts_3d after masking = {pts_3d.shape}")
        print(f"\nTriangulated {pts_3d.shape[0]} 3D points")


        return pts_3d, img1_pts, img2_pts
    
    
    


    

    





  





        
    

    
    







if __name__ == "__main__":
    
    path_to_images = "/home/iso/Documents/MATLAB/GIT/Structure_SFM_Motion/buddha_images"

    sfm = SfmHelpers(path=path_to_images, single_img=False, viz=True)

    images = sfm.get_images()
    sfm.set_intrinsics(images)


    img1, img2 = images[0], images[1]

    

    # Extract SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Non-Max Suppression
    _, kp1_nms, des1_nms = sfm.non_max_suppression(kp1, des1)
    _, kp2_nms, des2_nms = sfm.non_max_suppression(kp2, des2)



    

    
    #Match features and visualize keypoints (inside match_marker)
    matches, good_matches = sfm.match_marker(des1_nms, des2_nms,
                                            img1=img1, img2=img2,
                                            kp1=kp1_nms, kp2=kp2_nms,
                                            min_pts=50, max_ratio=0.6,
                                            relax_step=0.05,
                                            show_keypoints=True)

    img1_nms_viz = cv2.drawKeypoints(img1, kp1_nms, None, color=(255, 0, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_nms_viz = cv2.drawKeypoints(img2, kp2_nms, None, color=(0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 5. Visualize Keypoints after NMS
    img1_nms_viz = cv2.drawKeypoints(img1, kp1_nms, None, color=(255, 0, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_nms_viz = cv2.drawKeypoints(img2, kp2_nms, None, color=(0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sfm.viz_images([img1_nms_viz, img2_nms_viz], titles=["Image 1 (NMS)", "Image 2 (NMS)"])



    # 6. Draw good matches
    match_img = cv2.drawMatches(img1, kp1_nms, img2, kp2_nms, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sfm.viz_images([match_img], titles=["Matched Features"])

    # 7. Extract matched locations
    matched_pts1 = np.float32([kp1_nms[m.queryIdx].pt for m in good_matches])
    matched_pts2 = np.float32([kp2_nms[m.trainIdx].pt for m in good_matches])

    # 8. Compute Essential Matrix and Recover Pose
    #E, R, t, inlier_mask = sfm.essential_matrix(matched_pts1, matched_pts2)
    E, inlier_mask = sfm.essential_matrix(matched_pts1, matched_pts2)
    
    # Filter inlier points using the inlier_mask
    matched_pts1_inliers = matched_pts1[inlier_mask.ravel() == 1]
    matched_pts2_inliers = matched_pts2[inlier_mask.ravel() == 1]

    # Use manual pose estimation
    R, t= sfm.posesFromE(E, matched_pts1_inliers.T, matched_pts2_inliers.T)


    

    print("\nManual posesFromE Output:")
    print("Rotation R\n", R)
    print("Translation t:\n", t)

    # 9. Build projection matrices from recovered pose
    pose_c1 = sfm.k @ np.hstack((np.eye(3), np.zeros((3, 1))))
    pose_c2 = sfm.k @ np.hstack((R, t.reshape(3, 1)))

    # 10. Transpose inlier points to shape (2, N)
    # pts1_norm = matched_pts1_inliers.T.astype(np.float32)  # shape (2, N)
    # pts2_norm = matched_pts2_inliers.T.astype(np.float32)  # shape (2, N)

    # pass points in (N, 2)  
    pts_3d, img1_valid, img2_valid = sfm.triangulate_pts(pose_c1, pose_c2, matched_pts1_inliers, matched_pts2_inliers)

    

    # Convert filtered points (N, 2) to cv2.KeyPoint list
    kp1_filtered = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=3) for pt in img1_valid]
    kp2_filtered = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=3) for pt in img2_valid]

    # Draw keypoints on original images (copy to avoid modifying originals)
    img1_filtered_kp = cv2.drawKeypoints(img1.copy(), kp1_filtered, None, color=(0, 255, 0), flags=0)
    img2_filtered_kp = cv2.drawKeypoints(img2.copy(), kp2_filtered, None, color=(0, 255, 0), flags=0)

    # Use your existing viz function to display
    sfm.viz_images([img1_filtered_kp, img2_filtered_kp], titles=["Image 1 Filtered", "Image 2 Filtered"])

    print(f"Original SIFT keypoints: {len(kp1)}, {len(kp2)}")
    print(f"After NMS: {len(kp1_nms)}, {len(kp2_nms)}")
    print(f"Good matches: {len(good_matches)}")
    print(f"After RANSAC (inliers): {len(matched_pts1_inliers)}")
    print(f"After triangulation: {len(pts_3d)}")


    plt.show()
  
        
