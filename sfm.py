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

    def pre_process(self, img, mask=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if mask is not None:
            enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        return enhanced

    def get_images(self):
        '''
            Load images and store into an array 
        '''
        print(f"----Loading the images from {self.path}----")
        
        imgs= [cv2.imread(os.path.join(self.path,file)) for file in sorted(os.listdir(self.path))]
        
        # comment out the pre_process out put to check the changes in the processed image
        imgs = [self.pre_process(img) for img in imgs]

        self.viz_images(imgs)

        return imgs
    
    def set_intrinsics(self, imgs):     # to set the intrinsics of the camera
        if len(imgs[0].shape) == 2:
            self.height, self.width = imgs[0].shape
        else:
            self.height, self.width, _ = imgs[0].shape

        focal_length = 1.2 * max(self.width, self.height)  # heuristic if fx, fy not known

        self.K = np.array([
            [focal_length, 0, self.width / 2],
            [0, focal_length, self.height / 2],
            [0, 0, 1]
        ])

        if self.viz:
            print(f"----Intrinsic matrix K set to----")
            print(self.K)

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
    
    def essential_matrix(self, matched_pt1, matched_pt2):

        """
        parameters: 
        -   matched points

        return: 
        - E:    Essential matrix
        -   matrix
        -   Translation vector
        -   RANSAC inlier mask
        """

        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        _, R, t, _ = cv2.recoverPose(E, matched_pts1, matched_pts2, self.K)

        if self.viz:
            print("Essential Matrix E:\n", E)
            print("Recovered Rotation R:\n", R)
            print("Recovered Translation t:\n", t)
        
        return E, R, t, mask
    





  





        
    

    
    







if __name__ == "__main__":
    
    path_to_images = "/home/iso/Documents/MATLAB/GIT/Structure_SFM_Motion/buddha_images"

    sfm = SfmHelpers(path=path_to_images, single_img=False, viz=True)

    images = sfm.get_images()
    sfm.set_intrinsics(images)


    img1, img2 = images[0], images[1]

    

    # Extract SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, imgd1 = sift.detectAndCompute(img1, None)
    kp2, imgd2 = sift.detectAndCompute(img2, None)

    

    #Match features and visualize keypoints (inside match_marker)
    matches, good_matches = sfm.match_marker(imgd1, imgd2,
                                            img1=img1, img2=img2,
                                            kp1=kp1, kp2=kp2,
                                            min_pts=50, max_ratio=0.6,
                                            relax_step=0.05,
                                            show_keypoints=True)
    # Draw matched image
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    _, kp1_nms, imgd1_nms = sfm.non_max_suppression(kp1, imgd1)

    # Optionally, apply NMS to img2 as well
    _, kp2_nms, imgd2_nms = sfm.non_max_suppression(kp2, imgd2)

    img1_nms_viz = cv2.drawKeypoints(img1, kp1_nms, None, color=(255, 0, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_nms_viz = cv2.drawKeypoints(img2, kp2_nms, None, color=(0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Visualize using class method
    sfm.viz_images([match_img], titles=["Matched Features"])

    sfm.viz_images([img1_nms_viz, img2_nms_viz], titles=["Image 1 (NMS)", "Image 2 (NMS)"])

    # Convert matched keypoints to float arrays
    matched_pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    matched_pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    E, R, t, inlier_mask = sfm.essential_matrix(matched_pts1, matched_pts2)


    plt.show()
  
        
