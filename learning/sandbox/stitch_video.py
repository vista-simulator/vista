import os
import sys
import argparse
import cv2
import numpy as np


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


class Stitcher():
    def __init__(self):
        pass

    def stitch(self, imgs, ratio=0.4, reproj_thresh=0.5):
        imgs = imgs[:2]#[::-1]
        assert len(imgs) == 2, 'Only support 2 images now'

        if True:
            img1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
            # We select only inlier points
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]

            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
            lines1 = lines1.reshape(-1,3)
            img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
            lines2 = lines2.reshape(-1,3)
            img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
            merged = np.concatenate([img3, img5], axis=1)
            cv2.imwrite('test.png', merged)
            import pdb; pdb.set_trace()

            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append([m])
            match_show = cv2.drawMatchesKnn(imgs[0],kp1,imgs[1],kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite('test.png', match_show)
            import pdb; pdb.set_trace()

            pt1 = np.float32([kp1[m[0].queryIdx].pt for m in good])
            pt2 = np.float32([kp2[m[0].trainIdx].pt for m in good])
            H, status = cv2.findHomography(pt1, pt2, cv2.RANSAC, reproj_thresh)
            result = cv2.warpPerspective(imgs[0], H, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0] + imgs[1].shape[0]))
            # result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]
            mask = np.all(result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] == 0, axis=2)
            result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = cv2.addWeighted(
                result[0:imgs[1].shape[0], 0:imgs[1].shape[1]], 0.5, imgs[1], 0.5, 0.0)
            result[0:imgs[1].shape[0], 0:imgs[1].shape[1]][mask] = imgs[1][mask]
            cv2.imwrite('test.png', result)
            import pdb; pdb.set_trace()
        elif False:
            img1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)

            orb = cv2.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1,None)
            kp2, des2 = orb.detectAndCompute(img2,None)
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 10 matches.
            good = matches[:10]
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite('test.png', img3)
            import pdb; pdb.set_trace()

            pt1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pt2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            H, status = cv2.findHomography(pt1, pt2, cv2.RANSAC, reproj_thresh)
            result = cv2.warpPerspective(imgs[0], H, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0] + imgs[1].shape[0]))
            result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]
            cv2.imwrite('test.png', result)
            import pdb; pdb.set_trace()
        else:
            kps = []
            descs = []
            for img in imgs:
                kp, desc = self.detect_and_describe(img)
                kps.append(kp)
                descs.append(desc)

            matches, good = self.match_keypoints(kps[0], kps[1], desc[0], desc[1], ratio)

            matches_show = cv2.drawMatchesKnn(imgs[0], kps[0], imgs[1], kps[1], good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # matches_show = self.draw_matches(imgs[0], kps[0], imgs[1], kps[1], matches)
            cv2.imwrite('test.png', matches_show)

            H, status = self.find_homography(kps[0], kps[1], matches, ratio, reproj_thresh)
            result = cv2.warpPerspective(imgs[0], H, (imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))
            result[0:imgs[1].shape[0], 0:imgs[1].shape[1]] = imgs[1]

            if False:
                cv2.imshow('Matches', matches_show)
                cv2.waitKey(0)

                cv2.imshow('Stitched', result)
                cv2.waitKey(0)

    def find_homography(self, kp1, kp2, rawMatches, ratio, reproj_thresh):
        matches = []
        for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        pt1 = np.float32([kp1[i].pt for (_, i) in matches])
        pt2 = np.float32([kp2[i].pt for (i, _) in matches])

        H, status = cv2.findHomography(pt1, pt2, cv2.RANSAC, reproj_thresh)

        return H, status

    def detect_and_describe(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # sift = cv2.SIFT_create()
        sift = cv2.ORB_create()
        # sift = cv2.BRISK_create()
        kp, desc = sift.detectAndCompute(gray, None)
        return kp, desc

    def match_keypoints(self, kp1, kp2, desc1, desc2, ratio):
        # bf = cv2.BFMatcher()
        bf = cv2.DescriptorMatcher_create('BruteForce')
        matches = bf.knnMatch(desc1, desc2, k=2)

        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append([m])
        return matches, good

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        # initialize the output visualization image
        (hA, wA) = img1.shape[:2]
        (hB, wB) = img2.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = img1
        vis[0:hB, wA:] = img2
        # loop over the matches
        for m in matches:
			# only process the match if the keypoint was successfully
			# matched
            trainIdx = m[0].trainIdx
            queryIdx = m[0].queryIdx
            # draw the match
            ptA = (int(kp1[queryIdx].pt[0]), int(kp1[queryIdx].pt[1]))
            ptB = (int(kp2[trainIdx].pt[0]) + wA, int(kp2[trainIdx].pt[1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
        return vis


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help='Path to a video')
    parser.add_argument('--img-dir', default=None, help='Directory of source images')
    parser.add_argument('--frame-idcs', type=int, nargs='+', default=[0, 1], help='Frame indices of images to be stitched')
    parser.add_argument('--inspect-video', action='store_true', default=False, help='Inspect instead of stitch video')
    parser.add_argument('--use-custom-stitcher', action='store_true', default=False, help='Use customized stitcher')
    parser.add_argument('--show-image', action='store_true', default=False, help='Show image')
    args = parser.parse_args()

    if args.video is not None:
        video_path = os.path.expanduser(args.video)
        assert os.path.exists(video_path)
    elif args.img_dir is not None:
        img_dir_path = os.path.expanduser(args.img_dir)
        assert os.path.isdir(img_dir_path)
    else:
        print('Either --video or --img-dir should be specified')
        sys.exit()

    # obtain images to be stitched from video 
    src_imgs = []
    if args.video is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Fail to open video {}'.format(video_path))
            sys.exit()
        
        current_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if False: # rectify image
                mtx = np.array([
                    [262.614237, 0.000000, 500.320088],
                    [0.000000, 261.384971, 285.845478],
                    [0.000000, 0.000000, 1.000000],
                ])
                dist = np.array([0.007158, -0.001713, -0.000778, -0.000663, 0.000000])
                img_h, img_w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w,img_h), 0, (img_w,img_h))
                if False:
                    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                else:
                    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (img_w,img_h), 5)
                    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                # compare = np.concatenate([frame, dst], axis=0)
                # cv2.imwrite('test.png', compare)
                # import pdb; pdb.set_trace()
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                frame = dst

            if not (args.inspect_video or current_frame_idx <= np.max(args.frame_idcs)):
                break

            if ret:
                if current_frame_idx in args.frame_idcs:
                    src_imgs.append(frame)

                if args.show_image:
                    frame_show = cv2.putText(frame.copy(), '{:04d}'.format(current_frame_idx), (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    cv2.imshow('Video', frame_show)

                    key = cv2.waitKey(20)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        cv2.waitKey(-1) # wait until any key is pressed

                current_frame_idx += 1
            else:
                print('Video capture ends with failing to read new frame')
                break
        cap.release()
        cv2.destroyAllWindows()

        # check source images
        if args.show_image:
            merged_img = np.concatenate(src_imgs, axis=1)
            cv2.imshow('Source Images', merged_img)
            cv2.waitKey(0)
    elif args.img_dir is not None:
        for fpath in os.listdir(args.img_dir):
            if os.path.splitext(fpath)[-1].lower() in ['.jpg', '.png', '.jpeg']:
                fpath = os.path.join(args.img_dir, fpath)
                frame = cv2.imread(fpath)
                src_imgs.append(frame)
    else:
        raise ValueError('You should not be here')

    # perform stitching
    if args.use_custom_stitcher:
        stitcher = Stitcher()
        stitcher.stitch(src_imgs)
    else:
        stitcher = cv2.Stitcher_create()
        # stitcher.setPanoConfidenceThresh(0.1)
        # src_imgs = [img[200:400,380:580] for img in src_imgs]
        status, stitched = stitcher.stitch(src_imgs)
        
        # status = stitcher.estimateTransform(src_imgs)
        # status, stitched = stitcher.composePanorama(src_imgs)

        if status == 0:
            if args.show_image:
                cv2.imshow('Stitched', stitched)
                cv2.waitKey(0)
            cv2.imwrite('test.png', stitched)
        else:
            print('Fail to stitch images ({})'.format(status))

    # post processing

    # save stitched image


if __name__ == '__main__':
    main()
