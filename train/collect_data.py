"""
Step 1. Collect data.
"""

import cv2


import os
import datetime


from funcs.face_haar import haarFaceDetect


def create_data_folder():
    # create folders
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    if not os.path.exists("./dataset/train"):
        os.mkdir("./dataset/train")

    if not os.path.exists("./dataset/val"):
        os.mkdir("./dataset/val")

    na_train_dir = "./dataset/train/0_natural"
    an_train_dir = "./dataset/train/1_angry"
    di_train_dir = "./dataset/train/2_disgust"
    fe_train_dir = "./dataset/train/3_fear"
    ha_train_dir = "./dataset/train/4_happy"
    sa_train_dir = "./dataset/train/5_sad"
    su_train_dir = "./dataset/train/6_surprise"

    fer_dirs = [na_train_dir, an_train_dir, di_train_dir, fe_train_dir, ha_train_dir, sa_train_dir, su_train_dir]

    for fer_dir in fer_dirs:
        if not os.path.exists(fer_dir):
            os.mkdir(fer_dir)

    na_val_dir = "./dataset/val/0_natural"
    an_val_dir = "./dataset/val/1_angry"
    di_val_dir = "./dataset/val/2_disgust"
    fe_val_dir = "./dataset/val/3_fear"
    ha_val_dir = "./dataset/val/4_happy"
    sa_val_dir = "./dataset/val/5_sad"
    su_val_dir = "./dataset/val/6_surprise"

    fer_dirs = [na_val_dir, an_val_dir, di_val_dir, fe_val_dir, ha_val_dir, sa_val_dir, su_val_dir]

    for fer_dir in fer_dirs:
        if not os.path.exists(fer_dir):
            os.mkdir(fer_dir)

    return


def collect_train_data(exp="natural", save=False):

    """
    natural / angry / disgust / fear / happy / sad / surprise
    :return:
    """

    # video stream
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    faceDetModel = haarFaceDetect()

    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No 2D video signal. Please change the Camera ID, or check the webcam.")
            break

        # face detection
        frame = cv2.flip(frame, 1)
        haarfile = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")

        face_exist, frame, face_frame = faceDetModel.run(frame, haar_file=haarfile)

        # beautiful
        # frame = 1.1 * frame
        # frame[frame > 255] = 255
        # frame = frame.astype(np.uint8)

        cv2.imshow("video cam", frame)
        k = cv2.waitKey(1)

        if face_exist:

            # save face image
            timenow = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S_')
            savename = timenow + "{:d}.png".format(count)

            # save image
            if save:
                save_dir = None
                if count % 5 == 0:
                    na_dir = "./dataset/val/0_natural/"
                    an_dir = "./dataset/val/1_angry/"
                    di_dir = "./dataset/val/2_disgust/"
                    fe_dir = "./dataset/val/3_fear/"
                    ha_dir = "./dataset/val/4_happy/"
                    sa_dir = "./dataset/val/5_sad/"
                    su_dir = "./dataset/val/6_surprise/"
                else:
                    na_dir = "./dataset/train/0_natural/"
                    an_dir = "./dataset/train/1_angry/"
                    di_dir = "./dataset/train/2_disgust/"
                    fe_dir = "./dataset/train/3_fear/"
                    ha_dir = "./dataset/train/4_happy/"
                    sa_dir = "./dataset/train/5_sad/"
                    su_dir = "./dataset/train/6_surprise/"

                if exp == "natural":
                    save_dir = na_dir

                if exp == "angry":
                    save_dir = an_dir

                if exp == "disgust":
                    save_dir = di_dir

                if exp == "fear":
                    save_dir = fe_dir

                if exp == "happy":
                    save_dir = ha_dir

                if exp == "sad":
                    save_dir = sa_dir

                if exp == "surprise":
                    save_dir = su_dir

                print("saving {:s} ...".format(savename))
                cv2.imwrite(save_dir + savename, face_frame)

                count += 1

        if k == 27:
            break

    return

