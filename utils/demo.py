import numpy as np
import cv2
import tqdm
import time
from torch.cuda.amp import autocast
import torch
import shutil

from .utils import *


def convert_img_to_video(image_dir, video_dir, img_size, fps=20):
    image_list = os.listdir(image_dir)
    image_list.sort(key=lambda x: int(x.split('.')[0]))
    # img_size = (590, 960)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in image_list:
        frame = cv2.imread(os.path.join(image_dir, i))
        video_writer.write(frame)
        # cv2.imshow('rr', frame)
        # cv2.waitKey(20)

    video_writer.release()
    print("Finish writing video: ", video_dir)


def _frame_from_video(video, args):
    while video.isOpened():
        success, frame = video.read()

        if success:
            yield frame  # preprocessed
        else:
            break


def process_predictions(data, model, frame, args):
    # read data
    batch = []
    for image in data:
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        batch.append(image)
    data.pop(0)
    batch = torch.from_numpy(np.concatenate(batch, axis=0)).unsqueeze(0)
    images = batch.cuda()
    input_image = images.detach()
    target_image = images.detach()
    # inference
    s_ = time.time()
    with autocast():
        #output, loss, logit = model.forward(input_image, gt=target_image, label=None, train=False)
        # score = logit.view(-1).item()

        output, loss = model.forward(input_image, gt=target_image, label=None, train=False)
        loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean()
    print("INFERENCE SINGLE FRAME TIME COST: {}s".format(time.time() - s_))
    score = psnr(loss.item())
    score = (score - 3.5880800460751594) / (25.215500099559964 - 3.5880800460751594)

    if score < 0.5:
        res = "Anomaly," + str(format(score, '.5f'))
    else:
        res = "Normal," + str(format(score, '.5f'))

    output = output[:, -3:]
    predict = 255 * (output + 1.) / 2
    predict = predict.squeeze(0).byte().cpu().numpy().transpose((1,2,0))
    predict = cv2.resize(predict, (frame.shape[1], frame.shape[0]))
    predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
    # print("The category of frame is {}, psnr is {}, time cost: {}".format(res, score, time.time()-start_time))
    return res, predict, frame


def run_on_video(cam, model, args):
    frames = _frame_from_video(cam, args)
    data = []
    # args.h, args.w = 64, 64
    frame_length = args.t_length
    frame_interval = args.interval
    for i, frame in enumerate(frames):
        if i % frame_interval != 0:  # interval
            continue

        # preprocess
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sp = frame.shape
        real_height, read_weight = sp[0], sp[1]
        if int(real_height) != args.h or int(read_weight) != args.w:
            image = cv2.resize(image, (args.w, args.h)).astype(dtype=np.float32)
        image = 2 * (image / 255) - 1.0

        assert len(data) <= frame_length, "fragment length must less than 5."

        if len(data) < frame_length - 1:
            data.append(image)
        elif len(data) == frame_length - 1:
            data.append(image)
            yield process_predictions(data, model, frame, args)
        else:
            print("DATA LENGTH ERROR")


def demo(model, args):
    model.eval()

    demo_type, demo_dir = args.demo.split('|')
    if demo_type == "webcam":
        cam = cv2.VideoCapture(0)
        for res, predict, frame in tqdm.tqdm(run_on_video(cam, model, args)):
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            frame = np.concatenate((frame, predict), axis=1)
            frame = cv2.putText(frame, res.upper(), (80, 80), font, 2, (255,215,0), 2)
            # cv2.imwrite(os.path.join("/home/rail/Documents/airs_scene/frames", str(int(time.time()))+".jpg"), vis)
            WINDOW_NAME = "Anomaly Detection"
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
    elif demo_type == "video":
        cam = cv2.VideoCapture(demo_dir)
        image_dir = demo_dir.split('.')[0]
        image_size = (1280, 1024)  # w h
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        os.makedirs(image_dir, exist_ok=True)

        counter = 0
        for res, predict, frame in tqdm.tqdm(run_on_video(cam, model, args)):
            if counter == 0:
                sp = frame.shape
                image_size = (sp[1]*2, sp[0])

            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            frame = np.concatenate((frame, predict), axis=1)
            frame = cv2.resize(frame, image_size)
            frame = cv2.putText(frame, res.upper(), (60, 60), font, 3, (255,215,0), 3)
            cv2.imwrite(os.path.join(image_dir, str(counter)+".jpg"), frame)
            counter += 1

        print("TOTAL SAVED FRAMES NUM IS {}".format(counter))
        cam.release()
        convert_img_to_video(image_dir, image_dir+'_demo.avi', image_size, fps=30)

    else:
        print("MUST SPECIFY A DEMO TYPE.")

