from typing import Union
import cv2
from tqdm import tqdm
from os import mkdir
import numpy as np

def extract_frames(filename: str, file_dst: str, frame_per_second: int, start_time: Union[list, int]=None, end_time: Union[list, int]=None):
    """
    Extract the frames from video

    Args:
        cat_code: the category code of the video
        filename: the name of video
        file_dst: the path of destination folder
        frame_per_second: extract frames per second
        start_time: the list of start time
        end_time: the list of end time
    
    Returns:
        None, create an image for each selected frame

    Raises:
        ValueError: Error! length of start time list must be equal to length of end time list
        ValueError: Error! time in end time list exceeds total seconds
        ValueError: Error! start time is bigger than end time
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = total_frames / fps
    
    if isinstance(start_time, list) and isinstance(end_time, list) and len(start_time) != len(end_time):
        raise ValueError("Error! length of start time list must be equal to length of end time list")
    
    if isinstance(start_time, int):
        start_time = [start_time]
    if isinstance(end_time, int):
        end_time = [end_time]
    if not start_time:
        start_time = [0]
    if not end_time:
        end_time = [total_seconds]
    
    check_list = [start <= end for start, end in zip(start_time, end_time)]
    if max(end_time) > total_seconds:
        raise ValueError("Error! time in end time list exceeds total seconds")
    elif not min(check_list):
        raise ValueError("Error! start time is bigger than end time")

    # Initialization
    frameCount = 0
    
    start_frame_list = [round(time * fps) for time in start_time]
    end_frame_list = [round(time * fps) for time in end_time]
    duration_of_frame = round(fps / frame_per_second)

    print("Original FPS: ", fps)
    print("Extracted FPS: ", frame_per_second)
    print("Number of frames: ", total_frames)
    print("Total seconds: ", total_seconds)
    print("Extracting...")

    mkdir(file_dst)
    # Extract frames from file
    for start, end in zip(start_frame_list, end_frame_list):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        pbar = tqdm(total=end-start)
        while start < end:
            ret, image = cap.read()
            if not ret:
                break
            cv2.imwrite(file_dst + ("/{:0>6d}".format(frameCount)) + '.jpg', image)
            start += duration_of_frame
            pbar.update(duration_of_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
            frameCount += 1
        pbar.close()
    print("Finish")
    print()

