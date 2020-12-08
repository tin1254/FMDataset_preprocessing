import sys

if len(sys.argv) != 4:
    raise ValueError(
        "Usage: python3 associate.py path_to_dataset_dir rgb_images_dir depth_images_dir\n" +
        "python3 associate.py /home/tin/Datasets/FMDataset/dorm1/dorm1_slow color aligned")

file_path = sys.argv[1]
rgbd_path = sys.argv[2]+"/"
depth_path = sys.argv[3]+"/"


with open(file_path+"/TIMESTAMP.txt") as f:
    lines = f.readlines()

for line in lines:
    if line[0] != "#":
        line = line.replace("\n", "").split(",")
        timestamp = line[0]
        color = rgbd_path+line[1]
        depth = depth_path+line[2]
        print(" ".join([timestamp, color, timestamp, depth]))
