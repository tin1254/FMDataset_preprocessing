# Align depth map of [FastFusion](https://github.com/zhuzunjie17/FastFusion)

Align depth image to color image of the dataset FMDataset from [FastFusion](https://github.com/zhuzunjie17/FastFusion)


## Usage
``` cpp
// At the root directory of this project
mkdir build && cd build
cmake .. && make -j
// ./preprocess "dataset_path" "output_path" "0 or 1 for visualization"
./preprocess "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/" "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/aligned/" 1
```

![](https://raw.githubusercontent.com/tin1254/FMDataset_preprocessing/master/result.gif)