# Align depth map to coordinate of color map of the dataset FMDataset from [FastFusion](https://github.com/zhuzunjie17/FastFusion)

## Build

```
// At the root directory of this project
mkdir build && cd build
cmake .. && make -j
```

## Usage
`./preprocess "dataset_path" "output_path" "0 or 1 for visualization"`

### Example
```bash
// Align
./preprocess "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/" "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/aligned_depth/" 1

// Associate color and depth images
python3 associate.py <PATH_TO_FMDataset>/dorm1/dorm1_slow color aligned_depth
```

![](https://raw.githubusercontent.com/tin1254/FMDataset_preprocessing/master/result.gif)
