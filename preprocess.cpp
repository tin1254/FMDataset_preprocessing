#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>

/**
 * This code converts FMDataset into TUM_RGBD and TUM_VI format.
 * FMDataset: https://github.com/zhuzunjie17/FastFusion
 * TUM_VI: https://vision.in.tum.de/data/datasets/visual-inertial-dataset
 * TUM_RGBD: https://vision.in.tum.de/data/datasets/rgbd-dataset
 *
 * Preprocessing steps:
 *
 * In TUM_RGBD dataset, the color and depth images are aligned. However,
 * in FMDataset, color and depth are not in the same coordindate.
 * We will first align depth to color and save depth again as png.
 * IMU format is the same as TUM_VI, so we leave it as it is.
 *
 *
 *  Depth:
 *  1. Unproject distorted pixel onto image plane (Unproject the coordinate of
 *the color image)
 *  2. Apply transformation color->depth
 *  3. Project the points into distorted pixel (depth image)
 *  4. Bilinear interpolate the value with the depth image at the projected
 *pixel location
 *  5. Fill in the interpolated value into the corresponding location of the
 *pixel at step 1
 *
 *
 *  Color (TODO):
 *  Verify if the result of undistortPoints in opencv is the same as Inverse
 *  Brown-Conrady in realsense
 *
 *    Reference:
 *    https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0#intrinsic-camera-parameters
 *    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga55c716492470bfe86b0ee9bf3a1f0f7e
 **/

const std::string DATASET_PATH =
    "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/";
const std::string ALIGNED_DEPTH_PATH =
    "/home/tin/Datasets/FMDataset/dorm1/dorm1_slow/aligned/";
const float DEPTH_SCALE = 1000;
const size_t IMAGE_WIDTH = 640, IMAGE_HEIGHT = 480;
const float eps = 1e-16;

float R_c_d_raw[3][3] = {{0.999980211, -0.00069964811, -0.0062491186},
                         {0.000735448, 0.999983311, 0.00572841847},
                         {0.006245006, -0.00573290139, 0.999964058}},
      t_c_d_raw[3] = {-0.057460, -0.001073, -0.002205},
      K_d_raw[3][3] = {{583, 0, 325}, {0, 583, 240}, {0, 0, 1}},
      K_c_raw[3][3] = {{608, 0, 331}, {0, 608, 246}, {0, 0, 1}};

void LoadImages(const std::string &strFile,
                std::vector<std::string> &vstrRGBImageFilenames,
                std::vector<std::string> &vstrDepthImageFilenames,
                std::vector<int> &vTimestamps) {
  std::ifstream f;
  f.open(strFile.c_str());

  std::string s;
  getline(f, s);  // skip the first line
  while (!f.eof()) {
    getline(f, s);
    std::stringstream ss(s);

    std::string time, rgb, depth;
    getline(ss, time, ',');
    if (time.empty()) break;
    vTimestamps.push_back(std::stoi(time));
    getline(ss, rgb, ',');
    vstrRGBImageFilenames.push_back(rgb);
    getline(ss, depth, ',');
    vstrDepthImageFilenames.push_back(depth);
  }
}

// Reference:
// https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L46
void UnprojectDistortedPixelToPoint(cv::Mat &point, const cv::Mat &K,
                                    const std::vector<float> &coeffs,
                                    const size_t pixel_x, const size_t pixel_y,
                                    float depth) {
  const auto &fx = K.at<float>(0, 0);
  const auto &fy = K.at<float>(1, 1);
  const auto &ppx = K.at<float>(0, 2);
  const auto &ppy = K.at<float>(1, 2);

  float x = (pixel_x - ppx) / fx;
  float y = (pixel_y - ppy) / fy;

  float r2 = x * x + y * y;
  float f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2;
  float ux = x * f + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x);
  float uy = y * f + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y);
  x = ux;
  y = uy;

  point = (cv::Mat_<float>(3, 1) << depth * x, depth * y, depth);
}

// Reference:
// https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L15
void ProjectPointToDistortedPixel(const cv::Mat &point, const cv::Mat &K,
                                  const std::vector<float> &coeffs,
                                  cv::Point2f &pixel) {
  const auto &px = point.at<float>(0);
  const auto &py = point.at<float>(1);
  const auto &pz = point.at<float>(2);

  const auto &fx = K.at<float>(0, 0);
  const auto &fy = K.at<float>(1, 1);
  const auto &ppx = K.at<float>(0, 2);
  const auto &ppy = K.at<float>(1, 2);

  float x = px / pz, y = py / pz;

  // float r2 = x * x + y * y;
  // float f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 *
  // r2; x *= f; y *= f; float dx = x + 2 * coeffs[2] * x * y + coeffs[3] * (r2
  // + 2 * x * x); float dy = y + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 *
  // y * y); x = dx; y = dy;

  pixel = cv::Point2f(x * fx + ppx, y * fy + ppy);
}

void UnprojectPixel(const size_t &cols, const size_t &rows, const cv::Mat &K,
                    const std::vector<float> &dist_coeffs,
                    std::vector<cv::Mat> &vp3D,
                    std::vector<std::pair<size_t, size_t>> &vpPixelIdx) {
  vp3D.clear();
  vpPixelIdx.clear();
  vp3D.reserve(cols * rows);
  vpPixelIdx.reserve(cols * rows);

  for (size_t c = 0; c < cols; ++c) {
    for (size_t r = 0; r < rows; ++r) {
      cv::Mat p3D;
      UnprojectDistortedPixelToPoint(p3D, K, dist_coeffs, c, r, 1);
      vp3D.push_back(p3D);
      vpPixelIdx.emplace_back(r, c);
    }
  }
}

void ProjectPoint(const std::vector<cv::Mat> &vp3D, const cv::Mat &K,
                  const std::vector<float> &dist_coeffs,
                  std::vector<cv::Point2f> &vp2D) {
  vp2D.clear();
  vp2D.reserve(vp3D.size());
  for (const auto &p3D : vp3D) {
    cv::Point2f p2D;
    ProjectPointToDistortedPixel(p3D, K, dist_coeffs, p2D);
    vp2D.push_back(p2D);
  }
}

// Reference:
// https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
ushort InterpolatePixelValue(const cv::Mat &img, const cv::Point2f &p) {
  if (!(0 < p.x || p.x >= img.size[0]) && !(0 < p.y || p.y >= img.size[1]))
    return 0.;

  cv::Mat patch;
  cv::getRectSubPix(img, cv::Size(1, 1), p, patch);
  return static_cast<ushort>(patch.at<float>(0));
}

void Transform3DPoints(const cv::Mat &R, const cv::Mat &t,
                       std::vector<cv::Mat> &vp3D) {
  for (auto &p : vp3D) p = R * p + t;
}

void AlignDepth(const cv::Mat &imD_original, cv::Mat &imD_aligned,
                const cv::Mat &K_c, const cv::Mat &K_d, const float depth_scale,
                const std::vector<float> &dist_coeffs, const cv::Mat &R,
                const cv::Mat &t) {
  std::vector<cv::Mat> vp3D;
  std::vector<std::pair<size_t, size_t>> vpPixelIdx;
  UnprojectPixel(IMAGE_WIDTH, IMAGE_HEIGHT, K_c, dist_coeffs, vp3D, vpPixelIdx);
  Transform3DPoints(R, t, vp3D);

  std::vector<cv::Point2f> vp2D;
  ProjectPoint(vp3D, K_d, dist_coeffs, vp2D);

  cv::Mat _imD;
  imD_original.convertTo(_imD, CV_32F);
  imD_aligned = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16UC1, cv::Scalar(0));

  for (size_t i = 0; i < vp3D.size(); ++i) {
    const size_t &r = vpPixelIdx[i].first;
    const size_t &c = vpPixelIdx[i].second;
    if (r >= 0 && r < imD_original.size[1] && c >= 0 &&
        c < imD_original.size[01]) {
      imD_aligned.at<ushort>(r, c) = InterpolatePixelValue(_imD, vp2D[i]);
    }
  }
}

int main(int argc, char **argv) {
  const cv::Mat R_c_d(3, 3, CV_32F, &R_c_d_raw),
      t_c_d(3, 1, CV_32F, &t_c_d_raw), K_d(3, 3, CV_32F, &K_d_raw),
      K_c(3, 3, CV_32F, &K_c_raw);
  const std::vector<float> dist_coeffs = {0.0644, -0.114, 0.00127, 0.00203, 0};

  std::vector<std::string> vstrRGBImageFilenames, vstrDepthImageFilenames;
  std::vector<int> vTimestamps;
  LoadImages(DATASET_PATH + "TIMESTAMP.txt", vstrRGBImageFilenames,
             vstrDepthImageFilenames, vTimestamps);

  cv::Mat R = R_c_d.t();
  cv::Mat t = -R * t_c_d;

  for (const auto &strName : vstrDepthImageFilenames) {
    cv::Mat imD, imD_aligned;
    imD = cv::imread(DATASET_PATH + "depth/" + strName,
                     CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    AlignDepth(imD, imD_aligned, K_c, K_d, DEPTH_SCALE, dist_coeffs, R,
               t);
    std::cout << ALIGNED_DEPTH_PATH + strName << std::endl;
    cv::imwrite(ALIGNED_DEPTH_PATH + strName, imD_aligned);
  }

  return 0;
}
