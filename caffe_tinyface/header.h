#ifndef HEADER_TINYFACE
#define HEADER_TINYFACE

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <set>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"    //caffe static library
#ifdef _DEBUG
#include <caffe_force_link-d.h> //force to link all symbols in caffe-d.lib to this project
#else
#include <caffe_force_link.h> //force to link all symbols in caffe.lib to this project
#endif // DEBUG

using namespace caffe;

class tinyFaceDetector {
private:
	string protoPath;
	string modelPath;
	string clusterPath;
	string filePath;
	string savePath;

	float thresh = 0.5; // parameters initialization
	float nmsthresh = 0.1;
	int index = 0;
	int max_faceindex = 0;
	int max_facenum = 0;

	typedef pair<float, int> couple;

	static void drawRect(cv::Mat& img, int xL, int yL, int xR, int yR);
	static void _nms2(float f, cv::Mat& m);
	static cv::Mat _nms1(float f, cv::Mat& m, vector<couple>&I, int mode);
	static cv::Mat nms(const cv::Mat& boxList, float thresh);
	static cv::Mat mergeCols(const cv::Mat& A, const cv::Mat& B);
	static void cvVectorSetInRange(cv::Mat& im, float min, float max);
	static void cvMat2caffeBlobData(const cv::Mat& image, float* const blobData);
	static vector<string> getFileName(std::string& pathName);
public:
	enum MODEL {
		GPU, CPU
	};

	tinyFaceDetector(string& protoP, string& modelP, string&clusterP, string& fileP) {
		protoPath = protoP;
		modelPath = modelP;
		clusterPath = clusterP;
		filePath = fileP;
		savePath = filePath + "/result";
	}
	~tinyFaceDetector() {}
	void run(MODEL m);

};

#endif // HEADER_TINYFACE

