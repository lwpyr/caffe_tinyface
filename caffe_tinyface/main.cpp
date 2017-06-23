

#include "header.h"

int main() {
	string protoPath = "C:/Users/LI_Wen/Desktop/tinyface/tiny-caffe/test.prototxt";
	string modelPath = "C:/Users/LI_Wen/Desktop/tinyface/tiny-caffe/matconvnet2caffe.caffemodel";
	string clusterPath = "C:/Users/LI_Wen/Desktop/tinyface/cluster.dat";
	string filePath = "C:/Users/LI_Wen/Desktop/tinyface/tiny-caffe/demo/data";

	tinyFaceDetector solver(protoPath, modelPath, clusterPath, filePath);
	solver.run(tinyFaceDetector::MODEL::GPU);
	return 0;
}