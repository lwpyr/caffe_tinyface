#include "header.h"

void tinyFaceDetector::drawRect(cv::Mat& img, int xL, int yL, int xR, int yR) {
	for (int i = xL; i <= xR; i++) {
		img.at<cv::Vec3f>(yL, i)[0] = 0;
		img.at<cv::Vec3f>(yL, i)[1] = 255;
		img.at<cv::Vec3f>(yL, i)[2] = 255;
		img.at<cv::Vec3f>(yR, i)[0] = 0;
		img.at<cv::Vec3f>(yR, i)[1] = 255;
		img.at<cv::Vec3f>(yR, i)[2] = 255;
	}
	for (int i = yL; i <= yR; i++) {
		img.at<cv::Vec3f>(i, xL)[0] = 0;
		img.at<cv::Vec3f>(i, xL)[1] = 255;
		img.at<cv::Vec3f>(i, xL)[2] = 255;
		img.at<cv::Vec3f>(i, xR)[0] = 0;
		img.at<cv::Vec3f>(i, xR)[1] = 255;
		img.at<cv::Vec3f>(i, xR)[2] = 255;
	}
}

typedef pair<float, int> couple;
cv::Mat tinyFaceDetector::_nms1(float f, cv::Mat& m, vector<couple>&I, int mode) {
	cv::Mat mm(I.size() - 1, 1, CV_32F);
	for (int i = 0; i < mm.rows; i++) {
		float temp = m.at<float>(I[i].second);
		if (mode == 1) {//max
			temp = f > temp ? f : temp;
		}
		else {//min
			temp = f < temp ? f : temp;
		}
		mm.at<float>(i, 0) = temp;
	}
	return mm;
}

void tinyFaceDetector::_nms2(float f, cv::Mat& m) {
	for (int i = 0; i < m.rows; i++) {
		float temp = m.at<float>(i, 0);
		m.at<float>(i, 0) = temp > f ? temp : f;
	}
}

cv::Mat tinyFaceDetector::nms(const cv::Mat& boxList, float thresh) {
	int numAll = boxList.rows;
	cv::Mat x1 = boxList.col(0);
	cv::Mat y1 = boxList.col(1);
	cv::Mat x2 = boxList.col(2);
	cv::Mat y2 = boxList.col(3);
	cv::Mat sc = boxList.col(5);
	cv::Mat area = (x2 - x1 + 1).mul(y2 - y1 + 1);
	vector<couple> coupleV(numAll);
	for (int i = 0; i < numAll; i++) {
		coupleV[i] = couple(boxList.at<float>(i, 5), i);
	}
	sort(coupleV.begin(), coupleV.end());

	vector<int> temp(sc.rows, 0);
	cv::Mat pick(temp);
	int counter = 0;
	while (!coupleV.empty()) {
		int last = coupleV.size();
		int i = coupleV.rbegin()->second;
		pick.at<float>(counter, 0) = i;
		counter++;
		cv::Mat xx1 = _nms1(x1.at<float>(i, 0), x1, coupleV, 1);
		cv::Mat yy1 = _nms1(y1.at<float>(i, 0), y1, coupleV, 1);
		cv::Mat xx2 = _nms1(x2.at<float>(i, 0), x2, coupleV, 0);
		cv::Mat yy2 = _nms1(y2.at<float>(i, 0), y2, coupleV, 0);
		cv::Mat w = xx2 - xx1 + 1;
		cv::Mat h = yy2 - yy1 + 1;
		_nms2(0, w);
		_nms2(0, h);
		cv::Mat inter = w.mul(h);

		cv::Mat areaI(inter.rows, 1, CV_32F);
		for (int i = 0; i < coupleV.size() - 1; i++) {
			areaI.at<float>(i, 0) = area.at<float>(coupleV[i].second);
		}
		cv::Mat o = inter / (areaI - inter + area.at<float>(i, 0));
		vector<couple> temp;
		for (int i = 0; i < o.rows; i++) {
			if (o.at<float>(i, 0) <= thresh) {
				temp.push_back(coupleV[i]);
			}
		}
		coupleV = temp;
	}
	cv::Mat ret(counter, 1, CV_32F);
	pick.rowRange(0, counter).copyTo(ret);
	return ret;
}

cv::Mat tinyFaceDetector::mergeCols(const cv::Mat& A, const cv::Mat& B)
{
	int totalRows = A.rows + B.rows;

	cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
	cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}

void tinyFaceDetector::cvVectorSetInRange(cv::Mat& im, float min, float max) {
	for (int i = 0; i < im.rows; i++) {
		if (im.at<float>(i, 0) < min) {
			im.at<float>(i, 0) = min;
		}
		else if (im.at<float>(i, 0) > max) {
			im.at<float>(i, 0) = max;
		}
	}
}

void tinyFaceDetector::cvMat2caffeBlobData(const cv::Mat& image, float* const blobData) // MATLAB caffe model, blob data structure is different with that of C++
{
	int height = image.rows, width = image.cols, channels = image.channels();
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			blobData[(0 * width + w)*height + h] = float(image.at<cv::Vec3f>(h, w)[2]); // here we can see the blob data structure is column major, while blob in cpp is row major
			blobData[(1 * width + w)*height + h] = float(image.at<cv::Vec3f>(h, w)[1]);
			blobData[(2 * width + w)*height + h] = float(image.at<cv::Vec3f>(h, w)[0]); // B,G,R => R,G,B 
		}
	}
}

vector<string> tinyFaceDetector::getFileName(std::string& pathName) {
	vector<string> nameList;
	boost::filesystem::path filePath(pathName);
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator iter(filePath); iter != end_iter; iter++)
	{
		if (!boost::filesystem::is_directory(iter->status())) {
			nameList.push_back(iter->path().string());
		}
	}
	return nameList;
}

void tinyFaceDetector::run(MODEL m) {
	if (m == GPU) {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(0);
	}
	else {
		Caffe::set_mode(Caffe::CPU);
	}

	Phase phase = TEST;
	boost::shared_ptr<Net<float> > net = boost::shared_ptr<Net<float> >(new Net<float>(protoPath, caffe::TEST));
	net->CopyTrainedLayersFrom(modelPath);

	cv::Vec3f averageImage({ (float)101.8384, (float)110.5463, (float)119.2996 }); // the mean of pixels at each channel => (B,G,R)

	vector<float> clusterV; // reading cluster.dat into cv::Mat clusterMat
	fstream fh;
	fh.open(clusterPath, ios::in);
	string buffer;
	int clusterRow = 0;
	int eleSum = 0;
	while (std::getline(fh, buffer)) {
		clusterRow++;
		float tempD = 0;
		stringstream tempSS(buffer);
		while (tempSS >> tempD) {
			clusterV.push_back(tempD);
			eleSum++;
		}
	}
	int clusterCol = eleSum / clusterRow;
	cv::Mat clusterMat({ clusterCol,clusterRow }, CV_32F, &clusterV[0]);

	cv::Mat clusterMat_h = clusterMat.col(3) - clusterMat.col(1) + 1;
	cv::Mat clusterMat_w = clusterMat.col(2) - clusterMat.col(0) + 1;

	int img_num = 0;

	boost::filesystem::create_directory(savePath);
	vector<string> fileNameList = getFileName(filePath);
	for (int i = 0; i < fileNameList.size(); i++)
	{
		string picturePath = fileNameList[i];
		cv::Mat imageU8 = cv::imread(picturePath); // reading the picture data and preprocessing
		cv::Mat imagefloat;
		imageU8.convertTo(imagefloat, CV_32F); // 8-bit unsigned short => 32-bit signed float
		cv::Mat imageResized;
		int raw_w = 500, raw_h = 300;
		cv::resize(imagefloat, imageResized, { raw_w,raw_h }, cv::INTER_LINEAR);
		double wMax = 0, wMin = 0;
		cv::minMaxIdx(clusterMat_w, &wMin, &wMax);
		double hMax = 0, hMin = 0;
		cv::minMaxIdx(clusterMat_h, &hMin, &hMax);
		int minScale = floor(MIN(floor(log2(wMax / raw_w)), floor(log2(hMax / raw_h))));
		int maxScale = floor(MIN(1, -log2(MAX(raw_h, raw_w) / 5000)));
		vector<float> scales;
		for (int j = minScale; j <= maxScale; j++)
		{
			scales.push_back(pow(2, j));
		}
		cv::Mat reg_box({ 6,0 }, CV_32F);
		for (int j = 0; j < scales.size(); j++) {
			float scale = 0.5;//scales[j];
			cv::Mat imageScaled;
			cv::resize(imageResized, imageScaled, cv::Size(), scale, scale, cv::INTER_LINEAR);
			imageScaled = imageScaled - averageImage;
			//
			set<int> ignoreTidsS;
			for (int k = 1; k < clusterRow; k++) {
				ignoreTidsS.insert(k);
			}
			set<int> tidsS;
			if (scale <= 1) {
				tidsS = { 5,6,7,8,9,10,11,12 };
			}
			else {
				tidsS = { 5,6,7,8,9,10,11,12,19,20,21,22,23,24,25 };
			}
			vector<int> ignoreTids;
			for (set<int>::iterator itor = ignoreTidsS.begin(); itor != ignoreTidsS.end(); itor++) {
				if (tidsS.find(*itor) == tidsS.end()) {
					ignoreTids.push_back(*itor);
				}
			}
			//
			int img_h = imageScaled.rows, img_w = imageScaled.cols;

			float* blobData = new float[3 * img_w* img_h];
			cvMat2caffeBlobData(imageScaled, blobData);
			net->blob_by_name("data")->Reshape(1, 3, img_w, img_h);
			Blob<float> * input_blobs = net->blob_by_name("data").get();
			switch (Caffe::mode()) {
			case Caffe::CPU:
				memcpy(input_blobs->mutable_cpu_data(), blobData, sizeof(float)* input_blobs->count());
				break;
			case Caffe::GPU:
				caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), blobData, input_blobs->mutable_gpu_data());
				break;
			default:
				LOG(FATAL) << "Unknow Caffe mode";
			}

			net->Forward();
			Blob<float>* score_final_blob = net->blob_by_name("fusex").get();
			int blob_h = score_final_blob->shape()[2];
			int blob_w = score_final_blob->shape()[3];
			const float* score_final_data = score_final_blob->cpu_data();
			const int channelTaken = 25;
			vector<float> score_clc(score_final_data, score_final_data + blob_h*blob_w*channelTaken);
			for (int k = 0; k < ignoreTids.size(); k++) {
				int c = ignoreTids[k] - 1;
				memset(&score_clc[c*blob_h*blob_w], 0, sizeof(float)*blob_h*blob_w);
			}

			vector<float> fyv, fxv, fcv;
			vector<float> scoresv;
			for (int k = 0; k < score_clc.size(); k++) {
				if (score_clc[k] > thresh) {
					fcv.push_back(k / (blob_h*blob_w));
					fxv.push_back((k % (blob_h*blob_w)) / blob_w);
					fyv.push_back(((k % (blob_h*blob_w)) % blob_w));
					scoresv.push_back(score_clc[k]);
				}
			}
			int _row = score_clc.size(), _col = 1;
			if (fyv.size() > 0) {
				cv::Mat fc(fcv);
				cv::Mat fy(fyv);
				cv::Mat fx(fxv);
				cv::Mat scores(scoresv);
				cv::Mat cy = fy * 8 - 1;
				cv::Mat cx = fx * 8 - 1;
				vector<float> chv, cwv;
				for (int k = 0; k < fcv.size(); k++) {
					chv.push_back(clusterMat_h.at<float>(fcv[k], 0));
					cwv.push_back(clusterMat_w.at<float>(fcv[k], 0));
				}
				cv::Mat ch(chv);
				cv::Mat cw(cwv);

				cv::Mat x1 = (cx - cw / 2) / scale;
				cv::Mat y1 = (cy - ch / 2) / scale;
				cv::Mat x2 = (cx + cw / 2) / scale;
				cv::Mat y2 = (cy + cw / 2) / scale;

				cvVectorSetInRange(x1, 0, raw_w - 1);
				cvVectorSetInRange(x2, 0, raw_w - 1);
				cvVectorSetInRange(y1, 0, raw_h - 1);
				cvVectorSetInRange(y2, 0, raw_h - 1);

				cv::Mat temp_reg_box({ 6,x1.rows }, CV_32F);
				x1.copyTo(temp_reg_box.colRange(0, 1));
				y1.copyTo(temp_reg_box.colRange(1, 2));
				x2.copyTo(temp_reg_box.colRange(2, 3));
				y2.copyTo(temp_reg_box.colRange(3, 4));
				fc.copyTo(temp_reg_box.colRange(4, 5));
				scores.copyTo(temp_reg_box.colRange(5, 6));
				reg_box = mergeCols(reg_box, temp_reg_box);
				delete blobData;
			}
		}
		cv::Mat valid_box = nms(reg_box, nmsthresh);
		img_num++;
		index++;

		cv::Mat post = imageResized.clone();
		for (int j = 0; j < valid_box.rows; j++) {
			int temp = valid_box.at<float>(j);
			drawRect(post, reg_box.at<float>(temp, 0), reg_box.at<float>(temp, 1), reg_box.at<float>(temp, 2), reg_box.at<float>(temp, 3));
		}
		cv::imwrite(savePath + "/" + std::to_string(index) + ".jpg", post);
	}
}
