#pragma once
#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>     // std::cout
#include <algorithm>    // std::max
#include <memory.h>

#define PI 3.14159
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
using namespace cv;
using namespace std;
namespace cv {
	using std::vector;
}


static const int os[6][5] = { {0, 1, 2, 3},
							  {1, 3, 4, 5},
							  {3, 5, 6, 7},
							  {5, 7, 8, 9},
							  {7, 9, 10, 11},
							  {9, 11, 12, 13} };

static const int L[12] = {3,5,7,9,13,17,25,33,49,65,97,129};
static const int Lstep[12] = {0,0,0,0,1,1,2,2,3,3,4,4};
class InterestPoint;
typedef std::vector<InterestPoint> IPTV;
class FilterLayer;

class InterestPoint {

public:

	float x;       //興趣點列座標
	float y;       //興趣點行座標
	float scale;     //興趣點所在的尺度
	int lapacian;  //興趣點
	float orientation; //興趣點的主要方向
	float descriptor[64]; //興趣點的描述子
};

class FilterLayer {
public:
	unsigned int heigh;
	unsigned int width;
	unsigned int step;
	unsigned int filter;
	float *value;
	int *laplacian;
	
	FilterLayer(unsigned int heigh, unsigned int width, unsigned int step, unsigned int filter) {
		this->heigh = heigh;
		this->width = width;
		this->step = step;
		this->filter = filter;
		value = new float[heigh*width];
		laplacian = new int[heigh*width];
	}

};

void SURF(Mat image, std::vector<InterestPoint> & IPTs, uint8_t octaves, unsigned int step, float threshold);
void BUILDFILTERLAYER(Mat *image, FilterLayer *layer);
bool isExtremum(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T, float threshold);
void interpolateStep(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T, float* P);
CvMat* deriv3D(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T);
CvMat* hessian3D(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T);
float ORORIENTAION(float iptx,float ipty, float scale, Mat *I);
inline int BOXDXX(Mat *image, int centerx, int centery, int h, int w, int s) {
	int *data = (int *)image->data;
	int row = image->size().height;
	int col = image->size().width;
	int step0 = (int)image->step1(0);
	int step1 = (int)image->step1(1);
	int ax1 = std::max(0, centerx - (h - 1) / 2 - 1);
	int ay1 = std::max(0, centery - (w - 1) / 2 - 1);
	int dx1 = std::min(row - 1, centerx + (h - 1) / 2);
	int dy1 = std::min(col - 1, centery + (w - 1) / 2);
	int ay2 = std::max(0, centery - (s - 1) / 2 - 1);
	int dy2 = std::min(col- 1, centery + (s - 1) / 2);
	int white = (data[dx1 * step0 + dy1 * step1] + data[ax1 * step0 + ay1 * step1] - data[ax1 * step0 + dy1 * step1] - data[dx1 * step0 + ay1 * step1]);
	int black = (data[dx1 * step0 + dy2 * step1] + data[ax1 * step0 + ay2 * step1] - data[ax1 * step0 + dy2 * step1] - data[dx1 * step0 + ay2 * step1]);
	assert(white>=0);
	assert(black>=0);
	return (white-3*black);
}

inline int BOXDYY(Mat *image, int centerx, int centery, int h, int w, int s) {
	int *data = (int *)image->data;
	int row = image->size().height;
	int col = image->size().width;
	int step0 = (int)image->step1(0);
	int step1 = (int)image->step1(1);
	int ax1 = std::max(0, centerx - (h - 1) / 2 - 1);
	int ay1 = std::max(0, centery - (w - 1) / 2 - 1);
	int dx1 = std::min(row - 1, centerx + (h - 1) / 2);
	int dy1 = std::min(col - 1, centery + (w - 1) / 2);
	int ax2 = std::max(0, centerx - (s - 1) / 2 - 1);
	int dx2 = std::min(row - 1, centerx + (s - 1) / 2);
	int white = (data[dx1 * step0 + dy1 * step1] + data[ax1 * step0 + ay1 * step1] - data[ax1 * step0 + dy1 * step1] - data[dx1 * step0 + ay1 * step1]);
	int black = (data[dx2 * step0 + dy1 * step1] + data[ax2 * step0 + ay1 * step1] - data[ax2 * step0 + dy1 * step1] - data[dx2 * step0 + ay1 * step1]);
	assert(white>=0);
	assert(black>=0);
	return (white-3*black);
}

inline int BOXDXY(Mat *image, int centerx, int centery, int h) {
	int *data = (int *)image->data;
	int row = image->size().height;
	int col = image->size().width;
	int step0 = (int)image->step1(0);
	int step1 = (int)image->step1(1);
	int dx1 = std::max(0, centerx - 1);
	int dy1 = std::max(0, centery - 1);
	int ax1 = std::max(0, dx1 - h);
	int ay1 = std::max(0, dy1 - h);
	int dx2 = std::min(row - 1, centerx + h);
	int dy2 = std::min(col - 1, centery + h);

	int white1 = (data[dx1 * step0 + dy1 * step1] + data[ax1 * step0 + ay1 * step1] - data[ax1 * step0 + dy1 * step1] - data[dx1 * step0 + ay1 * step1]);
	int white2 = (data[dx2 * step0 + dy2 * step1] + data[centerx * step0 + centery * step1] - data[centerx * step0 + dy2 * step1] - data[dx2 * step0 + centery * step1]);
	int black1 = (data[dx2 * step0 + dy1 * step1] + data[centerx * step0 + ay1 * step1] - data[centerx * step0 + dy1 * step1] - data[dx2 * step0 + ay1 * step1]);
	int black2 = (data[dx1 * step0 + dy2 * step1] + data[ax1 * step0 + centery * step1] - data[ax1 * step0 + dy2 * step1] - data[dx1 * step0 + centery * step1]);
	assert(white1>=0);
	assert(white2>=0);
	assert(black1>=0);
	assert(black2>=0);
	return (white1+white2-black1-black2);
	
}

inline int BOXHAARX(Mat *image, int centerx, int centery, int h) {
	int *data = (int *)image->data;
	int row = image->size().height;
	int col = image->size().width;
	int step0 = (int)image->step1(0);
	int step1 = (int)image->step1(1);
	int ax = max(0, centerx - 2 * h);
	int ay = max(0, centery - 2 * h);
	int dx = min(row-1, centerx + 2 * h);
	int dy = min(col-1, centery + 2 * h);
	int black = data[dx * step0 + centery * step1] + data[ax*step0 + ay * step1] - data[ax*step0 + centery * step1] - data[dx*step0 + ay * step1];
	int white = data[dx * step0 + dy * step1] + data[ax*step0 + centery * step1] - data[ax*step0 + dy * step1] - data[dx*step0 + centery * step1];
	assert(black >= 0);
	assert(white >= 0);
	return (white - black);
}

inline int BOXHAARY(Mat *image, int centerx, int centery, int h) {
	int *data = (int *)image->data;
	int row = image->size().height;
	int col = image->size().width;
	int step0 = (int)image->step1(0);
	int step1 = (int)image->step1(1);
	int ax = max(0, centerx - 2 * h);
	int ay = max(0, centery - 2 * h);
	int dx = min(row-1, centerx + 2 * h);
	int dy = min(col-1, centery + 2 * h);
	int black = data[centerx * step0 + dy * step1] + data[ax*step0 + ay * step1] - data[ax*step0 + dy * step1] - data[centerx*step0 + ay * step1];
	int white = data[dx*step0 + dy * step1] + data[centerx*step0 + ay * step1] - data[centerx*step0 + dy * step1] - data[dx*step0 + ay * step1];
	assert(black >= 0);
	assert(white >= 0);
	return (white - black);
}