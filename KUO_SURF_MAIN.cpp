#include "pch.h"
#include "opencv2/opencv.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "KUO_SURF.h"

#define OCTAVES 3
#define STEP 2
#define THRESHOLD 1000
#define MATCHTHRESHOLD 0.9


using namespace cv;
using namespace std;

namespace cv {
	using std::vector;
}

int main() {
	Mat sc1, gray1, sc2, gray2;
	//Step 1 Input
	sc1 = imread("F:\\image1.jpg", CV_LOAD_IMAGE_UNCHANGED);
	gray1 = sc1.clone();
	cvtColor(sc1, gray1, CV_BGR2GRAY);
	IPTV IPTs1;
	SURF(gray1, IPTs1, OCTAVES, STEP, THRESHOLD);
	for (int i = 0; i < IPTs1.size(); i++) {
		circle(sc1, Point(IPTs1.at(i).x, IPTs1.at(i).y), IPTs1.at(i).scale, Scalar(255, 0, 0), 2);
		line(sc1, Point(IPTs1.at(i).x, IPTs1.at(i).y), Point(IPTs1.at(i).x + IPTs1.at(i).scale*cos(IPTs1.at(i).orientation), 
			IPTs1.at(i).y + IPTs1.at(i).scale*sin(IPTs1.at(i).orientation)), Scalar(0, 255, 0), 1);
	}
	namedWindow("Original Picture1", WINDOW_AUTOSIZE);
	imshow("Original Picture1", sc1);

	waitKey(0);
	return 0;
}
