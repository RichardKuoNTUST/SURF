#include "pch.h"
#include "KUO_SURF.h"
#include <algorithm>

void SURF(Mat image, std::vector<InterestPoint> & IPTs, uint8_t octaves, unsigned int step, float threshold) {
	Mat IM;  //Àx¦s¿n¤À¹Ï
	IM.create(image.rows, image.cols, CV_32S);
	integral(image, IM, CV_32S);
	std::vector<FilterLayer *> LAYER;
	unsigned int imagerows = round(image.rows / step);
	unsigned int imagecols = round(image.cols / step);
	for (int i=0;i<=os[octaves-1][3];i++)
		LAYER.push_back(new FilterLayer(imagerows>>Lstep[i], imagecols>>Lstep[i], step<<Lstep[i], L[i]));

	for (int i = 0; i < LAYER.size(); i++)
		BUILDFILTERLAYER(&IM, LAYER[i]);

	float X[3] = {0.0,0.0,0.0};
	for (int i = 0; i < octaves; i++) {
		for (int j = 0; j < 2; j++) {
			for (int a = 0; a < LAYER[os[i][j + 2]]->heigh; a++) {
				for (int b = 0; b < LAYER[os[i][j + 2]]->width; b++) {
					if (isExtremum(a,b,LAYER[os[i][j]], LAYER[os[i][j+1]], LAYER[os[i][j+2]],threshold)) {
						interpolateStep(a, b, LAYER[os[i][j]], LAYER[os[i][j + 1]], LAYER[os[i][j + 2]], X);
						if ((fabs(X[0]) < 0.5) && (fabs(X[1]) < 0.5) && (fabs(X[2]) < 0.5)) {
							InterestPoint ipt;
							ipt.x = float((b + X[0]) * LAYER[os[i][j + 2]]->step);
							ipt.y = float((a + X[1]) * LAYER[os[i][j + 2]]->step);
							ipt.scale = float(3*LAYER[os[i][j + 1]]->filter + X[2]*(LAYER[os[i][j+1]]->filter-LAYER[os[i][j]]->filter))*3*0.1333f;
							ipt.lapacian = LAYER[os[i][j + 1]]->laplacian[a*LAYER[os[i][j+1]]->width+b];
							ipt.orientation = ORORIENTAION(ipt.x, ipt.y, ipt.scale, &IM);
							IPTs.push_back(ipt);
						}
					}
				}
			}
		}
	}
	IM.release();
}

void BUILDFILTERLAYER(Mat *Image, FilterLayer *layer) {
	int  i, ii, j, jj;
	int rows = layer->heigh;
	int cols = layer->width;
	int step = layer->step;
	int filter = layer->filter;
	float *v = layer->value;
	int *lap = layer->laplacian;
	int num = 0;
	float dxx, dyy, dxy;
	int l = filter * 3;
	int r = 2 * filter - 1;
	const float t = (1.0f / (filter*filter));
	for (i = 1, ii = 0; ii < rows; i += step, ii++) {
		for (j = 1, jj = 0; jj < cols; j += step, jj++) {
			dxx = (float)BOXDXX(Image, i, j, r, l, filter)*t; 
			dyy = (float)BOXDYY(Image, i, j, l, r, filter)*t;
			dxy = (float)BOXDXY(Image, i, j, filter)*t;
			v[num] = (dxx * dyy - 0.83338641f*dxy*dxy);
			lap[num] = ((dxx + dyy) >= 0) ? (1) : (0);
			num++;
		}
	}
}

bool isExtremum(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T, float threshold) {
	int stepTM = T->step / M->step;
	int stepTB = T->step / B->step;
	int s = T->step;
	int num = 0;
	int layerBorder = round((T->filter*3 + 1) / (2));
	float can = M->value[i*stepTM*M->width + j * stepTM];
	if ((i*s > layerBorder) && (i < (T->heigh*s - layerBorder)) && (j*s > layerBorder) && (j*s < (T->width*s - layerBorder))) {
		if (can > threshold) {
			for (int a = -1; a <= 1; a++) {
				for (int b = -1; b <= 1; b++) {
					if (can > T->value[(i + a)*T->width + j + b])
						num++;
					if (can > M->value[((i + a)*stepTM)*M->width + (j + b)*stepTM])
						num++;
					if (can > B->value[((i + a)*stepTB)*B->width + (j + b)*stepTB])
						num++;
				}
			}
			if (num == 26)
				return 1;
		}

	}
	return 0;
}

void interpolateStep(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T, float* P) {
	CvMat* dD, *H, *H_inv, X;
	double x[3] = {0.0,0.0,0.0};
	dD = deriv3D(i, j, B, M, T);
	H = hessian3D(i, j, B, M, T);
	H_inv = cvCreateMat(3, 3, CV_64FC1);
	cvInvert(H, H_inv, CV_SVD);
	cvInitMatHeader(&X, 3, 1, CV_64FC1, x, CV_AUTOSTEP);
	cvGEMM(H_inv, dD, -1, NULL, 0, &X, 0);

	cvReleaseMat(&dD);
	cvReleaseMat(&H);
	cvReleaseMat(&H_inv);

	P[2] = x[2];
	P[1] = x[1];
	P[0] = x[0];
}

CvMat* deriv3D(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T) {
	CvMat* dI;
	double dx, dy, ds;
	int stepTM = T->step / M->step;
	int stepTB = T->step / B->step;
	dx = (M->value[i*stepTM*M->width + (j + 1)*stepTM] - M->value[i*stepTM*M->width + (j - 1)*stepTM]) / (2.0);
	dy = (M->value[((i + 1)*stepTM)*M->width + j * stepTM] - M->value[((i - 1)*stepTM)*M->width + j * stepTM]) / (2.0);
	ds = (T->value[i*T->width + j] - B->value[i*stepTB*B->width + j * stepTB]) / (2.0);

	dI = cvCreateMat(3, 1, CV_64FC1);
	cvmSet(dI, 0, 0, dx);
	cvmSet(dI, 1, 0, dy);
	cvmSet(dI, 2, 0, ds);

	return dI;
}

CvMat* hessian3D(int i, int j, FilterLayer *B, FilterLayer *M, FilterLayer *T) {
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;
	int stepTM = T->step / M->step;
	int stepTB = T->step / B->step;
	v = M->value[i*stepTM*M->width + j * stepTM];
	dxx = (M->value[i*stepTM*M->width + (j + 2)*stepTM] + M->value[i*stepTM*M->width + (j - 2)*stepTM] - 2 * v) / (4.0);
	dyy = (M->value[((i + 2)*stepTM)*M->width + j * stepTM] + M->value[((i - 2)*stepTM)*M->width + j * stepTM] - 2 * v) / (4.0);
	dss = (T->value[i*T->width + j] + B->value[i*stepTB*B->width + j * stepTB] - 2 * v) / (2.0);
	dxy = (M->value[((i + 1)*stepTM)*M->width + (j + 1)*stepTM] - M->value[((i + 1)*stepTM)*M->width + (j - 1)*stepTM] -
		M->value[((i - 1)*stepTM)*M->width + (j + 1)*stepTM] + M->value[((i - 1)*stepTM)*M->width + (j - 1)*stepTM]) / (4.0);
	dxs = (T->value[i*T->width + j + 1] - T->value[i*T->width + j - 1] -
		B->value[i*stepTB*B->width + (j + 1)*stepTB] + B->value[i*stepTB*B->width+ (j - 1)*stepTB]) / (4.0);
	dys = (T->value[(i + 1)*T->width + j] - T->value[(i - 1)*T->width + j] -
		B->value[((i + 1)*stepTB)*B->width + j * stepTB] + B->value[((i - 1)*stepTB)*B->width + j * stepTB]) / (4.0);

	H = cvCreateMat(3, 3, CV_64FC1);
	cvmSet(H, 0, 0, dxx);
	cvmSet(H, 0, 1, dxy);
	cvmSet(H, 0, 2, dxs);
	cvmSet(H, 1, 0, dxy);
	cvmSet(H, 1, 1, dyy);
	cvmSet(H, 1, 2, dys);
	cvmSet(H, 2, 0, dxs);
	cvmSet(H, 2, 1, dys);
	cvmSet(H, 2, 2, dss);

	return H;
}

float ORORIENTAION(float iptx, float ipty, float scale, Mat *I) {
	float x[150] = {0.0};
	float y[150] = {0.0};
	float ang[150] = { 0.0 };
	int num = 0;
	float X = 0.0;
	float Y = 0.0;
	float g = 0.0;
	for (int i = -6; i <= 6; i++) {
		for (int j = -6; j <= 6; j++) {
			if (i * i + j * j <= 36) {
				X = round(iptx + i * scale);
				Y = round(ipty + j * scale);
				if (X>=0 && X<I->size().width && Y>=0 && Y<I->size().height) {
					g = exp(-((X-iptx) * (X-iptx) + (Y-ipty) * (Y-ipty)) / (2 * scale*scale)) / (2 * PI*scale*scale);
					x[num] = (float)BOXHAARX(I, Y, X, scale)*g;
					y[num] = (float)BOXHAARY(I, Y, X, scale)*g;
					ang[num] = atan(y[num] / x[num]) * 180 / PI;
				}
				//printf("%f  %f\n",x[num],y[num]);
				num++;
			}
		}
	}
	float maxs = 0.0;
	float sumx;
	float sumy;
	float o = 0.0;
	for (int ang1 = 0; ang1 <= 360; ang1+=9) {
		int ang2 = ((ang1 + 60) > 360) ? (ang1 - 300) : (ang1 + 60);
		sumx = 0.0;
		sumy = 0.0;
		for (int i = 0; i < num; i++) {
			if ((ang[i] >= ang1 && ang1 < ang2) || (ang[i]>=ang2 && ang[i]>=ang1 && ang1>300)) {
				sumx += x[i];
				sumy += y[i];
			}
		}
		if (sumx*sumx + sumy * sumy > maxs) {
			maxs = sumx * sumx + sumy * sumy;
			o = atan(sumy / sumx) * 180 / PI;
		}
	}
	return o;
}