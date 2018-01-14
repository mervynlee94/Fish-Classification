#include	<iostream>
#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	"Supp.h"
#include	<string>

using namespace std;
using namespace cv;
using namespace ml;

// Parameters
const int POLY_APPROX_EPSILON = 11;
const vector<string> folderName = { "Black Crappie", "Channel Catfish", "Tuna", "Longear Sunfish", "Sturgeon", "Walleye", "White Bass", "Yellow Perch" };

// Globals
int const	noOfImagePerCol = 1, noOfImagePerRow = 2; // create window partition 
Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];

void getLargestContour(Mat &img, vector<vector<Point> > &largestContour) {
	// Extract contours of the foreground image
	vector<vector<Point> > contoursH;
	vector<Vec4i> hierarchyH;
	findContours(img, contoursH, hierarchyH, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	// Extract contour with largest contour
	double largest_area = 0;
	int largest_contour_index;
	for (int i = 0; i < contoursH.size(); i++)
	{
		double area = contourArea(contoursH[i]);
		if (area > largest_area) { // If the area of contour > current largest contour
			largest_area = area;
			largest_contour_index = i; // Store the index of largest contour
		}
	}
	largestContour[0] = contoursH[largest_contour_index];
}

Mat rotateMat(Mat &img, double angle) {
	double offsetX, offsetY;
	double width = img.size().width;
	double height = img.size().height;
	Point2d center = Point2d(width / 2, height / 2);
	RotatedRect rotatedBounds = RotatedRect(center, img.size(), angle);
	Rect bounds = rotatedBounds.boundingRect();
	Mat resized = Mat::zeros(bounds.size(), img.type());
	offsetX = (bounds.width - width) / 2;
	offsetY = (bounds.height - height) / 2;
	Rect roi = Rect(offsetX, offsetY, width, height);
	img.copyTo(resized(roi));
	center += Point2d(offsetX, offsetY);
	Mat M = getRotationMatrix2D(center, angle, 1.0);
	warpAffine(resized, resized, M, resized.size());

	return resized;
}


double getOrientation(vector<Point> &pts, Mat &img, bool draw = true)
{
	//Construct a buffer used by the pca analysis
	Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
	//Store the position of the object
	Point pos = Point(pca_analysis.mean.at<double>(0, 0), pca_analysis.mean.at<double>(0, 1));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i, 0);
	}
	// Draw the principal components
	if (draw) {
		circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
		line(img, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]), CV_RGB(255, 255, 0));
		line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]), CV_RGB(0, 255, 255));
	}

	return atan2(eigen_vecs[0].y, eigen_vecs[0].x) * (180.0 / CV_PI);
}

/*!
* \brief Enlarge an ROI rectangle by a specific amount if possible
* \param from The image the ROI will be set on
* \param boundingBox The current boundingBox
* \param padding The amount of padding around the boundingbox
* \return The enlarged ROI as far as possible
*/
Rect enlargeROI(Mat frm, Rect boundingBox, int padding) {
	Rect returnRect = Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
	if (returnRect.x < 0)returnRect.x = 0;
	if (returnRect.y < 0)returnRect.y = 0;
	if (returnRect.x + returnRect.width >= frm.cols)returnRect.width = frm.cols - returnRect.x;
	if (returnRect.y + returnRect.height >= frm.rows)returnRect.height = frm.rows - returnRect.y;
	return returnRect;
}

Mat getSquareImage(const Mat& img, int target_width = 500) {
	int width = img.cols,
		height = img.rows;

	Mat square = Mat(target_width, target_width, img.type(), img.at<Vec3b>(0, 0));
	int max_dim = (width >= height) ? width : height;
	float scale = ((float)target_width) / max_dim;
	Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	resize(img, square(roi), roi.size());
	copyMakeBorder(square, square, 30, 30, 30, 30, BORDER_CONSTANT, img.at<Vec3b>(0, 0));

	return square;
}

void getFeaturePoints(String image, vector<float> &featurePoints, bool training) {
	Moments mom;
	double hu[7];
	Mat im_src = imread(image);
	Mat im_srcDisplay = getSquareImage(im_src);

	createWindowPartition(im_srcDisplay, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);
	im_srcDisplay.copyTo(win[0]);
	putText(legend[0], "Source Image", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	copyMakeBorder(im_src, im_src, 30, 30, 30, 30, BORDER_CONSTANT, im_src.at<Vec3b>(0, 0));
	// get Canny edge
	Mat im_gray;
	cvtColor(im_src, im_gray, CV_BGR2GRAY);

	Mat im_canny;
	GaussianBlur(im_gray, im_canny, Size(3, 3), 0, 0);
	Canny(im_canny, im_canny, 0, 150);
	dilate(im_canny, im_canny, Mat(), Point(-1, -1), 3);

	// Floodfill from point (0, 0)
	Mat im_floodfill = im_canny.clone();
	floodFill(im_floodfill, Point(0, 0), Scalar(255));

	// Invert floodfilled image
	Mat im_floodfill_inv;
	bitwise_not(im_floodfill, im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat im_out = (im_canny | im_floodfill_inv);

	// Extract contour with largest contour
	vector<vector<Point> > largestContour(1);
	getLargestContour(im_out, largestContour);

	// Get bounding rect of largest contour,
	// this is the region containing the fish
	Rect boundRect;
	boundRect = boundingRect(largestContour[0]);
	boundRect = enlargeROI(im_out, boundRect, 10);

	// Crop the fish and store into ROI,
	// rotate the image to principal axis
	Mat ROI = Mat(im_out, boundRect);
	double orientation = getOrientation(largestContour[0], im_out, false);
	im_out = rotateMat(ROI, orientation);

	getLargestContour(im_out, largestContour);

	// Get bounding rect of largest contour,
	// this is the region containing the fish
	boundRect = boundingRect(largestContour[0]);
	boundRect = enlargeROI(im_out, boundRect, 10);

	// Crop the fish and store into ROI,
	// normalize image size to 500x500
	ROI = Mat(im_out, boundRect);
	ROI = getSquareImage(ROI);

	getLargestContour(ROI, largestContour);
	// Get polygonal approximation of contour
	vector<Point> contoursApprox;
	approxPolyDP(largestContour[0], contoursApprox, POLY_APPROX_EPSILON, true);

	// Convert ROI to color image for display purposes
	cvtColor(ROI, ROI, CV_GRAY2BGR);
	getOrientation(largestContour[0], ROI);

	// Draw the contours to for display purposes
	drawContours(ROI, largestContour, 0, Scalar(0, 0, 255), 2); // Draw the largest contour using previously stored index.
	if (training) {
		ROI.copyTo(win[1]);
		putText(legend[1], "Training", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		imshow("Training", largeWin);
	}
	else {
		cout << image << endl;
		ROI.copyTo(win[1]);
		putText(legend[1], "Testing", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		imshow("Testing", largeWin);
	}

	// Calculate features:
	// hu moment 1, 2
	// contour perimeter
	// contour area
	mom = moments(contoursApprox, true);
	HuMoments(mom, hu);
	for (int i = 0; i < 2; i++) {
		cout << (float)hu[i] << ", ";
		featurePoints.push_back((float)hu[i]);
	}
	float cLength = (float)arcLength(contoursApprox, true);
	float cArea = (float)contourArea(largestContour[0]);
	cout << cLength << ", " << cArea << endl;
	featurePoints.push_back(cLength);
	featurePoints.push_back(cArea);
}

int main() {
	vector<vector<float> > trainingData;
	vector<int> labels;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	for (int folder = 0; folder < folderName.size(); folder++) {
		cout << "Features for " << folderName[folder] << ": " << endl;
		vector<String> images;
		glob("Inputs\\Fishes\\Training\\" + folderName[folder] + "\\*", images);
		for (int i = 0; i < images.size(); i++) {
			vector<float> featurePoints;
			getFeaturePoints(images[i], featurePoints, true);
			trainingData.push_back(featurePoints);
			labels.push_back(folder);
			waitKey();
		}
		cout << endl;
	}
	destroyWindow("Training");
	// Copy data from vector into Mat
	Mat trainingDataMat(trainingData.size(), trainingData[0].size(), CV_32F);
	for (size_t i = 0; i < trainingData.size(); i++)
		for (size_t j = 0; j < trainingData[i].size(); j++)
			trainingDataMat.at<float>(i, j) = trainingData[i][j];
	Mat labelsMat(labels);

	// Normalization of the data
	// Xnew = (X - mean) / sigma
	vector<double> mean, sigma;
	for (int i = 0; i < trainingDataMat.cols; i++) {  //take each of the features in vector
		Scalar meanOut, sigmaOut;
		meanStdDev(trainingDataMat.col(i), meanOut, sigmaOut);  //get mean and std deviation
		mean.push_back(meanOut[0]);
		sigma.push_back(sigmaOut[0]);
	}
	for (size_t i = 0; i < trainingData.size(); i++)
		for (size_t j = 0; j < trainingData[i].size(); j++)
			trainingDataMat.at<float>(i, j) = (trainingData[i][j] - mean[j]) / sigma[j];

	cout << trainingDataMat << endl << endl;
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);

	// Predict
	int correct = 0, total_tested = 0;
	for (int folder = 0; folder < folderName.size(); folder++) {
		cout << "Testing for " << folderName[folder] << ": " << endl;
		vector<String> images;
		glob("Inputs\\Fishes\\Testing\\" + folderName[folder] + "\\*", images);
		for (int i = 0; i < images.size(); i++) {
			vector<float> featurePoints;
			getFeaturePoints(images[i], featurePoints, false);
			Mat sampleMat(1, featurePoints.size(), CV_32F);
			for (int i = 0; i < featurePoints.size(); i++)
				sampleMat.at<float>(0, i) = (featurePoints[i] - mean[i]) / sigma[i]; //normalization

			int response = svm->predict(sampleMat);
			total_tested++;
			if (folder == response) {
				correct++;
			}
			cout << "EXPECTING: " << folderName[folder] << ", GET: " << folderName[response] << ((folder == response) ? " CORRECT" : " WRONG") << endl << endl;
			waitKey();
		}
	}
	cout << "GOT " << correct << " OUT OF " << total_tested << " CORRECT. (" << 100 * (correct / (total_tested*1.0)) << "%)" << endl;
	system("pause");
	return 0;

}
