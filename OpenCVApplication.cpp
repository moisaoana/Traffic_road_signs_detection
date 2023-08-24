// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

#define number_of_colors 6

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

bool check_pixel_color(Vec3b pixel, Vec3b expected_color)
{

	uchar b = pixel[0];
	uchar g = pixel[1];
	uchar r = pixel[2];

	if (b == expected_color[0] && g == expected_color[1] && r == expected_color[2]) {
		return true;
	}

	return false;
}

bool isInside(Mat img, int i, int j)
{
	if (i < 0 || j < 0)
		return false;
	if (i < img.rows && j < img.cols)
		return true;
	return false;
}

Mat label_objects(Mat img) {


	int iDirection[] = { -1, 0, +1, 0 };
	int jDireciton[] = { 0, -1, 0, +1 };

	Mat labels(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
			labels.at<uchar>(i, j) = 0;
	}

	int label = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {

			Vec3b pixel = img.at< Vec3b>(i, j);
			uchar b = pixel[0];
			uchar g = pixel[1];
			uchar r = pixel[2];

			if ((!check_pixel_color(pixel,Vec3b(255,0,255))) && labels.at<uchar>(i, j) == 0)
			{
				label++;
				printf("Label:%d\n", label);
				std::queue<Point> Queue;
				labels.at<uchar>(i, j) = label;
				Queue.push(Point(i, j));
				while (!Queue.empty()) {
					Point p = Queue.front();
					Queue.pop();
					//printf("fdb df  %d %d\n", p.x,p.y);
					for (int k = 0; k < 4; k++) {

						if (isInside(img, p.x + iDirection[k], p.y + jDireciton[k])) {
							Vec3b pixel = img.at< Vec3b>(p.x + iDirection[k], p.y + jDireciton[k]);
							uchar b1 = pixel[0];
							uchar g1 = pixel[1];
							uchar r1 = pixel[2];

							if (((b1 == b && g1 == g && r1 == r)  || (b1 == 255 && g1 == 255 && r1 == 255) || (b1 == 0 && g1==0 && r1 ==0) || (b1 == 0 && g1 == 255 && r1 == 0) || (b1 == 0 && g1 == 255 && r1 == 255) || (b1 == 0 && g1 == 165 && r1 == 255)) && labels.at<uchar>(p.x + iDirection[k], p.y + jDireciton[k]) == 0) {
								labels.at<uchar>(p.x + iDirection[k], p.y + jDireciton[k]) = label;
								//printf("Neoghbpor %d %d\n", p.x + iDirection[k], p.y + jDireciton[k]);
								//printf("Neoghbpor %d %d %d\n", b1, g1, r1);
								Queue.push(Point(p.x + iDirection[k], p.y + jDireciton[k]));
							}
						}

					}
				}
			}
		}
	}
	//imshow("labeled image", labels);
	return labels;

}

Vec3b* generate_signs_colors() {

	Vec3b* colors = (Vec3b*)malloc(number_of_colors * sizeof(Vec3b));

	//red
	colors[0][0] = 0;
	colors[0][1] = 0;
	colors[0][2] = 255;

	//blue
	colors[1][0] = 255;
	colors[1][1] = 0;
	colors[1][2] = 0;

	//black
	colors[2][0] = 255;
	colors[2][1] = 255;
	colors[2][2] = 0;

	//yellow
	colors[3][0] = 255;
	colors[3][1] = 255;
	colors[3][2] = 0;

	//green
	colors[4][0] = 0;
	colors[4][1] = 255;
	colors[4][2] = 0;

	//orange
	colors[5][0] = 0;
	colors[5][1] = 165;
	colors[5][2] = 255;
	return colors;
}


Mat standardize_colors(Mat img) {

	Mat dst(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			Vec3b pixel = img.at< Vec3b>(i, j);
			uchar b = pixel[0];
			uchar g = pixel[1];
			uchar r = pixel[2];
			
			//red
			if (r <= 255 && r >= 100 && b <= 50 && b >= 0 && g <= 50 && g >= 0) {

				dst.at<Vec3b>(i, j)[0] = 0;
				dst.at<Vec3b>(i, j)[1] = 0;
				dst.at<Vec3b>(i, j)[2] = 255;

			}
			else
			{
				dst.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
			}

		}
	}

	return dst;

}


Mat transform_image(Mat original_img, Vec3b* colors) {

	Mat img = standardize_colors(original_img);

	Mat dst(img.rows, img.cols, CV_8UC3);

	int iDirection[] = { -1, 0, +1, 0 };
	int jDireciton[] = { 0, -1, 0, +1 };

	for (int i = 0; i < number_of_colors; i++) {
		Vec3b color = colors[i];
		uchar b1 = color[0];
		uchar g1 = color[1];
		uchar r1 = color[2];
		printf("%d %d %d\n", b1, g1, r1);

	}
	for (int i = 0; i < img.rows; i++) {
		bool enterColor = false;
		bool enterWhite = false;
		bool exitWhite = false;
		bool exitColor = false;
		std::vector<Point>points;
		std::vector<Point>pointsYellow;
		std::vector<Point>pointsGreen;
		std::vector<Point>pointsOrange;
		std::vector<Point>pointsBlack;
		Vec3b col;
		for (int j = 0; j < img.cols; j++) {

			Vec3b pixel = img.at< Vec3b>(i, j);
			uchar b = pixel[0];
			uchar g = pixel[1];
			uchar r = pixel[2];
			bool found = false;

			for (int k = 0; k < number_of_colors; k++) {
				
				Vec3b color = colors[k];
				uchar b1 = color[0];
				uchar g1 = color[1];
				uchar r1 = color[2];
				//printf("%d %d %d\n", b1, g1, r1);
				if (b == b1 && g == g1 && r == r1) {
					found = true;
					//printf(" heere %d %d %d\n", b, g, r);
					if (!enterColor) {
						enterColor = true;
					}
					else {
						if(enterWhite && r==255 && g==0 &&b==0) //
							exitColor = true;
					}
					break;
				}
				if (check_pixel_color(pixel,Vec3b(255,255,255))) {
					if (enterColor && !enterWhite) {
						enterWhite = true;
						col = pixel;
						//points.push_back(Point(i, j));
					}
					if (enterColor && enterWhite && !exitColor) {
						points.push_back(Point(i, j));
					}
				}
				if (check_pixel_color(pixel, Vec3b(0, 0, 0))) {
					if (enterColor && !enterWhite) {
						enterWhite = true;
						col = pixel;
						//points.push_back(Point(i, j));
					}
					if (enterColor && enterWhite && !exitColor) {
						pointsBlack.push_back(Point(i, j));
					}
				}
				if (check_pixel_color(pixel, Vec3b(0, 255, 0))) {
					if (enterColor && !enterWhite) {
						enterWhite = true;
						col = pixel;
						//points.push_back(Point(i, j));
					}
					if (enterColor && enterWhite && !exitColor) {
						pointsGreen.push_back(Point(i, j));
					}
				}
				if (check_pixel_color(pixel, Vec3b(0, 255, 255))) {
					if (enterColor && !enterWhite) {
						enterWhite = true;
						col = pixel;
						//points.push_back(Point(i, j));
					}
					if (enterColor && enterWhite && !exitColor) {
						pointsYellow.push_back(Point(i, j));
					}
				}
				if (check_pixel_color(pixel, Vec3b(0, 165, 255))) {
					if (enterColor && !enterWhite) {
						enterWhite = true;
						col = pixel;
						//points.push_back(Point(i, j));
					}
					if (enterColor && enterWhite && !exitColor) {
						pointsOrange.push_back(Point(i, j));
					}
				}
				
			
			}

			if (!found) {
				
				dst.at<Vec3b>(i, j)[0] = 255;
				dst.at<Vec3b>(i, j)[1] = 0;
				dst.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				//printf("%d\n", found);
				
				dst.at<Vec3b>(i, j)[0] = b;
				dst.at<Vec3b>(i, j)[1] = g;
				dst.at<Vec3b>(i, j)[2] = r;
			}

			
		}
		if (exitColor) {
			for (int p = 0; p < points.size(); p++) {	
				//if (col[0] > 150 && col[0] > 150 && col[0] > 150) {
					dst.at<Vec3b>(points[p].x, points[p].y)[0] = 255;
					dst.at<Vec3b>(points[p].x, points[p].y)[1] = 255;
					dst.at<Vec3b>(points[p].x, points[p].y)[2] = 255;
				//}
				
			}
			for (int p = 0; p < pointsBlack.size(); p++) {
				//if (col[0] > 150 && col[0] > 150 && col[0] > 150) {
				dst.at<Vec3b>(pointsBlack[p].x, pointsBlack[p].y)[0] = 0;
				dst.at<Vec3b>(pointsBlack[p].x, pointsBlack[p].y)[1] = 0;
				dst.at<Vec3b>(pointsBlack[p].x, pointsBlack[p].y)[2] = 0;
				//}

			}
			for (int p = 0; p < pointsGreen.size(); p++) {
				//if (col[0] > 150 && col[0] > 150 && col[0] > 150) {
				dst.at<Vec3b>(pointsGreen[p].x, pointsGreen[p].y)[0] = 0;
				dst.at<Vec3b>(pointsGreen[p].x, pointsGreen[p].y)[1] = 255;
				dst.at<Vec3b>(pointsGreen[p].x, pointsGreen[p].y)[2] = 0;
				//}

			}
			for (int p = 0; p < pointsYellow.size(); p++) {
				//if (col[0] > 150 && col[0] > 150 && col[0] > 150) {
				dst.at<Vec3b>(pointsYellow[p].x, pointsYellow[p].y)[0] = 0;
				dst.at<Vec3b>(pointsYellow[p].x, pointsYellow[p].y)[1] = 255;
				dst.at<Vec3b>(pointsYellow[p].x, pointsYellow[p].y)[2] = 255;
				//}

			}
			for (int p = 0; p < pointsOrange.size(); p++) {
				//if (col[0] > 150 && col[0] > 150 && col[0] > 150) {
				dst.at<Vec3b>(pointsOrange[p].x, pointsOrange[p].y)[0] = 0;
				dst.at<Vec3b>(pointsOrange[p].x, pointsOrange[p].y)[1] = 165;
				dst.at<Vec3b>(pointsOrange[p].x, pointsOrange[p].y)[2] = 255;
				//}

			}
			points.clear();
			pointsBlack.clear();
			pointsGreen.clear();
			pointsYellow.clear();
			pointsOrange.clear();
		}
	}

	imshow("transformed_image", img);
	return dst;

}


bool indentify_access_restricted(Mat original,int label, Mat img, Mat labels) {

	printf("Identify object label %d\n", label);
	float area = 0;
	float white_area = 0;
	float perimeter = 0;
	float white_perimeter = 0;
	std::vector<Point>contour_points;
	int rowMin=INT_MAX, colMin=INT_MAX, rowMax=INT_MIN, colMax=INT_MIN;

	Mat contour_image(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			contour_image.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}

	int iDirection[] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int jDireciton[] = { 0, -1, 0, 1, -1, 1, 1, -1 };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (labels.at<uchar>(i, j) == label) {
				if (i < rowMin) {
					rowMin = i;
				}
				if (j < colMin) {
					colMin = j;
				}
				if (j > colMax) {
					colMax = j;
				}
				if (i > rowMax) {
					rowMax = i;
				}
				if (!check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {

					printf("%d %d %d %d\n", label, img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2]);
					return false;
				}
				else
				{
					//update area
					area++;
					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
						white_area++;
					}

					//update perimeter
					bool isOnTheContourRed = false;
					bool isOnTheContourWhite = false;

					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255))) {
						for (int k = 0; k < 8; k++)
						{
							if (isInside(img, i +iDirection[k], j + jDireciton[k])) {
								Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
								if (check_pixel_color(n1, Vec3b(255,0,255))) {
									contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
									isOnTheContourRed = true;
									contour_points.push_back(Point(i, j));
									break;
								}
							}
						}
						if (isOnTheContourRed) {
							perimeter++;
						}

					}
					else {
						if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
							for (int k = 0; k < 8; k++)
							{
								if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
									Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
									if (check_pixel_color(n1, Vec3b(0, 0, 255))) {
										contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
										isOnTheContourWhite = true;
										contour_points.push_back(Point(i, j));
										break;
									}
								}
							}
							if (isOnTheContourWhite) {
								white_perimeter++;
							}
						}

						
					}

				}
			}
		}
	}
	/*for (int i = 0; i < contour_points.size(); i++) {
		for (int k = 0; k < 8; k++)
		{
			int x_neighb = contour_points[i].x + iDirection[k];
			int y_neighb = contour_points[i].y + jDireciton[k];
			if (isInside(img, x_neighb, y_neighb)) {
				if (std::find(contour_points.begin(), contour_points.end(), Point(x_neighb, y_neighb)) != contour_points.end()) {
					int difx = abs(contour_points[i].x - x_neighb);
					int dify = abs(contour_points[i].y - y_neighb);
					if ((difx == 0 && dify == 1) || (difx == 1 && dify == 0)) {
						perimeter++;
					}
					else {
						perimeter += sqrt(2);
					}

				}
			}
		}
	}*/

	float ratio_red =  4 * PI * area/pow(PI/4 * perimeter, 2);
	float ratio_white = 4 * PI * white_area/pow((PI/4) * white_perimeter, 2);
	
	printf("area_red: %f\n", area);
	printf("perimetru_red: %f\n", perimeter);

	printf("area_white: %f\n", white_area);
	printf("perimetru_white: %f\n", white_perimeter);

	printf("ratio_red: %f\n", ratio_red);
	printf("ratio_white: %f\n", ratio_white);
	

	if (ratio_red > 0.98 && ratio_red < 1.06 && ratio_white > 0.98 && ratio_white < 1.05) {
		cv::rectangle(original, Point(colMin+1,rowMin+1), Point( colMax+1,rowMax+1), Scalar(255, 0, 0), 5);
		imshow("contour_image", contour_image);
		return true;
	}
	return false;
}


bool indentify_no_entry(Mat original,int label, Mat img, Mat labels) {

	printf("Identify object label %d\n", label);
	float area = 0;
	float white_area = 0;
	float perimeter = 0;
	float white_perimeter = 0;
	std::vector<Point>contour_points;

	int rowMin = INT_MAX, colMin = INT_MAX, rowMax = INT_MIN, colMax = INT_MIN;

	int iDirection[] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int jDireciton[] = { 0, -1, 0, 1, -1, 1, 1, -1 };

	Mat contour_image(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			contour_image.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (labels.at<uchar>(i, j) == label) {
				if (i < rowMin) {
					rowMin = i;
				}
				if (j < colMin) {
					colMin = j;
				}
				if (j > colMax) {
					colMax = j;
				}
				if (i > rowMax) {
					rowMax = i;
				}
				if (!check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {

					printf("%d %d %d %d\n", label, img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2]);
					return false;
				}
				else
				{
					//update area
					area++;
					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
						white_area++;
					}

					//update perimeter
					bool isOnTheContourRed = false;
					bool isOnTheContourWhite = false;

					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255))) {
						for (int k = 0; k < 8; k++)
						{
							if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
								Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
								if (check_pixel_color(n1, Vec3b(255, 0, 255))) {
									contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
									isOnTheContourRed = true;
									contour_points.push_back(Point(i, j));
									break;
								}
							}
						}
						if (isOnTheContourRed) {
							perimeter++;
						}

					}
					else {
						
						if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
							for (int k = 0; k < 8; k++)
							{
								if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
									Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
									if (check_pixel_color(n1, Vec3b(0, 0, 255))) {
										contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
										isOnTheContourWhite = true;
										contour_points.push_back(Point(i, j));
										break;
									}
								}
							}
							if (isOnTheContourWhite) {
								white_perimeter++;
							}
						}


					}

				}
			}
		}
	}
	

	float ratio_red = 4 * PI * area / pow(PI / 4 * perimeter, 2);
	float ratio_white = 4 * PI * white_area / pow((PI / 4) * white_perimeter, 2);

	printf("area_red: %f\n", area);
	printf("perimetru: %f\n", perimeter);

	printf("area_white: %f\n", white_area);
	printf("perimetru_white: %f\n", white_perimeter);

	printf("ratio_red: %f\n", ratio_red);
	printf("ratio_white: %f\n", ratio_white);

	if (ratio_red > 0.98 && ratio_red < 1.05 && ratio_white > 0.8 && ratio_white < 0.95) {
		cv::rectangle(original, Point(colMin + 1, rowMin + 1), Point(colMax + 1, rowMax + 1), Scalar(255, 0, 0), 5);
		imshow("contour_image_circle", contour_image);
		return true;
	}
	
	return false;
}


bool indentify_danger_warning(Mat original, int label, Mat img, Mat labels) {
	int iDirection[] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int jDireciton[] = { 0, -1, 0, 1, -1, 1, 1, -1 };

	Mat red(img.rows, img.cols, CV_8UC1);
	Mat white(img.rows, img.cols, CV_8UC1);

	int rowMin = INT_MAX, colMin = INT_MAX, rowMax = INT_MIN, colMax = INT_MIN;

	Mat contour_image(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			red.at<uchar>(i, j) = 255;
			white.at<uchar>(i, j) = 255;
			contour_image.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}

	
	float area_black = 0;
	float perimeter_black = 0;

	printf("Identify object label %d\n", label);


		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (labels.at<uchar>(i, j) == label) {
					if (i < rowMin) {
						rowMin = i;
					}
					if (j < colMin) {
						colMin = j;
					}
					if (j > colMax) {
						colMax = j;
					}
					if (i > rowMax) {
						rowMax = i;
					}
					if (!check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 0))) {

						printf("%d %d %d %d\n", label, img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2]);
						return false;
					}
					else
					{

						if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255))) {
							for (int k = 0; k < 8; k++)
							{
								if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
									Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
									if (check_pixel_color(n1, Vec3b(255, 0, 255))) {
										red.at<uchar>(i, j) = 0;
										contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
										break;
									}
								}
							}
						
						}
						else {

							if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
								for (int k = 0; k < 8; k++)
								{
									if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
										Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
										if (check_pixel_color(n1, Vec3b(0, 0, 255))) {
											white.at<uchar>(i, j) = 0;
											contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
											break;
										}
									}
								}
							
							}
							else {
								if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 0))) {
									area_black++;
									bool isOnContourBlack = false;
									for (int k = 0; k < 8; k++)
									{
										if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
											Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
											if (check_pixel_color(n1, Vec3b(255, 255, 255))) {
												isOnContourBlack = true;
												contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
												break;
											}
										}
									}
									if (isOnContourBlack) {
										perimeter_black++;
									}

								}
							}
						}

					}
				}
			}
		}

	

	Mat cannyRed,cannyWhite;
	double k = 0.4;
	int pH = 100;
	int pL = 200;
	Canny(red, cannyRed, pL, pH, 5);
	Canny(white, cannyWhite, pL, pH, 5);
	
	//get corners

	std::vector<Point> cornersRed;
	std::vector<Point> cornersWhite;

	goodFeaturesToTrack(cannyRed, cornersRed, 4, 0.5, 50);

	for (int i = 0; i < cornersRed.size(); i++) {
		Point corner = cornersRed[i];

		circle(red, corner, 10, 0);
	}

	goodFeaturesToTrack(cannyWhite, cornersWhite, 4, 0.5, 50);

	for (int i = 0; i < cornersWhite.size(); i++) {
		Point corner = cornersWhite[i];

		circle(white, corner, 10, 0);
	}

	float ratio_black = 4 * PI * area_black / pow(PI / 4 * perimeter_black, 2);
	

	printf("area_black: %f\n", area_black);
	printf("perimetru_black: %f\n", perimeter_black);

	printf("ratio_black: %f\n", ratio_black);


	printf("Nr corners %d\n", cornersRed.size()+cornersWhite.size());


	if (cornersRed.size() == 3 && cornersWhite.size() == 3 && ratio_black > 0.98 && ratio_black < 1.08) {
		cv::rectangle(original, Point(colMin + 1, rowMin + 1), Point(colMax + 1, rowMax + 1), Scalar(255, 0, 0), 5);
		//imshow("Risc ridicat de accidente", original);
		//imshow("corners", img);

		imshow("red", red);
		imshow("white", white);
		imshow("contour_image_triangle", contour_image);
		return true;
	}
	

	return false;
}



bool indentify_semaphore(Mat original,int label, Mat img, Mat labels) {
	int iDirection[] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int jDireciton[] = { 0, -1, 0, 1, -1, 1, 1, -1 };

	Mat red(img.rows, img.cols, CV_8UC1);
	Mat white(img.rows, img.cols, CV_8UC1);

	int rowMin = INT_MAX, colMin = INT_MAX, rowMax = INT_MIN, colMax = INT_MIN;

	Mat contour_image(img.rows, img.cols, CV_8UC3);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			red.at<uchar>(i, j) = 255;
			white.at<uchar>(i, j) = 255;
			contour_image.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}

	float area_orange = 0;
	float perimeter_orange = 0;

	float area_green = 0;
	float perimeter_green = 0;

	float area_yellow = 0;
	float perimeter_yellow = 0;

	printf("Identify object label %d\n", label);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (labels.at<uchar>(i, j) == label) {
				if (i < rowMin) {
					rowMin = i;
				}
				if (j < colMin) {
					colMin = j;
				}
				if (j > colMax) {
					colMax = j;
				}
				if (i > rowMax) {
					rowMax = i;
				}
				if (!check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 0)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 255, 0)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 255, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 165, 255))) {

					printf("%d %d %d %d\n", label, img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2]);
					return false;
				}
				else
				{

					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255))) {
						for (int k = 0; k < 8; k++)
						{
							if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
								Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
								if (check_pixel_color(n1, Vec3b(255, 0, 255))) {
									red.at<uchar>(i, j) = 0;
									contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
									break;
								}
							}
						}

					}
					else {

						if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
							for (int k = 0; k < 8; k++)
							{
								if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
									Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
									if (check_pixel_color(n1, Vec3b(0, 0, 255))) {
										white.at<uchar>(i, j) = 0;
										contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
										break;
									}
								}
							}

						}
						else {
							if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 255, 0))) {
								area_green++;
								bool isOnContourGreen = false;
								for (int k = 0; k < 8; k++)
								{
									if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
										Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
										if (check_pixel_color(n1, Vec3b(255, 255, 255))) {
											isOnContourGreen = true;
											contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
											break;
										}
									}
								}
								if (isOnContourGreen) {
									perimeter_green++;
								}

							}
							else {
								if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 255, 255))) {
									area_yellow++;
									bool isOnContourYellow = false;
									for (int k = 0; k < 8; k++)
									{
										if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
											Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
											if (check_pixel_color(n1, Vec3b(255, 255, 255))) {
												isOnContourYellow = true;
												contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
												break;
											}
										}
									}
									if (isOnContourYellow) {
										perimeter_yellow++;
									}

								}
								else {
									if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 165, 255))) {
										area_orange++;
										bool isOnContourOrange = false;
										for (int k = 0; k < 8; k++)
										{
											if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
												Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
												if (check_pixel_color(n1, Vec3b(255, 255, 255))) {
													isOnContourOrange = true;
													contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
													break;
												}
											}
										}
										if (isOnContourOrange) {
											perimeter_orange++;
										}

									}
								}

							}
						}
					}

				}
			}
		}
	}



	Mat cannyRed, cannyWhite;
	double k = 0.4;
	int pH = 100;
	int pL = 200;
	Canny(red, cannyRed, pL, pH, 5);
	Canny(white, cannyWhite, pL, pH, 5);

	//get corners

	std::vector<Point> cornersRed;
	std::vector<Point> cornersWhite;

	goodFeaturesToTrack(cannyRed, cornersRed, 4, 0.5, 50);

	for (int i = 0; i < cornersRed.size(); i++) {
		Point corner = cornersRed[i];

		circle(red, corner, 10, 0);
	}

	goodFeaturesToTrack(cannyWhite, cornersWhite, 4, 0.5, 50);

	for (int i = 0; i < cornersWhite.size(); i++) {
		Point corner = cornersWhite[i];

		circle(white, corner, 10, 0);
	}

	float ratio_orange = 4 * PI * area_orange / pow(PI / 4 * perimeter_orange, 2);
	float ratio_yellow = 4 * PI * area_yellow / pow(PI / 4 * perimeter_yellow, 2);
	float ratio_green = 4 * PI * area_green / pow(PI / 4 * perimeter_green, 2);

	
	printf("area_orange: %f\n", area_orange);
	printf("perimetru_orange: %f\n", perimeter_orange);
	printf("ratio_orange: %f\n", ratio_orange);

	printf("area_yellow: %f\n", area_yellow);
	printf("perimetru_yellow: %f\n", perimeter_yellow);
	printf("ratio_yellow: %f\n", ratio_yellow);

	printf("area_green: %f\n", area_green);
	printf("perimetru_green: %f\n", perimeter_green);
	printf("ratio_green: %f\n", ratio_green);

	printf("Nr corners %d\n", cornersRed.size() + cornersWhite.size());
	//imshow("corners", img);



	if (cornersRed.size() == 3 && cornersWhite.size() == 3 && ratio_green > 0.98 && ratio_green < 1.15 && ratio_yellow > 0.98 && ratio_yellow < 1.15 && ratio_orange > 0.98 && ratio_orange < 1.15) {
		cv::rectangle(original, Point(colMin + 1, rowMin + 1), Point(colMax + 1, rowMax + 1), Scalar(255, 0, 0), 5);
		imshow("red", red);
		imshow("white", white);
		imshow("contour_semapthore", contour_image);
		return true;
	}
		

	return false;
}


bool indentify_cedeaza(Mat original, int label, Mat img, Mat labels) {
	int iDirection[] = { -1, 0, 1, 0, -1, -1, 1, 1 };
	int jDireciton[] = { 0, -1, 0, 1, -1, 1, 1, -1 };

	Mat red(img.rows, img.cols, CV_8UC1);
	Mat white(img.rows, img.cols, CV_8UC1);

	int rowMin = INT_MAX, colMin = INT_MAX, rowMax = INT_MIN, colMax = INT_MIN;

	Mat contour_image(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			red.at<uchar>(i, j) = 255;
			white.at<uchar>(i, j) = 255;
			contour_image.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
	}

	float area_black = 0;
	float perimeter_black = 0;

	printf("Identify object label %d\n", label);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (labels.at<uchar>(i, j) == label) {
				if (i < rowMin) {
					rowMin = i;
				}
				if (j < colMin) {
					colMin = j;
				}
				if (j > colMax) {
					colMax = j;
				}
				if (i > rowMax) {
					rowMax = i;
				}
				if (!check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255)) && !check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255)) ) {

					printf("%d %d %d %d\n", label, img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2]);
					return false;
				}
				else
				{

					if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(0, 0, 255))) {
						for (int k = 0; k < 8; k++)
						{
							if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
								Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
								if (check_pixel_color(n1, Vec3b(255, 0, 255))) {
									red.at<uchar>(i, j) = 0;
									contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
									break;
								}
							}
						}

					}
					else {

						if (check_pixel_color(img.at<Vec3b>(i, j), Vec3b(255, 255, 255))) {
							for (int k = 0; k < 8; k++)
							{
								if (isInside(img, i + iDirection[k], j + jDireciton[k])) {
									Vec3b n1 = img.at<Vec3b>(i + iDirection[k], j + jDireciton[k]);
									if (check_pixel_color(n1, Vec3b(0, 0, 255))) {
										white.at<uchar>(i, j) = 0;
										contour_image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
										break;
									}
								}
							}

						}
					
					}

				}
			}
		}
	}



	Mat cannyRed, cannyWhite;
	double k = 0.4;
	int pH = 100;
	int pL = 200;
	Canny(red, cannyRed, pL, pH, 5);
	Canny(white, cannyWhite, pL, pH, 5);

	//get corners

	std::vector<Point> cornersRed;
	std::vector<Point> cornersWhite;

	goodFeaturesToTrack(cannyRed, cornersRed, 4, 0.5, 50);

	int minRow = INT_MAX;
	bool cornersPropertyTrue = false;
	
	for (int i = 0; i < cornersRed.size(); i++) {
		Point corner = cornersRed[i];

		if (corner.y == minRow) {
			cornersPropertyTrue = true;
		}
		if (corner.y < minRow) {
			minRow = corner.y;
		}
	
		circle(red, corner, 10, 0);
	}

	goodFeaturesToTrack(cannyWhite, cornersWhite, 4, 0.5, 50);

	for (int i = 0; i < cornersWhite.size(); i++) {
		Point corner = cornersWhite[i];

		circle(white, corner, 10, 0);
	}

	


	printf("Nr corners %d\n", cornersRed.size() + cornersWhite.size());
	//imshow("corners", img);

	


	if (cornersRed.size() == 3 && cornersWhite.size() == 3 && cornersPropertyTrue) {
		cv::rectangle(original, Point(colMin + 1, rowMin + 1), Point(colMax + 1, rowMax + 1), Scalar(255, 0, 0), 5);
		imshow("red", red);
		imshow("white", white);
		imshow("contour_image_cedeaza", contour_image);
		return true;
	}


	return false;
}

void identify_traffic_signs(Mat original,Mat img) {

	Mat labels = label_objects(img);
	int nr_signs_detected = 0;

	std::vector<char *> signs;

	int max_label = INT_MIN;
	//count number of labels
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {

			if (labels.at<uchar>(i, j) > max_label) {
				max_label = labels.at<uchar>(i, j);

			}
		}
	}

	//printf("Number of signs:%d\n", max_label);

	//identify objects in image

	for (int label = 1; label <= max_label; label++) {
		
		if (indentify_access_restricted(original,label, img, labels)) {
			printf("Access restricted detected, object label %d\n",label);

			nr_signs_detected++;
			signs.push_back("Circulatia interzisa in ambele sensuri\n");
		}
		else {
			
			if (indentify_no_entry(original,label, img, labels)) {
				printf("No entry detected, object label %d\n", label);

				nr_signs_detected++;
				signs.push_back("Acces interzis\n");
			}
			else {
				if (indentify_danger_warning(original,label, img, labels))
				{
					printf("Danger warning sign detected, object label %d\n", label);

					nr_signs_detected++;
					signs.push_back("Zona cu risc ridicat de accidente\n");
				}
				else {
					if (indentify_semaphore(original,label, img, labels))
					{
						printf("Semaphore sign detected, object label %d\n", label);

						nr_signs_detected++;
						signs.push_back("Atentie, semafor!\n");
					}
					else {
						if (indentify_cedeaza(original, label, img, labels))
						{
							printf("Cedeaza sign detected, object label %d\n", label);

							nr_signs_detected++;
							signs.push_back("Cedeaza trecerea\n");
						}
					}
				}
			}
		}
		
	}
	imshow("TRAFFIC SIGNS DETECTED", original);

	printf("Total number of signs detected: %d\n", nr_signs_detected);

	for (int i = 0; i < nr_signs_detected; i++) {
		printf(signs[i]);
	}

}

int main()
{
	int op;
	Vec3b* colors = generate_signs_colors();
	Mat intro = imread("TrafficSigns/intro.png", IMREAD_COLOR);
	imshow("Intro", intro);
	waitKey(0);
	
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Transform image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
			{
				Mat img;
				char fname[MAX_PATH];
				if (openFileDlg(fname))
				{
					img = imread(fname, IMREAD_COLOR);
				}

				Mat dst = transform_image(img, colors);
				imshow("img", img);
				imshow("transfomed image", dst);
				identify_traffic_signs(img,dst);
				waitKey(0);
				break;

			}
		}
	}
	while (op!=0);
	return 0;
}