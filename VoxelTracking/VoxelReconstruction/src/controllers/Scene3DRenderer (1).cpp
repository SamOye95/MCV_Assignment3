/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <stddef.h>
#include <string>

#include "../utilities/General.h"

using namespace std;
using namespace cv;
bool AUTOPARMS = true;

namespace nl_uu_science_gmt
{
	void calculateDifferenceHSV(const Mat& imageA, const Mat& imageB, Mat& imageOut);

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 11;
	const int S = 22;
	const int V = 38;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	const int E = 2;
	const int D = 3;
	m_erosion_size = E;
	m_dilation_size = D;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);

	createTrackbar("Erosion kernel size:", VIDEO_WINDOW, &m_erosion_size, max_kernel_size);
	createTrackbar("Dilation kernel size", VIDEO_WINDOW, &m_dilation_size, max_kernel_size);

	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}
	return true;
}


Mat& applyMask(Mat input, Mat mask) 
{
	Mat newImage;
	input.copyTo(newImage, mask);
	return newImage;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
	Camera* camera)
{
	//Mat object to grab the frames from the video
	Mat image = camera->getFrame();
	GaussianBlur(image, image, Size(5, 5), 3, 0, BORDER_DEFAULT);

	assert(!camera->getFrame().empty());
	Mat hsv_image;

	cvtColor(image, hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	//initialising the global  color thresholds
	Scalar lower = Scalar (5, 15, 35);
	Scalar higher = Scalar(255, 255, 255);

	imshow("Bilateral Blur", hsv_image);


	Mat tmp, foreground, background;
	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis
	
	if (!autoParameters)
	{

		


		// Background subtraction H
		absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
		threshold(tmp, foreground, m_h_threshold, 255, CV_THRESH_BINARY);
		//inRange(tmp, lower, higher, foreground);
		
		imshow("Mask", foreground);

		// Background subtraction S
		absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
		
		threshold(tmp, foreground, m_s_threshold, 255, CV_THRESH_BINARY);
		//inRange(tmp, lower, higher, foreground);
		bitwise_and(foreground, tmp, foreground);

		// Background subtraction V
		absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
		adaptiveThreshold(tmp, foreground, ADAPTIVE_THRESH_GAUSSIAN_C, 255, CV_THRESH_BINARY, 5 , 2);
		//inRange(tmp, lower, higher, foreground);
		bitwise_or(foreground, tmp, foreground);
	
	
		Mat erosion_kernel = getStructuringElement(MORPH_RECT,
			Size(2 * m_erosion_size + 1, 2 * m_erosion_size + 1),
			Point(m_erosion_size, m_erosion_size));

		Mat dilation_kernel = getStructuringElement(MORPH_RECT,
			Size(2 * m_dilation_size + 1, 2 * m_dilation_size + 1),
			Point(m_dilation_size, m_dilation_size));


		erode(foreground, foreground, erosion_kernel);
		dilate(foreground, foreground, dilation_kernel);
		erode(foreground, foreground, erosion_kernel);

	
	
	}
	else
	{
		merge(camera->getBgHsvChannels(), background);

		Mat difference(background.rows, background.cols, CV_8UC1);
		calculateDifferenceHSV(hsv_image, background, difference);

		double thresh = threshold(difference, foreground, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		threshold(difference, foreground, thresh * 0.5, 255, CV_THRESH_BINARY);

		bitwise_or(foreground, difference, foreground);

		threshold(foreground, foreground, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// Improve the foreground image
		Mat erosion_kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
		Mat dilation_kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));

		erode(foreground, foreground, erosion_kernel);
		dilate(foreground, foreground, dilation_kernel);
		GaussianBlur(foreground, foreground, Size(5, 5), 3, 0, BORDER_DEFAULT);
	}
	// Improve the foreground image
	camera->setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}
/**
* Calculate HSV image differences
*/
void calculateDifferenceHSV(const Mat& imageA, const Mat& imageB, Mat& imageOut)
{
	// create input and output image iterators
	MatConstIterator_<Vec3b> iterA = imageA.begin<Vec3b>();
	MatConstIterator_<Vec3b> iterB = imageB.begin<Vec3b>();
	MatIterator_<uchar> iterOut = imageOut.begin<uchar>();

	// iterate through every pixel
	while (iterOut != imageOut.end<uchar>())
	{
		// calculate input pixel difference
		Vec3i pixelA(*iterA), pixelB(*iterB);
		Vec3i difference = pixelA - pixelB;

		// calculate hue
		double hueA = (double)pixelA[0] / 180.0;
		double hueB = (double)pixelB[0] / 180.0;

		// calculate saturation
		double saturationA = (double)pixelA[1] / 255.0;
		double saturationB = (double)pixelB[1] / 255.0;

		// pixel output
		double scaler = pow(min(min(saturationA, saturationB) * abs(hueA - hueB) + abs(saturationA - saturationB), 1.0), 0.3);
		*iterOut = (uchar)(sqrt(pow(difference[2], 2) + pow(difference[1], 2) + pow(difference[0], 2)) * scaler);
		iterA++, iterB++, iterOut++;
	}
}
} /* namespace nl_uu_science_gmt */
