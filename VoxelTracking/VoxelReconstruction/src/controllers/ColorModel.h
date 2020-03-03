#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <xtgmath.h>
#include <fstream>
#include <ios>

typedef unsigned char uchar;

/**		Color model class, representing a histogram based, binned color model.
	*	Default binCount is 8. Adapts to different bin counts.
	*	This model does not care which color representation (RGB, HSV, etc.) is fed into it.
	*/
class ColorModel {
public:
	ColorModel(int binCount = 8)
	{
		hist = cv::Mat::zeros(3, 256 / binCount, CV_32SC1);
		this->binSize = 256 / binCount;
		this->binCount = binCount;
	}

	// Adds a 3-parameter color representation point to the histogram. (RGB, HSV, etc.)
	void addPoint(int r, int g, int b)
	{
		hist.at<int>(0, floor(r / binCount)) += 1;
		hist.at<int>(1, floor(g / binCount)) += 1;
		hist.at<int>(2, floor(b / binCount)) += 1;
	}

	// Compares the color model to another color model 
	// using the total distance of histagram bin values
	int compare(ColorModel& model2)
	{
		int sumDiff = 0;
		for (int i = 0; i < hist.rows; i++)
		{
			const int* hist_i = hist.ptr<const int>(i);
			const int* hist2_i = model2.hist.ptr<const int>(i);
			for (int j = 0; j < hist.cols; j++)
			{
				sumDiff += abs(hist_i[j] - hist2_i[j]);
			}
		}
		return sumDiff;
	}

	// Output to file
	void save(const char* file)
	{
		std::fstream outFile;

		//Overwrite the current color model by opening with trunc flag.
		outFile.open(file, std::ios::out | std::ios::trunc);

		if (!outFile.is_open())
		{
			printf("File could not be opened.");
		}
		for (int i = 0; i < hist.rows; i++)
		{
			for (int j = 0; j < hist.cols; j++)
			{
				outFile << hist.at<int>(i, j) << " ";
				// eg:
				// outFile << "i :" << i << ", j :" << j << " = " << hist.at<int>(i, j) << " \n"; 
			}
		}
		outFile << "/n";
		outFile.flush();
		outFile.close();
	}

	// Read from disk 
	void load(const char* file)
	{
		std::ifstream inFile(file);
		for (int i = 0; i < hist.rows; i++)
		{
			int* hist_i = hist.ptr<int>(i);
			for (int j = 0; j < hist.cols; j++)
			{
				inFile >> hist_i[j];
			}
		}
		inFile.close();
	}

protected:
	cv::Mat hist;				// Histogram data
	int binSize, binCount;		// bin information
};