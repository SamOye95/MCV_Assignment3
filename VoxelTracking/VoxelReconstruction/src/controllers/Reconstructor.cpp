/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"



#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048),
				// added user function
				//m_width(7168),
				m_step(32),

				// Redefining the scene width for clusters
				m_width(6144),
				m_clusters(4),
				m_clusterCenters(4)
				
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */






void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(auto) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
* Initialize label for each visible voxels
* use kmeans for clustering and to calculate cluster center to draw the tracks
*/
void Reconstructor::labelClusters(bool isFirstFrame)
{
	vector<int> labels(m_visible_voxels.size());
	vector<Point2f> points;
	Mat centers;

	for (int i = 0; i < m_visible_voxels.size(); i++)
	{
		//projection of the points by removing the z-axis
		points.push_back(Point(m_visible_voxels[i]->x, m_visible_voxels[i]->y));
		if (!isFirstFrame)
		{
			labels[i] = m_visible_voxels[i]->label;
		}
	}

	// clustering the voxels based on the cluster count
	if (isFirstFrame)
	{
		// use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
		kmeans(points, m_clusters, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			4, KMEANS_PP_CENTERS, centers);

		// fill the label based on kmeans calculation
		for (int i = 0; i < m_visible_voxels.size(); i++)
		{
			m_visible_voxels[i]->label = labels[i];
		}

	}
	else
	{
		// use the user-supplied labels instead of computing them from the initial centers
		kmeans(points, m_clusters, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			1, KMEANS_USE_INITIAL_LABELS, centers);
	}

	// set path tracker centers by calculating the cluster centers
	for (int i = 0; i < m_clusters; i++)
	{
		m_clusterCenters[i] = Point2i(centers.at<float>(i, 0), centers.at<float>(i, 1));
	}

	trackCenters.push_back(m_clusterCenters);
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(auto) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{

			int minLabel = 0;
			float minDistance = norm(m_clusterCenters[0] - Point2f(voxel->x, voxel->y));

			for (int i = 0; i < m_clusters; i++)
			{
				float distance = norm(m_clusterCenters[i] - Point2f(voxel->x, voxel->y));
				if (distance < minDistance)
				{
					minLabel = i;
					minDistance = distance;
				}
			}
			voxel->label = minLabel;

#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	///added user function
	if (!drawMesh)
	{
		//Fill volume data by looping through all visible voxels
		int size_h = m_height / m_step;
		int size_w = m_width / m_step;
		SimpleVolume<uint8_t> volData(Region(Vector3DInt32(0, 0, 0), Vector3DInt32(size_w, size_w, size_h)));
		for (size_t v = 0; v < visible_voxels.size(); v++)
		{
			volData.setVoxelAt(visible_voxels[v]->x / m_step + size_w / 2, visible_voxels[v]->y / m_step + size_w / 2, visible_voxels[v]->z / m_step + 1, 255);
		}

		//Convert volume data to triangle mesh
		MarchingCubesSurfaceExtractor<SimpleVolume<uint8_t>> surfaceExtractor(&volData, volData.getEnclosingRegion(), &m_mesh);
		surfaceExtractor.execute();
	}
}


} /* namespace nl_uu_science_gmt */
