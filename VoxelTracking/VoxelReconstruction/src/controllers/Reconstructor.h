/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Camera.h"


 //PolyVox libraries
#include "PolyVoxCore/MaterialDensityPair.h"
#include "PolyVoxCore/MarchingCubesSurfaceExtractor.h"
#include "PolyVoxCore/SurfaceMesh.h"
#include "PolyVoxCore/SimpleVolume.h"
#include "../../ColorModel.h"

using namespace PolyVox;

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */


	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Scalar color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
		int label;									// Voxel label
	};

private:
	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;						// Step size (space between voxels)
	const int m_width;
	const int m_clusters;					// Number of clusters

	std::vector<cv::Point2f> m_clusterCenters;	// The center point of the Cluster, for drawing the path
	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels

	SurfaceMesh<PositionMaterialNormal> m_mesh;	// PolyVox surface for drawing mesh


	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}

	// returns cluster count
	int getClusterCount() const
	{
		return m_clusters;
	}

	// returns scene length
	int getWidth() const
	{
		return m_width;
	}
	// returns PolyVox mesh drawing surface
	const SurfaceMesh<PositionMaterialNormal>& getMesh() const
	{
		return m_mesh;
	}

	// tracks the center points
	std::vector<std::vector<cv::Point2f>> trackCenters;
	// labels the clustering
	void labelClusters(bool isFirstFrame);

		// returns space between voxels
	int getStep() const
	{
		return m_step;
	}

	// create and save color models to file
	void createAndSaveColorModels();

	// create color models
	void createColorModels(std::vector<ColorModel>& models);	
	
	// assign labels based on color models
	void assignLabels(std::vector<int>& labels);				

	// if the voxels are already clustered or not
	bool isClustered = false;
	bool drawMesh = false;


};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
