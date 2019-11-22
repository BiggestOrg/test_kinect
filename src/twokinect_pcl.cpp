// 标准库头文件
#include <iostream>
#include <fstream>
#include <string>
#include <vector> 
#include <map>

// OpenCV
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp> 
#include <opencv2\imgproc\types_c.h>
#include <opencv2\highgui\highgui_c.h>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>
#include <opencv2\core\eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// OpenNI
#include <OpenNI.h> 
#include <aruco/aruco.h>

#include <pcl/common/common_headers.h>      
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

using namespace std;
//using namespace cv;
using namespace openni;
using namespace Eigen;


class CDevice {
public:
	string			devicename;
	string			depthWindowName;
	string			colorWindowName;
	Device*			pDevice;
	VideoStream*	pDepthStream;
	VideoStream*	pColorStream;
	cv::Mat			K;
	cv::Mat			coeffs;
	aruco::CameraParameters camParam;


	CDevice(int idx, const char* uri, string deviceName)
	{
		devicename.assign(deviceName);
		if (idx == 0)
			devicename.append("_master");
		else
			devicename.append("_sub");
		depthWindowName = devicename + "_depth";
		colorWindowName = devicename + "_color";

		pDevice = new Device();
		pDevice->open(uri);

		// 深度数据
		pDepthStream = new VideoStream();
		pDepthStream->create(*pDevice, SENSOR_DEPTH);
		VideoMode modeDepth;
		modeDepth.setResolution(640, 480);
		modeDepth.setFps(30);
		modeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
		pDepthStream->setVideoMode(modeDepth);
		pDepthStream->start();
		//cv::namedWindow(depthWindowName, CV_WINDOW_AUTOSIZE);

		//color
		pColorStream = new VideoStream();
		pColorStream->create(*pDevice, openni::SENSOR_COLOR);
		VideoMode modeColor;
		modeColor.setResolution(640, 480);
		modeColor.setFps(30);
		modeColor.setPixelFormat(PIXEL_FORMAT_RGB888);
		pColorStream->setVideoMode(modeColor);

		if (pDevice->isImageRegistrationModeSupported(IMAGE_REGISTRATION_DEPTH_TO_COLOR))
		{
			pDevice->setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
		}
		pColorStream->start();
		cv::namedWindow(colorWindowName, CV_WINDOW_AUTOSIZE);
	}
};

aruco::Dictionary dic;
aruco::MarkerDetector MDetector;
std::map<int, aruco::MarkerPoseTracker> MTracker_master;
std::map<int, aruco::MarkerPoseTracker> MTracker_sub;
float MarkerSize = 0.16f;

//cv::Mat camera_matrix_color;
//cv::Mat dist_coeffs_color;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


template <typename DataType>
void writeToCSVfile(std::string name, Eigen::Array<DataType, -1, -1> matrix)
{
	std::ofstream file(name.c_str());
	file << matrix.format(CSVFormat);
}

void eigenTransform2cvRvecTvec(const Transform<double, 3, Affine> frame, cv::Vec3d &rvec, cv::Vec3d &tvec)
{
	Translation<double, 3> t(frame.translation());
	Quaternion<double> q(frame.linear());
	tvec = cv::Vec3d(t.x(), t.y(), t.z());
	cv::Mat rotM;
	cv::eigen2cv(q.toRotationMatrix(), rotM);
	cv::Rodrigues(rotM, rvec);
}

void drawAxis(cv::InputOutputArray _image, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
	cv::InputArray _rvec, cv::InputArray _tvec, float length)
{
	CV_Assert(_image.getMat().total() != 0 &&
		(_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
	CV_Assert(length > 0);

	// project axis points
	vector<cv::Point3f > axisPoints;
	axisPoints.push_back(cv::Point3f(0, 0, 0));
	axisPoints.push_back(cv::Point3f(length, 0, 0));
	axisPoints.push_back(cv::Point3f(0, length, 0));
	axisPoints.push_back(cv::Point3f(0, 0, length));
	vector< cv::Point2f > imagePoints;
	projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

	// draw axis lines
	line(_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
	line(_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
	line(_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}

//openni图像流转化成点云
bool getCloudXYZCoordinate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB,
	openni::VideoFrameRef  colorFrame,
	openni::VideoFrameRef  depthFrame,
	VideoStream *	depthStream) {

	openni::RGB888Pixel *pColor = (openni::RGB888Pixel*)colorFrame.getData();

	float fx, fy, fz;
	int i = 0;
	//以米为单位
	double fScale = 0.001;
	openni::DepthPixel *pDepthArray = (openni::DepthPixel*)depthFrame.getData();
	for (int y = 0; y < depthFrame.getHeight(); y++) {
		for (int x = 0; x < depthFrame.getWidth(); x++) {
			int idx = x + y * depthFrame.getWidth();
			const openni::DepthPixel rDepth = pDepthArray[idx];
			openni::CoordinateConverter::convertDepthToWorld(*depthStream, x, y, rDepth, &fx, &fy, &fz);
			//是否需要反转？
			fx = fx;
			fy = fy;
			cloud_XYZRGB->points[i].x = fx * fScale;
			cloud_XYZRGB->points[i].y = fy * fScale;
			cloud_XYZRGB->points[i].z = fz * fScale;
			cloud_XYZRGB->points[i].r = pColor[i].r;
			cloud_XYZRGB->points[i].g = pColor[i].g;
			cloud_XYZRGB->points[i].b = pColor[i].b;
			i++;
		}
	}
	return true;

}



int main(int argc, char **argv)
{
	OpenNI::initialize();

	// 设备信息  
	openni::Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);
	const uint32_t deviceCount = aDeviceList.getSize();
	cout << "连接 " << deviceCount << " 个体感设备." << endl;
	if (deviceCount != 2)
	{
		cout << "kinect devices not enough" << endl;
		return 0;
	}

	cout << "设备 " << 0 << endl;
	cout << "设备名： " << aDeviceList[0].getName() << endl;
	cout << "设备URI: " << aDeviceList[0].getUri() << endl;
	CDevice dev_master(0, aDeviceList[0].getUri(), aDeviceList[0].getName());

	double k_master[] = { 518.4867555811269, 0, 301.2378554028732, 0, 517.9021815902831, 244.2168611511136, 0, 0, 1 };
	//double c_master[] = { 0.2969676497213135, -1.282267901212675, -0.004452857706333246, -0.004453468933053606, 2.014987980086707 };
	double c_master[] = { 0.2969676497213135, -1.282267901212675, -0.004452857706333246, -0.004453468933053606 };
	dev_master.K = cv::Mat(3, 3, CV_64FC1, k_master);
	dev_master.coeffs = cv::Mat(1, 4, CV_64FC1, c_master);
	dev_master.camParam.setParams(dev_master.K, dev_master.coeffs, cv::Size(640, 480));
	//camera_matrix_color = cv::Mat(3, 3, CV_32F, &k_master);
	//dist_coeffs_color = cv::Mat::zeros(5, 1, CV_32F);
	//camera_matrix_color = dev_master.K.clone();
	//dist_coeffs_color = dev_master.coeffs.clone();

	cout << "设备 " << 1 << endl;
	cout << "设备名： " << aDeviceList[1].getName() << endl;
	cout << "设备URI: " << aDeviceList[1].getUri() << endl;
	CDevice dev_sub(1, aDeviceList[1].getUri(), aDeviceList[1].getName());

	double k_sub[] = { 521.5044785405424, 0, 325.6884223968248, 0, 522.9233149474519, 252.8945754365468, 0, 0, 1 };
	//double c_sub[] = { 0.2271155572064441, -0.7912825841629555, -0.005510686460386698, -0.006743974360193664, 0.908899981405665 };
	double c_sub[] = { 0.2271155572064441, -0.7912825841629555, -0.005510686460386698, -0.006743974360193664 };
	dev_sub.K = cv::Mat(3, 3, CV_64FC1, k_sub);
	dev_sub.coeffs = cv::Mat(1, 4, CV_64FC1, c_sub);
	dev_sub.camParam.setParams(dev_sub.K, dev_sub.coeffs, cv::Size(640, 480));

	dic = aruco::Dictionary::load("ARUCO_MIP_36h12");
	MDetector.setDictionary("ARUCO_MIP_36h12", 0.05f);
	MDetector.setDetectionMode(aruco::DM_NORMAL);

	VideoFrameRef vf_color_master;
	VideoFrameRef vf_color_sub;
	VideoFrameRef vf_depth_master;
	VideoFrameRef vf_depth_sub;
	//OpenCV image
	cv::Mat cv_img_master;
	cv::Mat cv_img_sub;
	cv::Mat cv_depth_master;
	cv::Mat cv_depth_sub;

	Eigen::Affine3d frame_master_marker;
	Eigen::Affine3d frame_sub_marker;
	Eigen::Affine3d frame_master_sub;

	bool poseEstimationOK_master = false;
	bool poseEstimationOK_sub = false;

	//master pcl云和可视化
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB_master(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud_XYZRGB_master->width = 640;
	cloud_XYZRGB_master->height = 480;
	cloud_XYZRGB_master->points.resize(cloud_XYZRGB_master->width * cloud_XYZRGB_master->height);
	pcl::visualization::PCLVisualizer::Ptr viewer_master(new pcl::visualization::PCLVisualizer("Viewer_master"));
	viewer_master->setCameraPosition(0, 0, -2, 0, 1, 0, 0);
	viewer_master->addCoordinateSystem(0.3);

	//sub pcl云和可视化
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_XYZRGB_sub(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud_XYZRGB_sub->width = 640;
	cloud_XYZRGB_sub->height = 480;
	cloud_XYZRGB_sub->points.resize(cloud_XYZRGB_sub->width * cloud_XYZRGB_sub->height);
	pcl::visualization::PCLVisualizer::Ptr viewer_sub(new pcl::visualization::PCLVisualizer("Viewer_sub"));
	viewer_sub->setCameraPosition(0, 0, -2, 0, 1, 0, 0);
	viewer_sub->addCoordinateSystem(0.3);

	pcl::visualization::PCLVisualizer::Ptr viewer_fusion(new pcl::visualization::PCLVisualizer("Viewer_fusion"));
	viewer_fusion->setCameraPosition(0, 0, -2, 0, 1, 0, 0);
	viewer_fusion->addCoordinateSystem(0.3);

	while (true)
	{
		if (dev_master.pColorStream->readFrame(&vf_color_master) == STATUS_OK &&
			dev_master.pDepthStream->readFrame(&vf_depth_master) == STATUS_OK &&
			dev_sub.pColorStream->readFrame(&vf_color_sub) == STATUS_OK &&
			dev_sub.pDepthStream->readFrame(&vf_depth_sub) == STATUS_OK)
		{
			//master
			{
				cv::Mat cv_img(vf_color_master.getHeight(), vf_color_master.getWidth(), CV_8UC3, (void*)vf_color_master.getData());
				cv::cvtColor(cv_img, cv_img_master, CV_RGB2BGR);
				cv::flip(cv_img_master, cv_img_master, 1);
				cv::imshow(dev_master.colorWindowName, cv_img_master);

				{
					vector<aruco::Marker> Markers = MDetector.detect(cv_img_master);
					for (auto &marker : Markers)
					{
						MTracker_master[marker.id].estimatePose(marker, dev_master.camParam, MarkerSize);
					}
					if (dev_master.camParam.isValid() && MarkerSize != -1)
					{
						for (unsigned int i = 0; i < Markers.size(); ++i)
						{
							if (Markers[i].isPoseValid())
							{
								cv::Mat transformationMatrix_master = Markers[i].getTransformMatrix();
								cv2eigen(transformationMatrix_master, frame_master_marker.matrix());
								poseEstimationOK_master = true;
							}
							else
							{
								poseEstimationOK_master = false;
							}
						}
					}
				}

				getCloudXYZCoordinate(cloud_XYZRGB_master, vf_color_master, vf_depth_master, dev_master.pDepthStream);
				pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_XYZRGB_master);
				viewer_master->addPointCloud<pcl::PointXYZRGB>(cloud_XYZRGB_master, rgb, "cloud_master");
				viewer_master->spinOnce();
				viewer_master->removeAllPointClouds();
			}

			//sub
			{
				cv::Mat cv_img(vf_color_sub.getHeight(), vf_color_sub.getWidth(), CV_8UC3, (void*)vf_color_sub.getData());
				cv::cvtColor(cv_img, cv_img_sub, CV_RGB2BGR);
				cv::flip(cv_img_sub, cv_img_sub, 1);
				cv::imshow(dev_sub.colorWindowName, cv_img_sub);

				{
					vector<aruco::Marker> Markers = MDetector.detect(cv_img_sub);
					for (auto &marker : Markers)
					{
						MTracker_sub[marker.id].estimatePose(marker, dev_sub.camParam, MarkerSize);
					}
					if (dev_sub.camParam.isValid() && MarkerSize != -1)
					{
						for (unsigned int i = 0; i < Markers.size(); ++i)
						{
							if (Markers[i].isPoseValid())
							{
								cv::Mat transformationMatrix_sub = Markers[i].getTransformMatrix();
								cv2eigen(transformationMatrix_sub, frame_sub_marker.matrix());
								poseEstimationOK_sub = true;
							}
							else
							{
								poseEstimationOK_sub = false;
							}
						}
					}
				}

				getCloudXYZCoordinate(cloud_XYZRGB_sub, vf_color_sub, vf_depth_sub, dev_sub.pDepthStream);
				pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_XYZRGB_sub);
				viewer_sub->addPointCloud<pcl::PointXYZRGB>(cloud_XYZRGB_sub, rgb, "cloud_sub");
				viewer_sub->spinOnce();
				viewer_sub->removeAllPointClouds();

			}

			//求转换矩阵-》转换点云-》叠加
			if (poseEstimationOK_master && poseEstimationOK_sub)
			{
				//master->sub转换
				cv::Vec3d rvec_master, tvec_master;
				eigenTransform2cvRvecTvec(frame_master_marker, rvec_master, tvec_master);
				drawAxis(cv_img_master, dev_master.K, dev_master.coeffs, rvec_master, tvec_master, 0.3f);
				Eigen::Affine3d frame_master_sub = frame_master_marker * frame_sub_marker.inverse();

				//sub->master转换
				cv::Vec3d rvec, tvec;
				eigenTransform2cvRvecTvec(frame_sub_marker, rvec, tvec);
				drawAxis(cv_img_sub, dev_sub.K, dev_sub.coeffs, rvec, tvec, 0.3f);
				Eigen::Affine3d frame_sub_master = frame_sub_marker * frame_master_marker.inverse();

				//求rvec，tvec
				eigenTransform2cvRvecTvec(frame_sub_master, rvec, tvec);
				Eigen::Matrix4d frame_matrix = frame_sub_master.matrix();

				std::cout << "frame sub master" << std::endl;
				writeToCSVfile<double>("d:\\frame_sub_master.csv", frame_matrix);
				std::cout << "save matrix into csv file OK.\n";

				std::cout << "frame sub marker" << std::endl;
				frame_matrix = frame_sub_marker.matrix();
				writeToCSVfile<double>("d:\\frame_sub_marker.csv", frame_matrix);
				std::cout << "save matrix into csv file OK.\n";

				drawAxis(cv_img_sub, dev_sub.K, dev_sub.coeffs, rvec, tvec, 0.3f);

				imshow("color master", cv_img_master);
				imshow("color fusion", cv_img_sub);

				//转换sub点云
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZRGB>());
				pcl::transformPointCloud(*cloud_XYZRGB_sub, *cloud_output, frame_sub_master.matrix());

				//叠加显示
				*cloud_output += *cloud_XYZRGB_master;
				pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_output);
				viewer_fusion->addPointCloud<pcl::PointXYZRGB>(cloud_output, rgb, "cloud");
				viewer_fusion->spinOnce(20);
				viewer_fusion->removeAllPointClouds();

				////转换master点云
				//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_output(new pcl::PointCloud<pcl::PointXYZRGB>());
				//pcl::transformPointCloud(*cloud_XYZRGB_master, *cloud_output, frame_sub_master.matrix());

				////叠加显示到master
				//*cloud_output += *cloud_XYZRGB_sub;
				//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_output);
				//viewer_fusion->addPointCloud<pcl::PointXYZRGB>(cloud_output, rgb, "cloud");
				//viewer_fusion->spinOnce();
				//viewer_fusion->removeAllPointClouds();
			}
			else
			{
				std::cout << endl;
			}

		}


		if (cv::waitKey(10) == 27 || cv::waitKey(10) == 'q')
		{
			dev_master.pDepthStream->stop();
			dev_master.pColorStream->stop();
			dev_master.pDevice->close();

			dev_sub.pDepthStream->stop();
			dev_sub.pColorStream->stop();
			dev_sub.pDevice->close();
			break;
		}
	}

	OpenNI::shutdown();
	return 0;
}