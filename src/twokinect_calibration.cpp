// ��׼��ͷ�ļ�
#include <iostream>
#include <fstream>
#include <string>
#include <vector> 
#include <map>

// OpenCVͷ�ļ�
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp> 
#include <opencv2\imgproc\types_c.h>
#include <opencv2\highgui\highgui_c.h>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>
#include <opencv2\core\eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// OpenNIͷ�ļ�
#include <OpenNI.h> 
#include <aruco/aruco.h>

using namespace std;
//using namespace cv;
using namespace openni;
using namespace Eigen;
//using namespace sen;

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

		// �������
		pDepthStream = new VideoStream();
		pDepthStream->create(*pDevice, SENSOR_DEPTH);
		VideoMode modeDepth;
		modeDepth.setResolution(640, 480);
		modeDepth.setFps(30);
		modeDepth.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
		pDepthStream->setVideoMode(modeDepth);
		pDepthStream->start();
		cv::namedWindow(depthWindowName, CV_WINDOW_AUTOSIZE);

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
float MarkerSize = 0.1f;

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



int main(int argc, char **argv)
{
	OpenNI::initialize();

	// ��ȡ�豸��Ϣ  
	openni::Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);
	const uint32_t deviceCount = aDeviceList.getSize();
	cout << "���� " << deviceCount << " ������豸." << endl;
	if (deviceCount !=2)
	{
		cout << "kinect devices not enough" << endl;
		return 0;
	}


	cout << "�豸 " << 0 << endl;
	cout << "�豸���� " << aDeviceList[0].getName() << endl;
	cout << "�豸URI: " << aDeviceList[0].getUri() << endl;
	CDevice dev_master(0, aDeviceList[0].getUri(), aDeviceList[0].getName());

	double k_master[] = { 518.4867555811269, 0, 301.2378554028732, 0, 517.9021815902831, 244.2168611511136, 0, 0, 1};
	double c_master[] = { 0.2969676497213135, -1.282267901212675, -0.004452857706333246, -0.004453468933053606, 2.014987980086707 };
	dev_master.K = cv::Mat(3, 3, CV_64FC1, k_master);
	dev_master.coeffs = cv::Mat(5, 1, CV_64FC1, c_master);
	dev_master.camParam.setParams(dev_master.K, dev_master.coeffs, cv::Size(640, 480));
	//camera_matrix_color = cv::Mat(3, 3, CV_32F, &k_master);
	//dist_coeffs_color = cv::Mat::zeros(5, 1, CV_32F);
	//camera_matrix_color = dev_master.K.clone();
	//dist_coeffs_color = dev_master.coeffs.clone();

	cout << "�豸 " << 1 << endl;
	cout << "�豸���� " << aDeviceList[1].getName() << endl;
	cout << "�豸URI: " << aDeviceList[1].getUri() << endl;
	CDevice dev_sub(1, aDeviceList[1].getUri(), aDeviceList[1].getName());

	double k_sub[] = { 521.5044785405424, 0, 325.6884223968248, 0, 522.9233149474519, 252.8945754365468, 0, 0, 1};
	double c_sub[] = { 0.2271155572064441, -0.7912825841629555, -0.005510686460386698, -0.006743974360193664, 0.908899981405665 };
	dev_sub.K = cv::Mat(3, 3, CV_64FC1, k_sub);
	dev_sub.coeffs = cv::Mat(5, 1, CV_64FC1, c_sub);
	dev_sub.camParam.setParams(dev_sub.K, dev_sub.coeffs, cv::Size(640, 480));

	dic = aruco::Dictionary::load("ARUCO_MIP_36h12");
	MDetector.setDictionary("ARUCO_MIP_36h12", 0.05f);
	MDetector.setDetectionMode(aruco::DM_NORMAL);
	
	VideoFrameRef vfDepth;
	VideoFrameRef vfcolor_master;
	VideoFrameRef vfcolor_sub;
	//OpenCV image
	cv::Mat cvDepthImg;
	cv::Mat cv_img_master;
	cv::Mat cv_img_sub;

	Eigen::Affine3d frame_master_marker;
	Eigen::Affine3d frame_sub_marker;
	Eigen::Affine3d frame_master_sub;

	bool poseEstimationOK_master = false;
	bool poseEstimationOK_sub = false;

	while (true)
	{
		if (dev_master.pColorStream->readFrame(&vfcolor_master) == STATUS_OK && dev_sub.pColorStream->readFrame(&vfcolor_sub) == STATUS_OK)
		{
			//master
			{
				cv::Mat cv_img(vfcolor_master.getHeight(), vfcolor_master.getWidth(), CV_8UC3, (void*)vfcolor_master.getData());
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

			}

			//sub
			{
				cv::Mat cv_img(vfcolor_sub.getHeight(), vfcolor_sub.getWidth(), CV_8UC3, (void*)vfcolor_sub.getData());
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

			}

			if (poseEstimationOK_master && poseEstimationOK_sub)
			{
				Eigen::Affine3d frame_master_sub = frame_master_marker * frame_sub_marker.inverse();

				cv::Vec3d rvec, tvec;
				eigenTransform2cvRvecTvec(frame_sub_marker, rvec, tvec);
				drawAxis(cv_img_sub, dev_sub.K, dev_sub.coeffs, rvec, tvec, 0.5f);

				Eigen::Affine3d frame_sub_master = frame_sub_marker * frame_master_marker.inverse();
				eigenTransform2cvRvecTvec(frame_sub_master, rvec, tvec);
				Eigen::Matrix4d frame_matrix = frame_sub_master.matrix();

				std::cout << "frame sub master" << std::endl;
				writeToCSVfile<double>("d:\\frame_sub_master.csv", frame_matrix);
				std::cout << "save matrix into csv file OK.\n";

				std::cout << "frame sub marker" << std::endl;
				frame_matrix = frame_sub_marker.matrix();
				writeToCSVfile<double>("frame_sub_marker.csv", frame_matrix);
				std::cout << "save matrix into csv file OK.\n";

				drawAxis(cv_img_sub, dev_sub.K, dev_sub.coeffs, rvec, tvec, 0.5f);
			}
			else
			{
				std::cout << endl;
			}
			imshow("color sub", cv_img_sub);
		}

		
		if (cv::waitKey(20) == 27 || cv::waitKey(20) == 'q')
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