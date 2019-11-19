// 标准库头文件
#include <iostream>
#include <string>
#include <vector> 
// OpenCV头文件
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp> 
#include "opencv2/imgproc/types_c.h"
#include <opencv2/highgui/highgui_c.h>
// OpenNI头文件
#include <OpenNI.h> 
// namespace
using namespace std;
using namespace openni;
using namespace cv;

class CDevice {
public:
	string			devicename;
	string			depthWindowName;
	string			colorWindowName;
	Device*			pDevice;
	VideoStream*	pDepthStream;
	VideoStream*	pColorStream;
	cv::VideoWriter outputColorVideo;
	cv::VideoWriter outputDepthVideo;

	CDevice(int idx, const char* uri, string deviceName)
	{
		devicename.assign(deviceName);
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

		cv::Size S = cv::Size(640, 480);

		string outputColorVideoPath = "d:\\" + colorWindowName + ".mp4";
		outputColorVideo.open(outputColorVideoPath, CAP_OPENCV_MJPEG, 30.0, S, true);
		if (!outputColorVideo.isOpened()) {
			cout << "fail to outputColorVideo!" << endl;
		}

		string outputDepthVideoPath = "d:\\" + depthWindowName + ".mp4";
		outputDepthVideo.open(outputDepthVideoPath, CAP_OPENCV_MJPEG, 30.0, S, true);
		if (!outputDepthVideo.isOpened()) {
			cout << "fail to outputColorVideo!" << endl;
		}
	}
};
int main(int argc, char **argv)
{
	OpenNI::initialize();

	// 获取设备信息  
	Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);
 
	vector<CDevice>  vDevices;

	cout << "电脑上连接着 " << aDeviceList.getSize() << " 个体感设备." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "设备 " << i << endl;
		const DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "设备名： " << rDevInfo.getName() << endl;
		cout << "设备Id： " << rDevInfo.getUsbProductId() << endl;
		cout << "供应商名： " << rDevInfo.getVendor() << endl;
		cout << "供应商Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "设备URI: " << rDevInfo.getUri() << endl;

		std::stringstream   ss;
		ss << rDevInfo.getName() << '_' << i;
		CDevice mDev(i, aDeviceList[i].getUri(), ss.str());
		vDevices.push_back(mDev);
	}

	// 获取深度图像帧      
	VideoFrameRef vfDepth;
	VideoFrameRef vfColor;
	//OpenCV image
	cv::Mat cvDepthImg;
	cv::Mat cvBGRImg;
	cv::Mat cvFusionImg;

	cv::namedWindow("fusion");

	while (true)
	{
		for (vector<CDevice>::iterator itDev = vDevices.begin(); itDev != vDevices.end(); itDev++)
		{


			if (itDev->pColorStream->readFrame(&vfColor) == STATUS_OK)
			{
				// convert data into OpenCV type
				cv::Mat cvRGBImg(vfColor.getHeight(), vfColor.getWidth(), CV_8UC3, (void*)vfColor.getData());
				cv::cvtColor(cvRGBImg, cvBGRImg, CV_RGB2BGR);
				cv::flip(cvBGRImg, cvBGRImg, 1);
				cv::imshow(itDev->colorWindowName, cvBGRImg);
				itDev->outputColorVideo << cvBGRImg;
			}

			if (itDev->pDepthStream->readFrame(&vfDepth) == STATUS_OK)
			{
				// 转换成 OpenCV 格式      
				const cv::Mat mImageDepth(vfDepth.getHeight(), vfDepth.getWidth(),
					CV_16UC1, const_cast<void*>(vfDepth.getData()));
				// 从 [0,Max] 转为 [0,255]      
				mImageDepth.convertTo(cvDepthImg, CV_8U, 255.0 / itDev->pDepthStream->getMaxPixelValue());
				cv::flip(cvDepthImg, cvDepthImg, 1);
				//cv::cvtColor(cvDepthImg, cvFusionImg, CV_GRAY2BGR);
				cv::imshow(itDev->depthWindowName, cvDepthImg);
				itDev->outputDepthVideo << cvDepthImg;
			}

			//cv::addWeighted(cvBGRImg, 0.5, cvFusionImg, 0.5, 0, cvFusionImg);
			//cv::imshow("fusion", cvFusionImg);
			//itDev->outputVideo << cvFusionImg;

		}   
		if (cv::waitKey(1) == 'q')
			break;
	}

	for (vector<CDevice>::iterator itDev = vDevices.begin();
		itDev != vDevices.end(); ++itDev)
	{
		itDev->pDepthStream->stop();
		itDev->pDepthStream->destroy();
		delete itDev->pDepthStream;

		itDev->pColorStream->stop();
		itDev->pColorStream->destroy();
		delete itDev->pColorStream;

		itDev->pDevice->close();
		delete itDev->pDevice;
	}
	OpenNI::shutdown();
	return 0;
}