// ��׼��ͷ�ļ�
#include <iostream>
#include <string>
#include <vector> 
// OpenCVͷ�ļ�
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp> 
#include "opencv2/imgproc/types_c.h"
#include <opencv2/highgui/highgui_c.h>
// OpenNIͷ�ļ�
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

	// ��ȡ�豸��Ϣ  
	Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);
 
	vector<CDevice>  vDevices;

	cout << "������������ " << aDeviceList.getSize() << " ������豸." << endl;

	for (int i = 0; i < aDeviceList.getSize(); ++i)
	{
		cout << "�豸 " << i << endl;
		const DeviceInfo& rDevInfo = aDeviceList[i];
		cout << "�豸���� " << rDevInfo.getName() << endl;
		cout << "�豸Id�� " << rDevInfo.getUsbProductId() << endl;
		cout << "��Ӧ������ " << rDevInfo.getVendor() << endl;
		cout << "��Ӧ��Id: " << rDevInfo.getUsbVendorId() << endl;
		cout << "�豸URI: " << rDevInfo.getUri() << endl;

		std::stringstream   ss;
		ss << rDevInfo.getName() << '_' << i;
		CDevice mDev(i, aDeviceList[i].getUri(), ss.str());
		vDevices.push_back(mDev);
	}

	// ��ȡ���ͼ��֡      
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
				// ת���� OpenCV ��ʽ      
				const cv::Mat mImageDepth(vfDepth.getHeight(), vfDepth.getWidth(),
					CV_16UC1, const_cast<void*>(vfDepth.getData()));
				// �� [0,Max] תΪ [0,255]      
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