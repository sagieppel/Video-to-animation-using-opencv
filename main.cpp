#include "opencv2/opencv.hpp"
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include <iostream>
#include <conio.h>
//Capture computer camera video stream and transform to various of animation like properties
 using namespace cv;
 using namespace std;
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int VideoToCanny();

 int VideoToAnimation();
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int, char**)
{
	//VideoToAnimation();
	VideoToCanny();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int VideoToAnimation()//Capture deafult camera image and transform to animation and display on screen
{

    VideoCapture cap(0); // open the default camera
   if(!cap.isOpened())  // check if we succeeded
      return -1;
	 Mat frame;
        cap >> frame;

    namedWindow("Animation",0);
	 namedWindow("Track",0);
     //Create Track bar to change bvarious of image properties
     int ColorSpace = 50;
     createTrackbar("Color red", "Track", &ColorSpace, 255);
	 int EdgeWidth = 1;
     createTrackbar("Edge With", "Track", &EdgeWidth, 30);
	 int IsColor = 1;
    createTrackbar("Is Color", "Track", &IsColor, 1);


    for(;;)
    {
  //   sigma=double(iSliderValue1)/100.0;
        cap >> frame; // get a new frame from camera

	//----------------------Reduce Color space-----------------------------------------------------------------------------
    frame/=ColorSpace;
	frame*=ColorSpace;

	//------------------------------Draw Edges-------------------------------------------------------------------
	for (int r=EdgeWidth;r<=frame.rows-EdgeWidth;r++) 
			for (int c=EdgeWidth;c<=frame.cols-EdgeWidth;c++) 
			{
				if (frame.at<cv::Vec3b>(r,c)!=frame.at<cv::Vec3b>(r,c+EdgeWidth) || frame.at<cv::Vec3b>(r+EdgeWidth,c)!=frame.at<cv::Vec3b>(r,c) ) frame.at<cv::Vec3b>(r,c)=cv::Vec3b(0,0,0); 
				else if (!IsColor) frame.at<cv::Vec3b>(r,c)=cv::Vec3b(255,255,255);
			}  
			//---------------Show on screen-------------------------------------------------------------------------------------------

        imshow("Animation",frame);
	
        if(waitKey(1)== 27) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int VideoToCanny() //Capture deafult camera image and transform to canny edge image and display on screen
{

    VideoCapture cap(0); // open the default camera
   if(!cap.isOpened())  // check if we succeeded
      return -1;
	 Mat frame;
        cap >> frame;
     
/* VideoWriter out("MyVideo.avi", VideoWriter::fourcc('M','J','P','G'), 30, frame.size(), true); //initialize the VideoWriter object 

   if ( !out.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
   {
        cout << "ERROR: Failed to write the video\n" ;
    //    return -1;
   }//*/

    Mat edges;
//-------------------Create tracker thatr control animation properties-----------------------------------------------	  
    namedWindow("edges",1);
 namedWindow("Tracker",1);
     //Create trackbar to change brightness
     int sigma = 150;
     createTrackbar("Sigma", "Tracker", &sigma, 800);
	 int size = 7;
     createTrackbar("Size", "Tracker", &size, 24);
	 int HighTresh = 8;
     createTrackbar("Canny Hight threshold", "Tracker", &HighTresh, 100);
	 int LowThresh = 3;
     createTrackbar("Canny Low threshold", "Tracker", &LowThresh, 100);
    for(;;)
    {
  //   sigma=double(iSliderValue1)/100.0;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, COLOR_BGR2GRAY);//transform to gray
	//	cvtColor(frame,frame, COLOR_BGR2GRAY);
		if (size%2==0)size++;
      if (sigma>0 || size>1) GaussianBlur(edges, edges, Size(size,size), sigma/100.0, sigma/100.0);//smooth
	   if (HighTresh<LowThresh) HighTresh=LowThresh+2;
        Canny(edges, edges, LowThresh,HighTresh,3);// HighTresh, LowThresh); //Transform to edge image
	//	bitwise_not(edges, edges);//inverse black to white 1 to zesro
		Mat mm;
	//	multiply(edges,frame,frame);
        imshow("edges",edges);
	// out << frame;//For image writing not in used 
	
		//getInput(key);

		//cout<<endl<<key<<" Sigma="<<sigma;
        if(waitKey(1)== 27) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
