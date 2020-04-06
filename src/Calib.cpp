//
//  Calib.cpp
//
//  Copyright © 2019 Songyou Peng. All rights reserved.
//
// The code is original from the sample camera calibration code in OpenCV (Details can be found in http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html )
// Modified and Changed to OOP by Songyou Peng, INRIA Grenoble Rhône-Alpes, 2016

#include "Calib.h"
#include <sstream>
#include <iomanip>

typedef struct {
    cv::Mat R;
    cv::Mat t;
    cv::Mat K;
    cv::Mat dist_coeff;
    cv::Point2f p;
} reprojectionCapsule;


Calib::Calib()
{
    numIntr = 3;
    RED = Scalar(0,0,255);
    GREEN = Scalar(0,255,0);
}

bool Calib::readSettings(Settings ini_s)
{
    if (!ini_s.goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
        return false;
    }
    s = ini_s;
    mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION; // **** Can be improved
    
    if (!s.calibZerok1Dist)
        ++numIntr;
    if (!s.calibZerok2Dist)
        ++numIntr;
    return true;
}


Mat concatenate_images(const Mat& img1, const Mat& img2, int horizontal_or_vertical)
{
	Mat res;
	//  Check if the two image have the same # of channels and type	
	//  If # channels or type is different	
	if (img1.type() != img2.type() || img1.channels() != img2.channels())    return res;
	int rows = img1.rows + img2.rows, cols = img1.cols + img2.cols;
	bool is_horizontal = true;
	if (horizontal_or_vertical >= 0)
	{
		if (horizontal_or_vertical > 0)
		{
			is_horizontal = false;
		}
		else if (cols > rows)
		{
			is_horizontal = false;
		}
	}
	// Get dimension of final image	
	if (is_horizontal)
	{
		rows = max(img1.rows, img2.rows);
	}
	else
	{
		cols = max(img1.cols, img2.cols);
	}
	// Create a black image	
	//res = Mat3b(rows, cols, Vec3b(0,0,0));	
	res = Mat::zeros(rows, cols, img1.type());
	// Copy images in correct position	
	img1.copyTo(res(Rect(0, 0, img1.cols, img1.rows)));
	if (is_horizontal)
	{
		img2.copyTo(res(Rect(img1.cols, 0, img2.cols, img2.rows)));
	}
	else
	{
		img2.copyTo(res(Rect(0, img1.rows, img2.cols, img2.rows)));
	}
	//imshow("img1", img1);waitKey();   imshow("img2", img2);   imshow("res", res); waitKey();  exit(0);	
	return res;
}

void Calib::cameraCalib()
{
    for(int i = 0;;++i)
    {
        Mat view;
        bool blinkOutput = false;
        
        view = s.nextImage();
        
        //-----  If no more image, or got enough, then stop calibration and show result -------------
        if( mode == CAPTURING && imagePoints.size() >= (unsigned)s.nrFrames )
        {
            if( runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints))
                mode = CALIBRATED;
            else
                mode = DETECTION;
        }
        
        if(view.empty())          // If no more images then stop loop.
        {
//            if( imagePoints.size() > 0 )
//                runCalibrationAndSave(s, imageSize,  cameraMatrix, distCoeffs, imagePoints);
            break;
        }
        
        imageSize = view.size();  // Format input image.
        if( s.flipVertical )    flip( view, view, 0 );
        
        vector<Point2f> pointBuf;
        
        bool found;
        switch( s.calibrationPattern ) // Find feature points on the input format
        {
            case Settings::CHESSBOARD:
				//cout << "s.boardSize : " << s.boardSize << endl;
                found = findChessboardCorners( view, s.boardSize, pointBuf,
                                              CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
                break;
            case Settings::CIRCLES_GRID:
                found = findCirclesGrid( view, s.boardSize, pointBuf );
                break;
            case Settings::ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid( view, s.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID );
                break;
            default:
                found = false;
                break;
        }
        
        if (found)                // If done with success,
        {
			cout << "image " << i << " : chessboard FOUND." << endl;
			// improve the found corners' coordinate accuracy for chessboard
            if( s.calibrationPattern == Settings::CHESSBOARD)
            {
                Mat viewGray;
                cvtColor(view, viewGray, COLOR_BGR2GRAY);
                cornerSubPix( viewGray, pointBuf, Size(11,11),
                             Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
            }
            
            if( mode == CAPTURING &&  // For camera only take new samples after delay time
               (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
            {
                imagePoints.push_back(pointBuf);
                prevTimestamp = clock();
                blinkOutput = s.inputCapture.isOpened();
            }
            
            // Draw the corners.
            drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found );
            
        }
        //if(!found)
		else cout << "image " << i << " : chessboard NOT FOUND."<< endl;
        
        //----------------------------- Output Text ------------------------------------------------
        string msg = (mode == CAPTURING) ? "100/100" :
        mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);
        
        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames );
        }
        
        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);
        
        if( blinkOutput )
            bitwise_not(view, view);
        
        //------------------------- Video capture  output  undistorted ------------------------------
//        if( mode == CALIBRATED && s.showUndistorsed )
//        {
//            Mat temp = view.clone();
//            undistort(temp, view, cameraMatrix, distCoeffs);
//        }
        //------------------------------ Show image and check for input commands -------------------
        imshow("Image View", view);
        char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);
        
        if( key  == ESC_KEY )
            break;
        
        if( key == 'u' && mode == CALIBRATED )
            s.showUndistorsed = !s.showUndistorsed;
        
        if( s.inputCapture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }
        
    }
	cout << "AAA cam" << endl;
    
    // -----------------------Show the undistorted image for the image list ------------------------
    if( s.inputType == Settings::IMAGE_LIST && s.showUndistorsed )
    {
		cout << "BBB cam" << endl;
		Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                imageSize, CV_16SC2, map1, map2);
		cout << "CCC cam" << endl;

        for(int i = 0; i < (int)s.imageList.size(); i++ )
        {
            view = imread(s.imageList[i], 1);
            if(view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
			//imshow("Image View", rview);
			Mat distUndist = concatenate_images(view, rview, 1);
			imshow("left : distorted, right : undistorted", distUndist);
            char c = (char)waitKey();
            if( c  == ESC_KEY || c == 'q' || c == 'Q' )
                break;
        }
		cout << "DDD cam" << endl;
	}
	cout << "EEE" << endl;

}

bool Calib::runCalibrationAndSave(Settings& s, Size imageSize, Mat&  cameraMatrix, Mat& distCoeffs,vector<vector<Point2f> > imagePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    
    bool ok = runCalibration(s,imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
                             reprojErrs, totalAvgErr);
    cout << (ok ? "Calibration succeeded" : "Calibration failed")
    << ". avg re projection error = "  << totalAvgErr << endl;
    
    
	cout << "AAA run" << endl;

    
    if( ok )
        saveCameraParams( s, imageSize, cameraMatrix, distCoeffs, rvecs ,tvecs, reprojErrs,
                         imagePoints, totalAvgErr);
	cout << "BBB run" << endl;
    return ok;
}

bool Calib::runCalibration( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
                           vector<float>& reprojErrs,  double& totalAvgErr)
{
    
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = 1.0;
    
    
    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
    
    objectPoints.resize(imagePoints.size(),objectPoints[0]);
    
    bool ok;
    
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, s.flag|CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    
    
    cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;
    
    ok = checkRange(cameraMatrix) && checkRange(distCoeffs);
    
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    
    return ok;
}

double Calib::computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                        const vector<vector<Point2f> >& imagePoints,
                                        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                        const Mat& cameraMatrix , const Mat& distCoeffs,
                                        vector<float>& perViewErrors)
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());
    
    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
        
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }
    
    return std::sqrt(totalErr/totalPoints);
}

void Calib::calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
                                     Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
    corners.clear();
    
    switch(patternType)
    {
        case Settings::CHESSBOARD:
        case Settings::CIRCLES_GRID:
            for( int i = 0; i < boardSize.height; ++i )
                for( int j = 0; j < boardSize.width; ++j )
                    corners.push_back(Point3f(float( j*squareSize ), float( i*squareSize ), 0));
            break;
            
        case Settings::ASYMMETRIC_CIRCLES_GRID:
            for( int i = 0; i < boardSize.height; i++ )
                for( int j = 0; j < boardSize.width; j++ )
                    corners.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
            break;
        default:
            break;
    }
}

//void writeMatToFile(cv::Mat& m, const char* filename)
void writeMatToFile(cv::Mat& m, const string& filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}


// Print camera parameters to the output file
void Calib::saveCameraParams( Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                             const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                             const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
                             double totalAvgErr )
{
	cout << "AAA save" << endl;
	FileStorage fs( s.outputFileName, FileStorage::WRITE );
	writeMatToFile(cameraMatrix, base_path + "out_camera_matrix.txt");
	writeMatToFile(distCoeffs, base_path + "out_distort_coeff.txt");
/*
    ofstream txt_cameraMatrix;
    ofstream txt_distortCoeff;
    ofstream txt_rotationMat;
    ofstream txt_tVec;
    ofstream txt_twoD;
	cout << "BBB save" << endl;

    txt_cameraMatrix.open (base_path + "out_camera_matrix.txt");
    txt_distortCoeff.open (base_path + "out_distort_coeff.txt");
    txt_twoD.open(base_path + "out_camera_points.txt");
    txt_rotationMat.open (base_path + "out_rotation_matrix.txt");
    txt_tVec.open(base_path + "out_translation_vector.txt");
	cout << "CCC save" << endl;
	
	cout << "base_path + out_camera_matrix.txt : " << base_path + "out_camera_matrix.txt" << endl;
	cout << "cameraMatrix.size() : " << cameraMatrix.size() << endl;

    txt_cameraMatrix << cameraMatrix;
	cout << "DDD save" << endl;
	cout << "base_path + out_distort_coeff.txt : " << base_path + "out_distort_coeff.txt" << endl;
	txt_cameraMatrix.close();
	cout << "EEE save" << endl;
	txt_distortCoeff << distCoeffs;
	cout << "FFF save" << endl;
	txt_distortCoeff.close();
	cout << "GGG save" << endl;
*/

    time_t tm;
    time( &tm );
    struct tm *t2 = localtime( &tm );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );
	cout << "EEE save" << endl;

    fs << "calibration_Time" << buf;
    
    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nrOfFrames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_Width" << imageSize.width;
    fs << "image_Height" << imageSize.height;
    fs << "board_Width" << s.boardSize.width;
    fs << "board_Height" << s.boardSize.height;
    fs << "square_Size" << s.squareSize;
    fs << "k1_Dist" << s.calibZerok1Dist;
    fs << "k2_Dist" << s.calibZerok2Dist;
	cout << "AAA save" << endl;

    if( s.flag & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "FixAspectRatio" << s.aspectRatio;
    
    if( s.flag )
    {
        sprintf( buf, "flags: %s%s%s%s",
                s.flag & CV_CALIB_USE_INTRINSIC_GUESS ? " +use_intrinsic_guess" : "",
                s.flag & CV_CALIB_FIX_ASPECT_RATIO ? " +fix_aspectRatio" : "",
                s.flag & CV_CALIB_FIX_PRINCIPAL_POINT ? " +fix_principal_point" : "",
                s.flag & CV_CALIB_ZERO_TANGENT_DIST ? " +zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );
        
    }
	cout << "FFF save" << endl;

    fs << "flagValue" << s.flag;
    
    fs << "Camera_Matrix" << cameraMatrix;
    fs << "Distortion_Coefficients" << distCoeffs;
	cout << "GGG save" << endl;

    fs << "Avg_Reprojection_Error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "Per_View_Reprojection_Errors" << Mat(reprojErrs);
    
    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        
        
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));
            
            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
            
            Mat outr;
            Rodrigues(r, outr);
            //txt_rotationMat << outr.reshape(0,1) << endl;            txt_tVec << t << endl;
			std::stringstream ss;	ss << std::setw(2) << std::setfill('0') << i;
			writeMatToFile(outr, base_path + "out_rotation_matrix" + ss.str() + ".txt");
			writeMatToFile(t, base_path + "out_translation_vector" + ss.str() + ".txt");            
        }
        
        //txt_rotationMat.close();        txt_tVec.close();
        
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "Extrinsic_Parameters" << bigmat;
        
    }

	cout << "HHH save" << endl;

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
            //txt_twoD << imagePoints[i] << endl;
			std::stringstream ss;	ss << std::setw(2) << std::setfill('0') << i;

			writeMatToFile(Mat(imagePoints[i]), base_path + "out_camera_points" + ss.str() + ".txt");
		}

        fs << "Image_points" << imagePtMat;
        
        //txt_twoD.close();
    }
	cout << "III save" << endl;

}


