//  Copyright © 2019 Songyou Peng. All rights reserved.

#include "Proc.h"
#include "setting.h"
#include "Calib.h"

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

int main(int argc, const char * argv[])
{
    int capture_ini_idx(0); // Index for initial calibration
    int calibrated_image_idx(0); // Index for new calibration with next pose image
    int mode_num(0); // 0 for initial image capturing, 1 for next pose estimation
    int keyNum = 0; // variable for key pressing
    bool chess_flag(false); // flag for calibration
    Proc *obj = new Proc;
    Settings *setting = new Settings;
    
    std::string inputSettingsFile = "../config/wizard.xml";
    obj -> base_path = "../out/";
    obj -> imagelist_path = obj -> base_path + "images/image_list.xml";
    obj -> extractInfo(inputSettingsFile);// Extract Info

    // Define the path of different contents
    std::string nextpose_image_path = obj -> base_path + "images/nextpose/";
    std::string ini_calib_image_path = obj -> base_path + "images/ini_calib/";
    setting->input = obj -> imagelist_path;
    setting->outputFileName = obj -> base_path + "out_camera_data.xml";
    
    std::cout << "Please choose the mode. (0: initial images capture, 1: show next pose, 2: calibrate existing images) " <<std::endl;
    std::cin >> mode_num;

    switch (mode_num)
    {
        case 0:
            obj -> resetImagePath(); // Reset the xml file containing the image list
			//std::cout << "Press space key to capture an image, and please capture at least " << int(setting->nrFrames * 1.1) << " images" << std::endl;
			std::cout << "Press space key to capture an image, and please capture at least two images" << std::endl;
            break;
        case 1:
            obj -> extract_points();// Read all the points of next pose
            obj -> extract_calibPara(); // Read calibration parameters
            obj -> build_3Dpoints();
            std::cout << "Press space key to capture an image" << std::endl;
            break;
        case 2:
            break;
    }

    if (mode_num!=2)
    {
        obj -> openCamera();
		for(;;)
        {
            keyNum = cv::waitKey(5);
            if(!(obj -> controlFrame(keyNum))) break; // Show frame
            
            if(mode_num == 1) // Show the next pose
            {
				//cout << "AA 1" << endl;
                obj -> showNextPose();
                if(obj -> plotGuide(0) && (char)keyNum == ' ') // Captured a new image
                {
					cout << "AA 2" << endl;
					// Get the index of the current new pose
                    calibrated_image_idx = obj -> update_captureIndex();
					cout << "AA 3" << endl;
					bool b_saved = obj -> captureImage(nextpose_image_path, calibrated_image_idx);
					if (b_saved)
					{
						cout << "AA 4" << endl;
						//Add the path of new captured image to the image list
						obj->addImagePath(mode_num);
					}
					cout << "AA 5" << endl;
					//break;
                }
            }
            else
            {
				if ((char)keyNum == ' ') // Captured a new image
				{
					//cout << "space bar is pressed" << endl;
					if (chess_flag)
					{
						//cout << "chess_flag is true" << endl;
						bool b_saved = obj->captureImage(ini_calib_image_path, capture_ini_idx);
						if (b_saved)
						{
							//Add the path of new captured image to the image list
							obj->addImagePath(mode_num);
							capture_ini_idx++;
						}
						//is_new_detection = false;
					}
					else
					{
						//cout << "chess_flag is false" << endl;
						std::cout << "Warning: checkboard is not detected!" << std::endl;
					}
				}
                chess_flag = obj -> plotGuide(0);
            }
            obj -> showFrame();
        }
        cv::destroyAllWindows();
    }
    
    // Camera calibration using OpenCV
    FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
        std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
        return false;
    }
    fs["Settings"] >> *setting;
    fs.release();// close Settings file
    
    Calib *calib = new Calib;
    calib -> base_path = obj -> base_path;
    // Copy the setting information to the calibration class
    calib -> readSettings(*setting);
    // Calibrate the camera using the captured images
    calib -> cameraCalib();
    
    cv::destroyAllWindows();
    std::cout << std::endl;
    delete obj;
    delete calib;
    delete setting;
    
    return 0;
}
