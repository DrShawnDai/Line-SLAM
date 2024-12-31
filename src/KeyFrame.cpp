#include "KeyFrame.h"

bool KeyFrame::mbInitialComputations = true;
cv::Mat KeyFrame::K, KeyFrame::K_inv, KeyFrame::DistCoef; 
float KeyFrame::mnMinX, KeyFrame::mnMinY, KeyFrame::mnMaxX, KeyFrame::mnMaxY; 
float KeyFrame::mfGridElementWidthInv, KeyFrame::mfGridElementHeightInv;

KeyFrame::KeyFrame(const string imagePath, const cv::Mat &Pos, const string strSettingsFile, const size_t id):
                    index(id), mpLBDextractor(static_cast<LBDextractor*>(NULL))
{
    cv::Mat img = cv::imread(imagePath);
    img.copyTo(image);

    Pos.copyTo(Tcw);

    cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t() ;
    cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3) ;

    Twc = cv::Mat::eye(4, 4, CV_32F);
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    twc.copyTo(Twc.rowRange(0, 3).col(3));
    
    if(mbInitialComputations)
    {
        mnMinX = 0.0f;
        mnMaxX = image.cols;
        mnMinY = 0.0f;
        mnMaxY = image.rows;

        mfGridElementWidthInv  = static_cast<float> (FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float> (FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        // std::cout << "camera intrinsic matrix: " << std::endl;
        // std::cout << K << std::endl;

        cv::invert(K, K_inv);

        DistCoef = cv::Mat::ones(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        // std::cout << "camera distortion parameter: " << std::endl;
        // std::cout << DistCoef << std::endl;

        mbInitialComputations = false;
    }

    cv::undistort(image, undist_image, K, DistCoef, K);
}

size_t KeyFrame::GetIndex()
{
    return index;
}

cv::Mat KeyFrame::GetImage()
{
    return undist_image.clone();
}

cv::Mat KeyFrame::GetPose()
{
    return Tcw.clone();
}
cv::Mat KeyFrame::GetPoseInverse()
{
    return Twc.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    return Tcw.rowRange(0, 3).colRange(0, 3).clone();
}
cv::Mat KeyFrame::GetTranslation()
{
    return Tcw.rowRange(0, 3).col(3).clone();
}

void KeyFrame::InsertKeyPoint(const cv::Point &point)
{
    keypoints.push_back(point);
    return;
}

cv::Point* KeyFrame::GetKeyPoint(const size_t id)
{
    return &keypoints[id];
}

std::vector<cv::Point>* KeyFrame::GetKeyPoints()
{
    return &keypoints;
}

void KeyFrame::DrawKeyPoint(const size_t id)
{
    cv::circle(undist_image, keypoints[id], 3, cv::Scalar(0,255,0), -1);
    cv::putText(undist_image, std::to_string(id + 1), cv::Point(keypoints[id].x - 10, keypoints[id].y),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 1);
    return;
}

void KeyFrame::DrawPoint(const cv::Point pt, const size_t id, cv::Scalar color)
{
    cv::circle(undist_image, pt, 3, color, -1);
    cv::putText(undist_image, std::to_string(id + 1), cv::Point(pt.x - 10, pt.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    return;
}

void KeyFrame::InsertKeyLine(const cv::line_descriptor::KeyLine &line)
{
    mvKeyLines.push_back(line);
    return;
}

cv::line_descriptor::KeyLine* KeyFrame::GetKeyLine(const size_t id)
{
    return &mvKeyLines[id];
}

std::vector<cv::line_descriptor::KeyLine>* KeyFrame::GetKeyLines()
{
    return &mvKeyLines;
}

cv::Mat KeyFrame::GetLineDescriptor(const size_t id)
{
    return mLineDescriptors.row(id).clone();
}

cv::Mat KeyFrame::GetAllLinesDescriptors()
{
    return mLineDescriptors.clone();
}

void KeyFrame::AssignLineFeaturesToGrid()
{
    int NLines = mvKeyLines.size();
    int nReserve = 0.5f * NLines / (FRAME_GRID_COLS * FRAME_GRID_ROWS);

    for (unsigned int i=0; i<FRAME_GRID_COLS; i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS; j++) 
            mLineGrid[i][j].reserve(nReserve);

    for (int i=0; i<NLines; i++)
    {
        const cv::line_descriptor::KeyLine &kl = mvKeyLines[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kl, nGridPosX, nGridPosY))
            mLineGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

bool KeyFrame::PosInGrid(const cv::line_descriptor::KeyLine &kl, int &posX, int &posY)
{
    posX = round((kl.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kl.pt.y - mnMinY) * mfGridElementHeightInv);

    // Undistortion could cause to go out of the image
    if (posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

std::vector<size_t> KeyFrame::GetLineFeaturesInArea(const float &x, const float &y, const float &r) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(mvKeyLines.size());

    const int nMinCellX = max(0, (int)floor((x-mnMinX-r) * mfGridElementWidthInv));
    if(nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int) FRAME_GRID_COLS-1, (int)ceil((x-mnMinX+r) * mfGridElementWidthInv));
    if(nMaxCellX < 0)
        return vIndices;
    
    const int nMinCellY = max(0, (int)floor((y-mnMinY-r) * mfGridElementHeightInv));
    if(nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;
    
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y-mnMinY+r) * mfGridElementHeightInv));
    if(nMaxCellY < 0)
        return vIndices;
    
    for (int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mLineGrid[ix][iy];
            if(vCell.empty ())
                continue;
            
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::line_descriptor::KeyLine &kl = mvKeyLines[vCell[j]];
                const float distx = kl.pt.x - x;
                const float disty = kl.pt.y - y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}

void KeyFrame::DrawKeyLine(const size_t id)
{
    cv::line(undist_image, mvKeyLines[id].getStartPoint(), mvKeyLines[id].getEndPoint(), cv::Scalar(0, 255, 255), 2);
    return;
}

void KeyFrame::DrawLine(const cv::line_descriptor::KeyLine line, cv::Scalar color, const int width)
{
    cv::line(undist_image, line.getStartPoint(), line.getEndPoint(), color, width);
    return;
}

cv::Mat KeyFrame::pixelToCam(const cv::Mat &pixel)
{
    cv::Mat Pc = cv::Mat::ones(3, 1, CV_32F);

    // Xc = (u - cx) / fx, Vc = (v - cy) / fy, Zc = 1
    Pc.at<float>(0) = (pixel.at<float>(0) - K.at<float>(0, 2)) / K.at<float>(0, 0);
    Pc.at<float>(1) = (pixel.at<float>(1) - K.at<float>(1, 2)) / K.at<float>(1, 1);

    return Pc.clone();
}

cv::Mat KeyFrame:: camToWorld(const cv::Mat &Pc)
{
    cv::Mat Rcw = this->GetRotation();
    cv::Mat tcw = this->GetTranslation();

    // Pw = Rwc * (Pc - tcw)
    cv::Mat Pw = Rcw.t() * (Pc - tcw);
    return Pw.clone();
}

void KeyFrame::ExtractLBD()
{
    (*mpLBDextractor)(undist_image, cv::Mat(), mvKeyLines, mLineDescriptors);
}

void KeyFrame::SetExtractor(LBDextractor* lbd_extract)
{
    mpLBDextractor = lbd_extract;
}
