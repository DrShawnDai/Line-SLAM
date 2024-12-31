#ifndef FRAMEHANDLE_H
#define FRAMEHANDLE_H

#include "KeyFrame.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#define CV_EVENT_MOUSEMOVE      0   // 鼠标滑动
#define CV_EVENT_LBUTTONDOWN    1   // 左键点击
#define CV_EVENT_RBUTTONDOWN    2   // 右键点击
#define CV_EVENT_MBUTTONDOWN    3   // 中键点击
#define CV_EVENT_LBUTTONUP      4   // 左键放开
#define CV_EVENT_RBUTTONUP      5   // 右键放开
#define CV_EVENT_MBUTTONUP      6   // 中键放开
#define CV_EVENT_LBUTTONDBLCLK  7   // 左键双击
#define CV_EVENT_RBUTTONDBLCLK  8   // 右键双击
#define CV EVENT_MBUTTONDBLCLK  9   // 中键双击

cv::Point sta_point; 
cv::Point end_point;

void MouseOnImgHandle(int event, int x, int y, int flags, void* param)
{
    KeyFrame& kf = *(KeyFrame*) param;

    // 点击鼠标左键选取角点
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        cv::Point point(x, y);

        kf.InsertKeyPoint(point);
        kf.DrawKeyPoint(kf.GetKeyPoints()->size() - 1);

        std::cout << "select a point in image " << kf.GetIndex() << ", [x, y]: " << point << std::endl;
    }

    // 按住鼠标右键选取线
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        sta_point.x = x;
        sta_point.y = y;
    }

    if (event == CV_EVENT_RBUTTONUP)
    {
        end_point.x = x;
        end_point.y = y;
    
        if (abs(sqrt(pow(sta_point.x - end_point.x, 2) + pow(sta_point.y - end_point.y, 2))) > 20)
        {
            cv::line_descriptor::KeyLine keyline;
            keyline.startPointX = sta_point.x;
            keyline.startPointY = sta_point.y;
            keyline.endPointX   = end_point.x;
            keyline.endPointY   = end_point.y;

            kf.InsertKeyLine(keyline);
            kf.DrawKeyLine(kf.GetKeyLines()->size() - 1);

            std::cout << "select a line in image " << kf.GetIndex() 
                      << ", start: " << sta_point << ", end: " << end_point << std::endl;
        }
        else 
            std::cout << "line's length need to be more than 20 pixel" << std::endl;
    }

    return;
}

#endif // FRAMEHANDLE_H