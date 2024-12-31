#ifndef MAP_H
#define MAP_H

#include "KeyFrame.h"
#include "MapPoint.h"
#include "MapLine.h"

class Map
{ 
public:
    Map();

    void InsertMapPoint(MapPoint* pMP);
    std::vector<MapPoint*> GetAllMapPoints();

    void InsertMapLine(MapLine* pML);
    std::vector<MapLine*> GetAllMapLines();

    void InsertKeyFrame(KeyFrame* pKF);
    std::vector<KeyFrame*> GetAllKeyFrames();

    MapLine* GetMapLine(const size_t id);
    MapPoint* GetMapPoint(const size_t id);

private:
    std::vector<MapPoint*> mvpMPs;
    std::mutex mMutexMapPoints;

    std::vector<MapLine*> mvpMLs;
    std::mutex mMutexMapLines;

    std::vector<KeyFrame*> mvpKFs;
    std::mutex mMutexKeyFrames;
};

#endif // MAP_H