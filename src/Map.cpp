#include "Map.h"

Map::Map()
{

}

void Map::InsertMapPoint(MapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexMapPoints); 
    mvpMPs.push_back(pMP);
    return;
}

void Map::InsertMapLine(MapLine* pML)
{
    unique_lock<mutex> lock(mMutexMapLines); 
    mvpMLs.push_back(pML);
    return;
}

void Map::InsertKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexKeyFrames);
    mvpKFs.push_back(pKF);
    return;
}

std::vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMapPoints);
    return std::vector<MapPoint*> (mvpMPs.begin(), mvpMPs.end()) ;
}

std::vector<MapLine*> Map::GetAllMapLines()
{
    unique_lock<mutex> lock(mMutexMapLines);
    return std::vector<MapLine*>(mvpMLs.begin(), mvpMLs.end());
}

std::vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexKeyFrames);
    return std::vector<KeyFrame*> (mvpKFs.begin(), mvpKFs.end());
}

MapLine* Map::GetMapLine(const size_t id)
{
    unique_lock<mutex> lock(mMutexMapLines);
    return mvpMLs[id];
}

MapPoint* Map:: GetMapPoint(const size_t id)
{
    unique_lock<mutex> lock(mMutexMapPoints);
    return mvpMPs[id];
}