cd ..
if [ ! -d "build" ]; then
  mkdir build
fi
cd build/
cmake ..
make

cd ../examples
./plucker/plucker
