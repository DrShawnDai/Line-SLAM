cd ..
if [ ! -d "build" ]; then
  mkdir build
fi
cd build/
cmake ..
make

cd ../examples
if [ ! -d "saveimg" ]; then
  mkdir saveimg
fi
./optimize/optimize