export ADIOS_CXXFLAG=`/path/to/adios_install/bin/adios2-config --cxx-flags`
export ADIOS_LDFLAG=`/path/to/adios_install/bin/adios2-config --cxx-libs`
nvcc -c -o srrf_cuda.o srrf_cuda.cu 
mpic++ -c -I/path/to/install_opencv/include/opencv4 $ADIOS_CXXFLAG -o preprocess.o preprocess.cpp 
mpic++ -o preprocess_cpp preprocess.o -L/path/to/install_opencv/lib64/ -static-libstdc++ -lopencv_core -lopencv_videoio -lopencv_imgcodecs $ADIOS_LDFLAG -L$CUDA_HOME/lib64 -lcudart srrf_cuda.o 
