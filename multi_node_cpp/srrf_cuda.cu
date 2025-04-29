#include "srrf_cuda.h"
#include "srrf_kernel.h"
cudaEvent_t start_total, stop_total;
float milliseconds = 0;
int width, height, n, border, nRingCoordinates;
double magnification, spatialRadius;
int magnification_int, widthM, heightM, outputPixNum, borderM, widthMBorderless, heightMBorderless; 
double * pixels_dev, * GxArray, * GyArray, * radArray, * SrrfArray_dev;
double * xRingCoordinates0, * yRingCoordinates0, * cubic_x, * cubic_x2, * cubic_y;

void srrf_setup(    
    const int width_in,
    const int height_in,
    const int n_in, 
    const int border_in, 
    const int nRingCoordinates_in,
    const double magnification_in,
    const double spatialRadius_in
){
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);
    width = width_in;
    height=height_in;
    n = n_in;
    border = border_in;
    nRingCoordinates = nRingCoordinates_in;
    magnification = magnification_in;
    spatialRadius = spatialRadius_in;
    magnification_int = static_cast<int>(magnification);
    widthM = width * magnification_int;
    heightM = height * magnification_int;
    outputPixNum = widthM*heightM;
    borderM = border * magnification_int;
    widthMBorderless = widthM - borderM * 2;
    heightMBorderless = heightM - borderM * 2;
    cudaMalloc((void**)&pixels_dev, sizeof(double)*n*width*height);
    cudaMalloc((void**)&GxArray, sizeof(double)*n*width*height);
    cudaMalloc((void**)&GyArray, sizeof(double)*n*width*height);
    cudaMalloc((void**)&radArray, sizeof(double)*n*widthMBorderless*heightMBorderless);
    cudaMalloc((void**)&SrrfArray_dev, sizeof(double)*widthM*heightM);
    cudaMalloc((void**)&xRingCoordinates0, sizeof(double)*nRingCoordinates);
    cudaMalloc((void**)&yRingCoordinates0, sizeof(double)*nRingCoordinates);
    cudaMalloc((void**)&cubic_x, sizeof(double)*magnification_int*nRingCoordinates*4);
    cudaMalloc((void**)&cubic_x2, sizeof(double)*magnification_int*4);
    cudaMalloc((void**)&cubic_y, sizeof(double)*magnification_int*nRingCoordinates*4);
    const dim3 nthrds(magnification_int, 4, 4);
    const dim3 nblcks(1, (int)nRingCoordinates/4, 1); 
    init_RingCoordinates_cubic<<<nblcks,nthrds>>>(xRingCoordinates0,yRingCoordinates0,cubic_x,cubic_x2,cubic_y,spatialRadius,magnification,borderM,nRingCoordinates);
}

void srrf_preprocess(
    const double * pixels,
    double * outputPic,    
    const int blockx,
    const int blocky,
    const int gridx,
    const int gridy,
    const int gridz
){
    cudaMemcpy(pixels_dev, pixels, sizeof(double)*n*width*height, cudaMemcpyHostToDevice);   
    const dim3 nthrds(blockx, blocky, 1);
    const dim3 nblcks(gridx, gridy, gridz); 
    calculateSRRF<<<nblcks, nthrds>>>(SrrfArray_dev,radArray,pixels_dev,cubic_x2,cubic_y,width,height,n,border,magnification);
    cudaMemcpy(outputPic, SrrfArray_dev, sizeof(double)*widthM*heightM, cudaMemcpyDeviceToHost);   

}

void srrf_end(double* time){
    cudaFree(pixels_dev);
    cudaFree(GxArray);
    cudaFree(GyArray);
    cudaFree(radArray);
    cudaFree(SrrfArray_dev);
    milliseconds = 0;
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&milliseconds, start_total, stop_total);
    //std::cout << "total time : " << milliseconds << std::endl;
    *time = (double) milliseconds;
}
