#ifndef SRRF_KERNEL_H_
#define SRRF_KERNEL_H_

__global__ void calculateSRRF(
    double * SRRFArray,
    const double * radArray,
    const double * pixels,
    const double * cubic_x2,
    const double * cubic_y,
    const int width,
    const int height,
    const int n,
    const int border, 
    const int nRingCoordinates,
    const double magnification
){
    int widthM = 224;
    int heightM = 224;
    int i, j;
    for(j = threadIdx.y+blockDim.y*blockIdx.y; j<heightM; j+=blockDim.y*gridDim.y){
        for(i = threadIdx.x+blockDim.x*blockIdx.x; i<widthM; i+=blockDim.x*gridDim.x){
            SRRFArray[j*widthM+i]=pixels[__double2int_rn(j/1.75)*widthM+__double2int_rn(i/1.75)];
        }    
    }
}
#endif

