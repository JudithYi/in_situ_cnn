#ifndef SRRF_CUDA_H_
#define SRRF_CUDA_H_

void srrf_setup(
    const int width,
    const int height,
    const int n, 
    const int border, 
    const int nRingCoordinates,
    const double magnification,
    const double spatialRadius
);

void srrf_preprocess(
    const double * pixels,
    double * outputPic,    
    const int blockx,
    const int blocky,
    const int blockz,
    const int gridx,
    const int gridy,
    const int gridz

);

void srrf_end(double *time);

#endif

