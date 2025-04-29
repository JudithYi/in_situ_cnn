import cv2
import numpy as np
from SRRFclass_numba import SRRF
import os, sys 
import numba 
import time
import random
from mpi4py import MPI
from datetime import datetime

@numba.njit(parallel=True)
def multi_input(srrf, input_array, width, height, shiftX, shiftY,output):
    for idx_pic in numba.prange(len(input_array)):
        threadId = numba.np.ufunc.parallel._get_thread_id()
        srrf[threadId].calculate(input_array[idx_pic], width, height, shiftX, shiftY)
        for idx in range(srrf[threadId].widthHeightM):
            srrf[threadId].calculateSRRF(idx)
        output[idx_pic]=srrf[threadId].return_result()
        statement = "thread id " + str(threadId) + " using solver " + str(threadId) + " calculate sequence " + str(idx_pic)
        print(statement)

def minMaxScaler(image: np.ndarray):
    image_normed = [None]
    min_scale = np.min(image)
    max_scale = np.max(image)
    if max_scale - min_scale != 0:
        image_normed = (image- min_scale) / (max_scale - min_scale)
    return np.asarray(image_normed, dtype=np.float64)


def process_SRRF_mpi(rank, size, nthread, numInput, numPic, width, height, filepaths):
    numba.config.THREADING_LAYER = 'omp'
    if rank == 1:
       print("numba.config.NUMBA_NUM_THREADS: ", numba.config.NUMBA_NUM_THREADS, flush = True)
       print("numba.get_num_threads(): ", numba.get_num_threads(), flush = True)

    startT = time.time()
    
    input_array = np.zeros((numInput, numPic, width, height))
    
    for i in range (0, numPic):
        j = random.randint(0, len(filepaths) - 1)
        filename = filepaths[j]
        im = cv2.imread(filename, -1)
        imf = np.array(im,np.float64)
        input_array[:, i] = imf

    ringRadius = 0.5
    radialityMagnification = 5
    symmetryAxes = 6
    framesPerTimePoint = 0
    frameStart = 0
    frameEnd = 0
    maxTemporalBlock = 100
    prefSpatialBlock = 0
    display = 'Radiality'
    srrftype = 'TRA'
    doIntegrateLagTimes = True
    srrforder = 1
    radialityPositivityConstraint = True # always negation of generic dialog value !gd.getNextBoolean();
    renormalize = False
    doGradSmooth = False
    doIntensityWeighting= True
    doGradWeight = False
    psfWidth = 1.3489362 # psfWidth = (float) min(max(gd.getNextNumber(),1.0f),5.0f) / 2.35f;
    doMinimizePatterning = True
    doLinearise = False
    blockBorderNotConsideringDrift = int(ringRadius + 3)
    if doGradWeight and (psfWidth > ringRadius) :
        blockBorderNotConsideringDrift = int(np.floor(psfWidth) + 4)
    _blockBorderConsideringDrift = int(blockBorderNotConsideringDrift + 1)
    if frameStart == 0: 
        _frameStart = 1
    else: 
        _frameStart = frameStart

    if frameEnd == 0:
        _frameEnd = len(input_array)
    else:
        _frameEnd = frameEnd

    if framesPerTimePoint == 0 or framesPerTimePoint > len(input_array) :
        _framesPerTimePoint = _frameEnd - _frameStart + 1
    else:
        _framesPerTimePoint = framesPerTimePoint

    _nTimePoints = (_frameEnd - _frameStart + 1) / _framesPerTimePoint
    SRRFsolver = [SRRF() for i in range(nthread)]
    for solver in SRRFsolver:
        solver.setupSRRF(radialityMagnification, srrforder, symmetryAxes, \
                    ringRadius, psfWidth, _blockBorderConsideringDrift, \
                    True, renormalize, doIntegrateLagTimes, \
                    radialityPositivityConstraint, \
                    doGradWeight, doIntensityWeighting, doGradSmooth,\
                    display)
    '''
    SRRFsolver.setupSRRF(magnification = radialityMagnification, SRRForder=srrforder, symmetryAxis = symmetryAxes, \
                    spatialRadius = ringRadius, psfWidth=psfWidth, border = _blockBorderConsideringDrift, \
                    doRadialitySquaring = True, renormalize = renormalize, doIntegrateLagTimes= doIntegrateLagTimes, \
                    radialityPositivityConstraint = radialityPositivityConstraint, \
                    doGradWeight = doGradWeight, doIntensityWeighting = doIntensityWeighting, doGradSmoothing = doGradSmooth,\
                    display_name = display)
    '''
    
    shiftX = np.zeros(numPic,dtype=np.float64)
    shiftY = np.zeros(numPic,dtype=np.float64)

    output = np.zeros((numInput,width*1.75*height*1.75),dtype=np.float64)
    numba.set_num_threads(nthread)
    if rank == 1:
        print("rank: ", rank, " numba.get_num_threads(): ", numba.get_num_threads(), flush = True)
    multi_input(numba.typed.List(SRRFsolver), input_array, width, height, shiftX, shiftY, output)
    # print("numba.threading_layer():",  numba.threading_layer())
    output=output.reshape(numInput,width*1.75,height*1.75)
    resized_output = np.zeros((numInput, 224, 224))

    for i in range(output.shape[0]):
        image = output[i]
        resized_output[i] = minMaxScaler(image)
    
    resized_output = np.expand_dims(resized_output, axis=-1)
    now = datetime.now()
    date_time = now.strftime("%d_%H_%M_%S_%f")
    output_filename = f"{rank}_out_{date_time}.tif"

    # cv2.imwrite(output_filename,im)
    total_time = time.time() - startT
    if rank == 1:
        print("total time for generating ", numInput," images: ", total_time, flush = True)

    return resized_output
    



