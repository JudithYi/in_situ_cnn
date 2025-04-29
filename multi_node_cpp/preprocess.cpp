#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp> 
#include "srrf_cuda.h"
#include "mpi.h"
#include "adios2.h"
int main(int argc, char** argv) 
{
    int world_rank, world_size;
    MPI_Comm pre_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &pre_comm);
    std::string configFile = "config/config.xml";
    int width, height, numPic, numOutput_total, numOutput;
    int size, rank;
    std::size_t out_start, out_count, out_total;
    std::size_t label_start, label_count, label_total;
    MPI_Comm_rank(pre_comm, &rank);
    MPI_Comm_size(pre_comm, &size);
    width = 128;
    height = 128;
    double radialityMagnification = 5.0;
    int i,j,k;
    double time_per;
    double time_total = 0.0;
    int magnification = 5;
    double ringRadius = 0.5 * radialityMagnification;
    int symmetryAxes = 6 * 2;
    int blockBorderNotConsideringDrift = static_cast<int>(ringRadius + 3);
    int blockBorderConsideringDrift = blockBorderNotConsideringDrift + 1;
    int widthM = width * magnification;
    int heightM = height * magnification;
    int outputPixNum = 224*224;
    int borderM = blockBorderConsideringDrift * magnification;
    int widthMBorderless = widthM - borderM * 2;
    int heightMBorderless = heightM - borderM * 2;
    numPic = 12;
    numOutput_total = 16;
    int blockx = atoi(argv[1]);
    int blocky = atoi(argv[2]);
    int blockz = atoi(argv[3]);
    int gridx = atoi(argv[4]);
    int gridy = atoi(argv[5]);
    int gridz = atoi(argv[6]);
    int GLOBAL_BATCH_SIZE = 64;
    int GLOBAL_TRAIN_SIZE = 19200;
    int EPOCHS = 2;
    int NUM_ITER = static_cast<int>(GLOBAL_TRAIN_SIZE / GLOBAL_BATCH_SIZE);
    out_total = static_cast<std::size_t>(numOutput_total*outputPixNum);
    label_total = static_cast<std::size_t>(numOutput_total);
    numOutput = static_cast<int> (numOutput_total / size);
    out_start = static_cast<std::size_t>(rank*numOutput*outputPixNum);
    label_start = static_cast<std::size_t>(rank*numOutput);
    if(rank < numOutput_total % size){
        ++numOutput;
        out_start += static_cast<std::size_t>(rank*outputPixNum);
        label_start += static_cast<std::size_t>(rank);
    }
    if(rank >= numOutput_total % size){
        out_start += static_cast<std::size_t>((numOutput_total%size)*outputPixNum);
        label_start += static_cast<std::size_t>(numOutput_total%size);
    }
    out_count = static_cast<std::size_t>(numOutput*outputPixNum);
    label_count = static_cast<std::size_t>(numOutput);
    
    std::cout << rank << " " << numPic << "  " << numOutput << std::endl;
    adios2::ADIOS adios(configFile, pre_comm);
    adios2::IO io = adios.DeclareIO("writer");

    if (!io.InConfigFile())
    {
        // if not defined by user, we can change the default settings
        // BPFile is the default engine
        io.SetEngine("BPFile");
        io.SetParameters({{"num_threads", "1"}});

        // ISO-POSIX file output is the default transport (called "File")
        // Passing parameters to the transport
    }
    adios2::Engine writer = io.Open("globalArray", adios2::Mode::Write, pre_comm);
    adios2::Variable<double> outputImages;
    adios2::Variable<int> outputImageLabel;
    outputImages = io.DefineVariable<double>("image", {out_total}, {out_start}, {out_count});
    outputImageLabel = io.DefineVariable<int>("label", {label_total}, {label_start}, {label_count});

    std::vector<double> pix(width*height*numPic*numOutput);
    std::vector<double> outputPic(numOutput*outputPixNum);


    std::vector<int> picLabel(numOutput);
    for(i=0;i<numOutput;++i){
        picLabel[i]= rand() % (1 - 0 + 1);
    }
    cv::Mat image(width,height,CV_16U);
    //std::cout << image.at<unsigned short>(0,100) << std::endl;
    
    std::string fileEnd = ".tif";
    std::string fileName;
    int offset = 0;
    srrf_setup(width,height,numPic,blockBorderConsideringDrift,symmetryAxes,magnification,ringRadius);
    for(int idx_epoch; idx_epoch < EPOCHS*NUM_ITER; ++idx_epoch){
        offset = idx_epoch * numOutput * numPic;
        for(j=0;j<numOutput;++j){
        for(i=0;i<numPic && i<9;++i){
            fileName = "input/" + std::to_string(j+offset) + "/sequence/0000" + std::to_string(i+1) + fileEnd;
            image = cv::imread(fileName, -1);
            std::copy((unsigned short*)image.ptr<unsigned short>(),(unsigned short*)image.ptr<unsigned short>()+width*height, pix.data()+(i+j*numPic)*width*height);
        }
        for(i=9;i<numPic && i<99;++i){
            fileName = "input/" + std::to_string(j+offset) + "/sequence/000" + std::to_string(i+1) + fileEnd;
            image = cv::imread(fileName, -1);
            std::copy((unsigned short*)image.ptr<unsigned short>(),(unsigned short*)image.ptr<unsigned short>()+width*height, pix.data()+(i+j*numPic)*width*height);
        }
        }
        //std::cout << pix[100] << std::endl;
        for(i=0;i<numOutput;++i){
            srrf_preprocess(pix.data()+i*numPic*width*height,outputPic.data()+i*widthM*heightM,blockx,blocky,blockz,gridx,gridy,gridz);
        }

        writer.BeginStep();
        writer.Put<double>(outputImages, outputPic.data());
        writer.Put<int>(outputImageLabel, picLabel.data());
        writer.EndStep();
    }
    writer.Close();
    srrf_end(&time_per);
    
    MPI_Finalize();
    return 0;
}

