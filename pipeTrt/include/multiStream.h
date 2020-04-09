#ifndef HPIPE_INCLUDED
#define HPIPE_INCLUDED
#include <string>
#include"NvInfer.h"
#include "Utils.h"
#include "pipe.h"
#include <vector>
//#include <thread>




class multiStreamTrt{
    //TO-DO: Replace pointers with smart pointers ********
    public:
        multiStreamTrt(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
        ~multiStreamTrt();
        bool build();
        void launchInference(std::vector<float>);
        bool getOutput(std::vector<float>& output); // ***

        size_t getNChannels();
        size_t getHeight();
        size_t getWidth();
        //size_t getOutputSize();
        //To-DO *****************
        bool getSerializedEngines(std::vector<std::string> enginePath);
        bool teardown();

    private:
        int batchSize=1; 
        int nNetworks = 2;
        const int inputIndex = 0;
        const int outputIndex =1;
        
        nvinfer1::Dims inputDims;
        int outputDims[3];

        nvinfer1::ICudaEngine* mEngine;

        void splitNetwork(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder,nvinfer1::IBuilderConfig* config);
        nvinfer1::ILayer* addLayerToNetwork(nvinfer1::INetworkDefinition*& network, nvinfer1::ILayer* layer,nvinfer1::ITensor* input);
        
        //unsigned int nThreads = std::thread::hardware_concurrency();
        std::vector<std::shared_ptr<Pipe>> mPipes;

        bool serializeEngines(); 
    
};



#endif