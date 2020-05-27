/**
 * @file multiStream.h
 *
 * @brief This message displayed in Doxygen Files index
 *
 * @ingroup PackageName
 * (Note: this needs exactly one @defgroup somewhere)
 *
 * @author Paul Peters
 *
 */
#ifndef HPIPE_INCLUDED
#define HPIPE_INCLUDED
#include <string>
#include"NvInfer.h"
#include "Utils.h"
//#include "pipe.h"
#include <vector>
#include <queue>



/**
 * Optimized engine class.
 *
 * Optimizes CNN by dividing workload over the available cpu and gpu. 
 * Uses threading for when preforming inference
 * 
 * 
 *
 */
class multiStreamTrt{
    //TO-DO: Replace pointers with smart pointers ********
    public:
        multiStreamTrt(std::string deploy,std::string model, bool multiThread, int nNetworks=2);
        ~multiStreamTrt();

        bool build();

        /** 
        * @brief Launches inference with an input tensor.  
        * 
        *  
        *
        * @param input input vector to be infered
        */
        void launchInference(std::vector<float> input);
        
        /** 
        * @brief Get last output result from the model
        * 
        *  Checks the final pipe of the model and get the last output it generated
        *
        * @param output output vector reference
        */
        bool getOutput(std::vector<float>& output); // ***

        /** 
        * @brief get amount of channels of input tensor.  
        * 
        */
        size_t getNChannels();
        /** 
        * @brief get height of input tensor.  
        * 
        */        
        size_t getHeight();
        /** 
        * @brief get width of input tensor.  
        * 
        */        
        size_t getWidth();
        
        
        //size_t getOutputSize();
        //To-DO *****************
        bool getSerializedEngines(std::vector<std::string> enginePath);
        
        
        /** 
        * @brief Stops the running threads.  
        * 
        */
        bool teardown();
    private:
        bool multiThreading;
        int batchSize=1; 
        int nNetworks;
        const int inputIndex = 0;
        const int outputIndex =1;
        bool onlyCpu = false;
        nvinfer1::Dims inputDims;
        nvinfer1::Dims outputDims;
        
        nvinfer1::ICudaEngine* mEngine;
        void split(nvinfer1::INetworkDefinition*, std::vector<nvinfer1::INetworkDefinition*>&, std::queue<int>&);
        int buildEnginesCPU(std::vector<float> input);
        void buildEnginesGPU(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder,nvinfer1::IBuilderConfig* config);
        void loadNetwork(std::string depoly, std::string model);
        
        nvinfer1::ILayer* addLayerToNetwork(nvinfer1::INetworkDefinition*& network, nvinfer1::ILayer* layer,nvinfer1::ITensor* input);
        
        //unsigned int nThreads = std::thread::hardware_concurrency();
        //std::vector<std::shared_ptr<Pipe>> mPipes;

        bool serializeEngines(); 
    
};



#endif