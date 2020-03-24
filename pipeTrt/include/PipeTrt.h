#ifndef HPIPE_INCLUDED
#define HPIPE_INCLUDED
#include <string>
#include"NvInfer.h"
#include <vector>
#include "Utils.h"

namespace pipn{    

    class multiStreamTrt{
        //TO-DO: Replace pointers with smart pointers ********
        public:
            //multiStreamTrt(const string args);
            multiStreamTrt();
            bool build();
            bool infer(std::string number);
            bool saveEngines();

            //To-DO *****************
            bool getSerializedEngines(std::vector<std::string> enginePath);
            bool teardown();

        private:
            bool launchInference(std::vector<float> const & inputTensor, std::vector<float>& outputTensor, nvinfer1::ICudaEngine* engine, void*inputBinding, void* outputBinding);
            int batchSize = 1; 
            int nNetworks = 2;
            std::vector<nvinfer1::ICudaEngine*> mEngines;
            void softmax(std::vector<float>& tensor);
            bool splitNetwork(nvinfer1::INetworkDefinition* network, std::vector<nvinfer1::INetworkDefinition*>& splittedNetworks, nvinfer1::IBuilder* builder);
            nvinfer1::ILayer* addLayerToNetwork(nvinfer1::INetworkDefinition*& network, nvinfer1::ILayer* layer,nvinfer1::ITensor* input);
            
            //To-DO *****************
            bool serializeEngines(); 
        
    };
}
#endif