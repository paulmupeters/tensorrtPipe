#ifndef HPIPE_INCLUDED
#define HPIPE_INCLUDED
#include <string>
#include "Utils.h"

namespace pipes{    

    class multiStreamTrt{
        //TO-DO: Replace pointers with smart pointers ********
        template <typename T>
        using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>; 
        public:
            multiStreamTrt(const string args);
            bool build();
            bool infer(string number);


            //To-DO *****************
            bool getSerializedEngines(vector<string> enginePath);
            bool teardown();

        private:
            bool launchInference(vector<float> const & inputTensor, vector<float>& outputTensor, nvinfer1::ICudaEngine* engine, void*inputBinding, void* outputBinding);
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