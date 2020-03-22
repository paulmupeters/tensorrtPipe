//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
#include "PipeTrt.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "buffers.h"
#include <vector>
#include <queue>
#include <memory>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <exception>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>



template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

namespace pipes{
    multiStreamTrt::multiStreamTrt(const string args){
        cout<<"building"<< args <<"engines"<<endl;
        assert(build);
    }

    bool multiStreamTrt::build(){
        
        nvinfer1::IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
        builder->setMaxBatchSize(batchSize);
        builder->setMaxWorkspaceSize(8 << 20);

        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
        
        //change to dataDir variable -------------------------------------------->>><<<<
        parser->parse("../data/mnist.prototxt" ,"../../data/mnist.caffemodel", *network, nvinfer1::DataType::kFLOAT);
        assert(network->getNbLayers() > 0  && "Network was not parsed correctly");
        cout<<"parsed succes"<<endl;


        //network->markOutput(*blobNameToTensor->find("prob"));
        //ICudaEngine* newEngine = builder->buildCudaEngine(*network);    
        //assert(nullptr!=newEngine && "New engine points to null");


        vector<nvinfer1::INetworkDefinition*> splittedNetworks;
        for (int i = 0; i < nNetworks; i++)
            splittedNetworks.push_back(builder->createNetwork());
        
        
        bool splitted = splitNetwork(network ,splittedNetworks, builder);
        //serialize engines for future use 
        //assert(serializeEngines());
        builder -> destroy();
        return(splitted);
    }




    bool multiStreamTrt::infer(string number){
        void* inputBindings[nNetworks]{0};
        void* outputBindings[nNetworks]{0};

        vector<float> outputTensor;
        vector<vector<float>> outputs;
        

        const int inputIndex =0;
        const int outputIndex = 1;
        nvinfer1::Dims dims{mEngines[0]->getBindingDimensions(inputIndex)};
        const int INPUT_H = dims.d[1];
        const int INPUT_W = dims.d[2]; 
        size_t inputSize[nNetworks]{0};


        //First allocate memory on GPU
        for (int j = 0; j < nNetworks; j++){
            assert(mEngines[j]->bindingIsInput(0) ^ mEngines[j]->bindingIsInput(1));
            dims = {mEngines[j]->getBindingDimensions(inputIndex)};
            inputSize[j] = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<int>());
            // Create CUDA buffer for Tensor
            cudaMalloc(&inputBindings[j], inputSize[j] * sizeof(float));
            
            Dims dims{mEngines[j]->getBindingDimensions(outputIndex)};
            size_t outputSize = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<int>());
            // Create CUDA buffer for Tensor
            cudaMalloc(&outputBindings[j], outputSize * sizeof(float));
            
            outputTensor.resize(outputSize);
            outputs.push_back(outputTensor);           
        }

        //read pgm files and fill input tensor
        uint8_t buffer[INPUT_H * INPUT_W];
        std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/"};
        readPGMFile(locateFile(number + ".pgm", dirs), buffer, INPUT_H, INPUT_W);
        if (sizeof(buffer) != inputSize[0]){
            cout << "Couldn't read input Tensor" << endl; return false;
        }
        
        std::vector<float> inputVec(&buffer[0], &buffer[inputSize[0]]);
            
        for (int i = 0; i < nNetworks; i++){
            bool hidden = launchInference(inputVec, outputs[i],mEngines[i], inputBindings[i], outputBindings[i]); 
            if (i < nNetworks-1){
                if (outputs[i].size() != inputSize[i+1] || !hidden){
                    cout<<"Can not read hidden input"<<endl;
                    return false;
                }
                inputVec = outputs[i];
            }
                
        }
        //softmax(outputs[nNetworks]);
        int max_element = arg_max(outputs[nNetworks-1]);
        std::cout << "classified as: "<< max_element<<std::endl;
        return true;
    }

    bool multiStreamTrt::launchInference(vector<float> const& inputTensor, vector<float> & outputTensor, nvinfer1::ICudaEngine* engine, void* inputBinding, void* outputBinding){
        const int size_input = inputTensor.size() * sizeof(float);
        const int size_output= outputTensor.size() * sizeof(float);
        void * bindings[] = {inputBinding, outputBinding}; 
        

        // Create execution context and cuda streams
        // Use CUDA streams to manage the concurrency of copying and executing
        assert(engine-> getNbBindings() == 2 && "Number of bindings is not 2");
        //nvinfer1::IExecutionContext* execCont = engine -> createExecutionContext();
        std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> execCont{engine -> createExecutionContext()};
        //Create Cudastreams for execution of inference
        cudaStream_t newStream;
        CHECK(cudaStreamCreate(&newStream));
        
        
        // Copy Input Data to the GPU
        cudaMemcpyAsync(inputBinding, inputTensor.data(), batchSize * size_input, 
                        cudaMemcpyHostToDevice, newStream);
        
        // Launch an instance of the GIE compute kernel
        execCont->enqueue(batchSize, bindings, newStream, nullptr);

        // Copy Output Data to the Host
        cudaMemcpyAsync(outputTensor.data(), outputBinding, batchSize * size_output, 
                        cudaMemcpyDeviceToHost, newStream);

        // It is possible to have multiple instances of the code above
        // in flight on the GPU in different streams.
        // The host can then sync on a given stream and use the results
        cudaStreamSynchronize(newStream);
        cudaStreamDestroy(newStream);

        CHECK(cudaFree(inputBinding));
        CHECK(cudaFree(outputBinding));
        return true;

    }


    // splits the network in different cuda engines
    bool multiStreamTrt::splitNetwork(nvinfer1::INetworkDefinition* network,
    std::vector<nvinfer1::INetworkDefinition*>& splittedNetworks, nvinfer1::IBuilder* builder){
        std::cout<<"splitting into: "<< nNetworks << " networks"<<endl;
        try{ 
            const int nbLayers = network -> getNbLayers();
            int splitsNeeded = nNetworks;
            const int split = nbLayers/nNetworks;
            nvinfer1::ILayer* curLayer;
            nvinfer1::ILayer* newLayer;


            for (int i =  0; i <nbLayers; i++ ){
                curLayer = network -> getLayer(i);    
                if (i % split == 0 && splitsNeeded > 0){
                    ITensor* data = splittedNetworks.back() -> addInput( curLayer -> getInput(0)->getName() , curLayer -> getInput(0)->getType(),
                    curLayer -> getInput(0)->getDimensions());
                    assert(data && "input tensor not found\n");
                    
                    splitsNeeded -=1;
                    
                    newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, data);
                }
                else if ((i % split == split-1 && splitsNeeded > 0 )|| i == nbLayers-1){
                    newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, curLayer -> getInput(0));
                    splittedNetworks.back()->markOutput(*newLayer -> getOutput(0));

                    ICudaEngine* newEngine = builder->buildCudaEngine(*splittedNetworks.back()); 
                    assert(nullptr!=newEngine && "New engine points to null");
                    mEngines.push_back(newEngine);
                    cout<<"Network added succesfully"<<endl;


                    splittedNetworks.back()->destroy();
                    splittedNetworks.pop_back();
                }else{
                    newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, curLayer -> getInput(0));
                    assert(newLayer && "new layer not added to network\n");
                }
            }
        }
        catch(exception& e){
            std::cout<<"failed Splitting network"<<std::endl;
            std::cout << "Standard exception: " << e.what() << std::endl;
            return(false);
        }
        return(true);
    }

    nvinfer1::ILayer* multiStreamTrt::addLayerToNetwork(nvinfer1::INetworkDefinition*& network, nvinfer1::ILayer* layer, nvinfer1::ITensor* input){
        //nvinfer1::ITensor* input = layer->getInput(0);
        switch(layer->getType()){
            case nvinfer1::LayerType::kCONVOLUTION :{
                std::cout<<"Adding convolution layer."<<std::endl;
                nvinfer1::DimsHW kernelSize = ((nvinfer1::IConvolutionLayer *)layer) -> getKernelSize();
                nvinfer1::Weights kernelWeights = ((nvinfer1::IConvolutionLayer *)layer) -> getKernelWeights();
                nvinfer1::Weights biasWeights =  ((nvinfer1::IConvolutionLayer *)layer) -> getBiasWeights();
                int nbOutputMaps = ((nvinfer1::IConvolutionLayer *)layer) -> getNbOutputMaps();
                nvinfer1::IConvolutionLayer* newLayer = network->addConvolution(*input, nbOutputMaps, kernelSize, kernelWeights, biasWeights);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());
                return(newLayer);
                }
            case nvinfer1::LayerType::kSCALE:{
                std::cout<<"Adding scale layer."<<std::endl;
                nvinfer1::ScaleMode mode = ((nvinfer1::IScaleLayer *)layer) -> getMode();
                nvinfer1::Weights scale = ((nvinfer1::IScaleLayer *)layer) -> getScale();
                nvinfer1::Weights shift = ((nvinfer1::IScaleLayer *)layer) ->getShift();
                nvinfer1::Weights power = ((nvinfer1::IScaleLayer *)layer) -> getPower();
                nvinfer1::IScaleLayer* newLayer = network->addScale(*input, mode, shift, scale, power);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());   
                return(newLayer);
            }
            case nvinfer1::LayerType::kFULLY_CONNECTED:{
                std::cout<<"Adding fully connected layer."<<std::endl;
                nvinfer1::Weights kernelWeights = ((nvinfer1::IFullyConnectedLayer *)layer) -> getKernelWeights();
                nvinfer1::Weights biasWeights =  ((nvinfer1::IFullyConnectedLayer *)layer) -> getBiasWeights();
                int nbOutputs = ((nvinfer1::IFullyConnectedLayer *)layer) -> getNbOutputChannels();
                nvinfer1::IFullyConnectedLayer* newLayer = network -> addFullyConnected(*input,nbOutputs,kernelWeights, biasWeights);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());
                return(newLayer);
            }
            case nvinfer1::LayerType::kACTIVATION :{
                std::cout<<"Adding activation layer."<<std::endl;
                nvinfer1::ActivationType activationType = ((nvinfer1::IActivationLayer*)layer)->getActivationType() ;
                nvinfer1::IActivationLayer* newLayer = network ->addActivation(*input, activationType);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());
                return(newLayer);
            }
            case nvinfer1::LayerType::kPOOLING :{
                std::cout<<"Adding pooling layer."<<std::endl;
                nvinfer1::PoolingType poolingType = ((nvinfer1::IPoolingLayer*)layer) -> getPoolingType();
                nvinfer1::DimsHW stride = ((nvinfer1::IPoolingLayer*)layer) -> getWindowSize();
                nvinfer1::IPoolingLayer* newLayer = network-> addPooling(*input,poolingType,stride);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());
                return(newLayer);
            }
            case nvinfer1::LayerType::kSOFTMAX :{
                std::cout<<"Adding softmax layer."<<std::endl;
                nvinfer1::ISoftMaxLayer* newLayer = network->addSoftMax(*input);
                newLayer -> getOutput(0)->setName(layer->getOutput(0)->getName());
                return(newLayer);
                break;
            }
            default:{
                std::cout<<"adding default layer"<<std::endl;
                nvinfer1::ILayer* newLayer = layer;
                return(newLayer);
            }
        } 
    }

    /*
    bool multiStreamTrt::teardown(){
        return true;
    }
    */

    /*
    void multiStreamTrt::softmax(vector<float>& tensor)
    {
        size_t batchElements = tensor.size() / batchSize;

        for (int i = 0; i < batchSize; ++i)
        {
            float* batchVector = &tensor[i * batchElements];
            double maxValue = *max_element(batchVector, batchVector + batchElements);
            double expSum = accumulate(batchVector, batchVector + batchElements, 0.0, [=](double acc, float value) { return acc + exp(value - maxValue); });

            transform(batchVector, batchVector + batchElements, batchVector, [=](float input) { return static_cast(std::exp(input - maxValue) / expSum); });
        }
    }
    */

    /*
    bool multiStreamTrt::seriealizeEngines(){
        string enginePath;
        try{
            for (int i =0; i < nNetworks; i++){
                unique_ptr<IHostMemory, Destroy> engine_plan{engine->serialize()};
                mEngine[i] -> serialize(); 
                writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);

            }
        } catch{
            cout<<"failed serialization"<<endl;
            return false;
        }
        return true;
    }

    bool multiStreamTrt::getSerializedEngines(){
        return true;
    }
    */
}