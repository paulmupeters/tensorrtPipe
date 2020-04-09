//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
#include "multiStream.h"
//#include <queue>
#include <memory> 
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
//#include <cmath>
//#include <fstream>
#include <iostream>
//#include <sstream>
//#include <numeric>
 




multiStreamTrt::multiStreamTrt(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config){
    mEngine = builder->buildEngineWithConfig(*network,*config);
    inputDims = network-> getInput(0)->getDimensions();
    //outputDims = network-> getOutput(0)->getDimensions();
    splitNetwork(network, builder, config);

    std::cout<<"build succes"<<std::endl;
}

multiStreamTrt::~multiStreamTrt(){
    teardown();
}


void multiStreamTrt::launchInference(std::vector<float> input){
    mPipes[0]->infer(input);
}

bool multiStreamTrt::getOutput(std::vector<float>& output){
    if(!mPipes[nNetworks-1]->getOutput(output))
        return false;
    int max_element = arg_max(output);
    std::cout << "classified as: "<< max_element<<std::endl;
    return true;
}

size_t multiStreamTrt::getNChannels(){return inputDims.d[0];}
size_t multiStreamTrt::getHeight(){return inputDims.d[1];}
size_t multiStreamTrt::getWidth(){ return inputDims.d[2];}
//size_t multiStreamTrt::getOutputSize(){
//    return(outputDims.d[0] * outputDims.d[1] * outputDims.d[2]);
//}

// splits the network in different cuda engines
void multiStreamTrt::splitNetwork(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder,nvinfer1::IBuilderConfig* config){
    std::cout<<"splitting into: "<< nNetworks << " networks"<<std::endl;
    
    std::vector<nvinfer1::INetworkDefinition*> splittedNetworks; 
    for (int i=0; i<nNetworks; i++)
       splittedNetworks.push_back(builder->createNetwork());     
    const int nbLayers = network -> getNbLayers();
    int index = 0;
    int splitsNeeded = nNetworks;
    const int split = nbLayers/nNetworks;
    nvinfer1::ILayer* curLayer;
    nvinfer1::ILayer* newLayer;

    
    for (int i =  0; i <nbLayers; i++ ){
        curLayer = network -> getLayer(i);
        if (i % split == 0 && splitsNeeded > 0){
            nvinfer1::ITensor* data = splittedNetworks.back() -> addInput( curLayer -> getInput(0)->getName() , curLayer -> getInput(0)->getType(), 
                curLayer -> getInput(0)->getDimensions());
            assert(data && "input tensor not found\n");
            newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, data);
            splitsNeeded -=1;
        }
        else if ((i % split == split-1 && splitsNeeded > 0 )|| i == nbLayers-1){
            newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, curLayer -> getInput(0));
            splittedNetworks.back()->markOutput(*newLayer -> getOutput(0));
        
            nvinfer1::ICudaEngine* newEngine = builder->buildEngineWithConfig(*splittedNetworks.back(),*config); 
            assert(nullptr!=newEngine && "New engine points to null");
            
            auto newPipe = std::make_shared<Pipe>(index == nNetworks-1, newEngine);
            if (index > 0)
                mPipes.back()->setNextPipe(newPipe);
            
            mPipes.push_back(newPipe);
            std::cout<<"Network added succesfully"<<std::endl;

            ++index;
            splittedNetworks.back()->destroy();
            splittedNetworks.pop_back();
        }else{
            newLayer = addLayerToNetwork(splittedNetworks.back(), curLayer, curLayer -> getInput(0));
            assert(newLayer && "new layer not added to network\n");
        }
    }

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


bool multiStreamTrt::teardown(){
    mPipes[0]-> terminate();
    return true;
}


/*
bool multiStreamTrt::seriealizeEngines(){
    std::string enginePath;
    try{
        for (int i =0; i < nNetworks; i++){
            unique_ptr<IHostMemory, Destroy> engine_plan{engine->serialize()};
            mEngine[i] -> serialize(); 
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);

        }
    } catch{
        std::cout<<"failed serialization"<<std::endl;
        return false;
    }
    return true;
}

bool multiStreamTrt::getSerializedEngines(string fileName){
    return true;
}


*/