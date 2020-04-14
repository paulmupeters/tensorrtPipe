//! \file multiStream.cpp
//! 
//! \brief source file of multiStream.h
#include "multiStream.h"
#include <memory> 
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <string>
#include <iostream>

 




multiStreamTrt::multiStreamTrt(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
     bool multiThread=true, int nNetworks):multiThreading(multiThread), nNetworks(nNetworks){
    mEngine = builder->buildEngineWithConfig(*network,*config);
    inputDims = network-> getInput(0)->getDimensions();
    //outputDims = network-> getOutput(0)->getDimensions();
    buildEngines(network, builder, config);

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


void multiStreamTrt::split(nvinfer1::INetworkDefinition* network, std::vector<nvinfer1::INetworkDefinition*>& splittedNetworks,
  std::queue<int>& splittingPoints){
    int index = 0;
    nvinfer1::ILayer* curLayer;
    nvinfer1::ILayer* newLayer;
    int nbLayers = network -> getNbLayers();
    for(int i =0; i <nbLayers; i++){
        curLayer = network ->getLayer(i);
        if (i == splittingPoints.front()){
            nvinfer1::ITensor* data = splittedNetworks[index] -> addInput( curLayer -> getInput(0)->getName() , curLayer -> getInput(0)->getType(), 
                curLayer -> getInput(0)->getDimensions());
            assert(data && "input tensor not found\n");
            newLayer = addLayerToNetwork(splittedNetworks[index], curLayer, data);
            splittingPoints.pop();
        }
        else{
            newLayer = addLayerToNetwork(splittedNetworks[index], curLayer, curLayer -> getInput(0));
        }
        if (i == splittingPoints.front() -1 || i == nbLayers-1){
            splittedNetworks[index]->markOutput(*newLayer -> getOutput(0));
            ++index;
        }

    }

}

// splits the network in different cuda engines
void multiStreamTrt::buildEngines(nvinfer1::INetworkDefinition* network, nvinfer1::IBuilder* builder,nvinfer1::IBuilderConfig* config){

    const int nbLayers = network -> getNbLayers();
    const int splitLength = nbLayers/nNetworks;    
    std::vector<nvinfer1::INetworkDefinition*> splittedNetworks; 
    std::string namePipe ="Pipe#";        
    std::queue<int> splittingPoints;

    for (int i=0; i<nNetworks; i++){
        splittedNetworks.push_back(builder->createNetwork());
        int point = i * splitLength;
        splittingPoints.push(point);
    }
    std::cout<<"splitting into: "<< nNetworks << " networks"<<std::endl;
    split(network, splittedNetworks, splittingPoints);
    
    for(int i=0; i<nNetworks; i++){
            splittedNetworks[i]->setName((namePipe + std::to_string(i+1)).c_str());
            nvinfer1::ICudaEngine* newEngine = builder->buildEngineWithConfig(*splittedNetworks[i],*config); 
            assert(nullptr!=newEngine && "New engine points to null");
            auto newPipe = std::make_shared<Pipe>(i == nNetworks-1, multiThreading, newEngine);
            if (i > 0)
                mPipes.back()->setNextPipe(newPipe);
            
            mPipes.push_back(newPipe);
    }
}

nvinfer1::ILayer* multiStreamTrt::addLayerToNetwork(nvinfer1::INetworkDefinition*& network, nvinfer1::ILayer* layer, nvinfer1::ITensor* input){

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