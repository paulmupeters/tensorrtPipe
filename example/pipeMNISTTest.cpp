/*
Example of how to use pipeTrt, In this example the caffe parser is used to create the tensorrt network.
*/
#include <iostream>
#include <cassert>
#include "multiStream.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
using namespace std;




int main(int argc, char** argv)
{
    // Get input arguments
    string number = "5";
    if (argc > 1)
        number = string(argv[1]);
    string dataDir = "example/data/";


    // create tensorrt network, builder and builder configuration
    static Logger log(0);
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(log);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config ->setMaxWorkspaceSize(8 << 20);
    builder->setMaxBatchSize(1);
    
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
    parser->parse((dataDir + "mnist.prototxt").c_str() ,(dataDir + "mnist.caffemodel").c_str(), *network, nvinfer1::DataType::kFLOAT);
    assert(network->getNbLayers() > 0  && "Network was not parsed correctly");
    cout<<"parsed succes"<<endl;

    
    //multiStreamTrt sample(network, builder, config, false, 2);
    multiStreamTrt sample(network, builder, config, false, network->getNbLayers());


    // Prepare input
    const size_t INPUT_C = sample.getNChannels();
    const size_t INPUT_H = network->getInput(0)->getDimensions().d[1];
    const size_t INPUT_W = network->getInput(0)->getDimensions().d[2];
    //const size_t output_size = network->getInput(0)->getDimensions().d; 

    builder -> destroy();
    config -> destroy();
    network -> destroy();

    //read pgm files and fill input tensor
    std::vector<std::string> dirs{"example/data/", "data/", "../example/data/"};
    std::vector<float> inputVec;
    std::vector<float> outputVec(10);
    readPGMFile(locateFile(number + ".pgm", dirs), inputVec, INPUT_H, INPUT_W);


    for (int i = 0; i<100; i++){
        readPGMFile(locateFile(to_string(i%10) + ".pgm", dirs), inputVec, INPUT_H, INPUT_W);
        sample.launchInference(inputVec);
    }
    //if(){
    
    //}
    
    //sample.teardown();
    return 0;
}
/*
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (args.help)
    {
        std::cout<<"no help funciton implemented"<<endl;
        return 1;
    }
    if (!argsOK){
        gLogError << "Invalid arguments" << std::endl;
        return 0;
    }
    const char* gSampleName = "Paul";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);
    std::cout<<"gLogger created"<<endl;


    gLogInfo << "Building and running a GPU inference engine for MNIST" << std::endl;
*/