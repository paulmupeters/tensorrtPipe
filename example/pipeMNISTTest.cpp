#include <iostream>
#include "PipeTrt.h"
using namespace std;
int main(int argc, char** argv)
{

    std::cout<<"creating  trtpipe object"<<endl;
    pipn:: multiStreamTrt();
    

    /*
    samplesCommon::Args args;
    string number = "5";
    if (argc > 1)
        number = string(argv[1]);


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

    MNISTmultiStream sample;
    gLogInfo << "Building and running a GPU inference engine for MNIST" << std::endl;

    if (!sample.build()){
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer(number)){
        return gLogger.reportFail(sampleTest);
    }
    return gLogger.reportPass(sampleTest);
    */
    
}

