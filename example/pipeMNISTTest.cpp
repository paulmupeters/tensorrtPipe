#include <iostream>
#include "PipeTrt.h"
#include "NvInfer.h"
using namespace std;




int main(int argc, char** argv)
{
    
    string number = "5";
    if (argc > 1)
        number = string(argv[1]);


    multiStreamTrt sample;
  
    if (!sample.build()){
        cout<<"failed to build"<<endl;
        return 0;
    }
    cout<<"succes"<<endl;

    if (!sample.infer(number)){
        cout<<"inference failed"<<endl;
        return 0;
    }

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