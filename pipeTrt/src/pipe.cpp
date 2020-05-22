#include "pipe.h"
#include <algorithm>
#include <numeric>

/*

Pipe::~Pipe(){
    cudaStreamDestroy(mStream);
    CHECK(cudaFree(mBindings[0]));
    CHECK(cudaFree(mBindings[1]));
    if (threadingActivated)
        mThread.join();
    std::cout<<"Thread joined, exiting pipe "<<pipeName<<std::endl;
    std::cout<<"Elapsed time: "<<totalTime<<std::endl<<std::endl;;
}

void Pipe::launchInference(){
    do{
        if(!inputQueue.empty()){
            std::vector<float> const inputTensor = inputQueue.front();
            std::vector<float> outputTensor(output_size);
            inputQueue.pop();
            float elapsedTime;
            cudaEventRecord(start, mStream);
            // Copy Input Data to the GPU
            cudaMemcpyAsync(mBindings[0], inputTensor.data(), batchSize * input_size * sizeof(float), 
                            cudaMemcpyHostToDevice, mStream);
            // Launch an instance of the GIE compute kernel
            executionContext->enqueue(batchSize, mBindings, mStream, nullptr);
            // Copy Output Data to the Host
            cudaMemcpyAsync(outputTensor.data(), mBindings[1], batchSize * output_size * sizeof(float), 
                            cudaMemcpyDeviceToHost, mStream);
            cudaEventRecord(end, mStream);
            cudaStreamSynchronize(mStream);
            cudaEventElapsedTime(&elapsedTime, start,end);
            totalTime += elapsedTime;    

            if(isLast){
                Output newOutput(outputTensor);
                netOutputs.push_back(newOutput);
                //int max_element = arg_max(outputTensor);
                //std::cout << "classified as: "<< max_element<<std::endl;
            }
            else{
                nextPipe->infer(outputTensor);
            }
        }
    }while((!mTerminate || !inputQueue.empty()) && threadingActivated );
    threadRunning = false;
}

void Pipe::infer(std::vector<float> const inputTensor){
    inputQueue.push(inputTensor);
    if(!threadRunning && threadingActivated){
        mThread = std::thread(&Pipe::launchInference, this);
        threadRunning = true;
    }
    if(!threadingActivated)
        launchInference();
}


void Pipe::inferCPU(std::vector<float> const inputTensor){
    // Find the binding points for the input and output nodes
    armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
    armnnCaffeParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("prob");



    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    //armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuAcc);
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, {armnn::Compute::CpuRef}, runtime->GetDeviceSpec());

     // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    // Run a single inference on the test image
    std::array<float, 10> output;
    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
                                                 MakeInputTensors(inputBindingInfo, &input->image[0]),
                                                 MakeOutputTensors(outputBindingInfo, &output[0]));

    // Convert 1-hot output to an integer label and print
    int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "Predicted: " << label << std::endl;
    std::cout << "   Actual: " << input->label << std::endl;
}

void Pipe::setNextPipe(std::shared_ptr<Pipe> next){
    nextPipe = next; 
}

bool Pipe::getOutput(std::vector<float>& outputTensor){
    if(!isLast){
        return false;
    }
    if(netOutputs.back().accesed){
        return false;
    }
    outputTensor = netOutputs.back().output;
    return true;
}

bool Pipe::terminate(){
    mTerminate = true;
    while(threadRunning)
    {
        //wait
    }
    if(nextPipe)
        return nextPipe->terminate();
    
    return true;
}


void Pipe::allocateGpu(nvinfer1::ICudaEngine* engine){
        for (int i = 0; i < engine->getNbBindings(); ++i){
        nvinfer1::Dims dims{engine->getBindingDimensions(i)};
        size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, batchSize, std::multiplies<int>());
        // Create CUDA buffer for Tensor
        cudaMalloc(&mBindings[i], size * sizeof(float));
        
        if (engine->bindingIsInput(i))
            input_size = size;
        else
            output_size = size;
    }
}

*/