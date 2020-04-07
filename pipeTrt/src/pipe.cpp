#include "pipe.h"
#include <algorithm>
#include <numeric>



Pipe::~Pipe(){
    cudaStreamDestroy(mStream);
    CHECK(cudaFree(mBindings[0]));
    CHECK(cudaFree(mBindings[1]));
    mThread.join();
}

void Pipe::launchInference(){
    cudaEvent_t start;
    cudaEvent_t end;
    
    while(!mTerminate){
        if(!inputQueue.empty()){
            std::vector<float> const inputTensor = inputQueue.front();
            std::vector<float> outputTensor;
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
            cudaEventElapsedTime(&elapsedTime, start, end);
            totalTime += elapsedTime;    

            if(isLast){
                Output newOutput(outputTensor);
                netOutputs.push_back(newOutput);
            }
            else{
                nextPipe->infer(outputTensor);
            }
        }
    }
}

void Pipe::infer(std::vector<float> const inputTensor){
    inputQueue.push(inputTensor);
    if(!threadRunning){
        mThread = std::thread(&Pipe::launchInference, this);
        threadRunning = true;
    }
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

void Pipe::terminate(){
    mTerminate = true;
    if(nextPipe)
        nextPipe->terminate();
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