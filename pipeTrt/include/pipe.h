#include <vector>
#include <thread>
#include <queue>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "Utils.h"
#include <memory>
#include <string>


struct Output{
    Output(std::vector<float> outp) : output(outp)
    {
    }
    std::vector<float>& output;
    bool accesed = true;
};

class Pipe{
    public:
        Pipe(nvinfer1::ICudaEngine* engineI, bool lastI, bool threading): 
          isLast(lastI), threadingActivated(threading){
            allocateGpu(engineI);
            // Create execution context and cuda streams
            // Use CUDA streams to manage the concurrency of copying and executing
            //assert(engineI-> getNbBindings() == 2 && "Number of bindings is not 2");
            executionContext = engineI -> createExecutionContext();
            CHECKC(cudaStreamCreate(&mStream));
            cudaEventCreate(&start);
            cudaEventCreate(&end);

            inputQueue.setMaxSize(20);
            //pipeName = engineI->getName();
            
        };    
        ~Pipe();
        void infer(std::vector<float> const input);        
        bool getOutput(std::vector<float>&);
        void setNextPipe(std::shared_ptr<Pipe>);
        bool terminate();

    private:
        std::string pipeName;
        bool isLast;
        bool threadingActivated;
        bool threadRunning = false;
        bool mTerminate = false;
        std::thread mThread;
        std::vector<Output> netOutputs;
        MaxQueue<std::vector<float>> inputQueue;
        
        cudaEvent_t start;
        cudaEvent_t end;

        nvinfer1::IExecutionContext* executionContext;
        void allocateGpu(nvinfer1::ICudaEngine*);
        void launchInference();
        std::shared_ptr<Pipe> nextPipe{nullptr};
        cudaStream_t mStream;
        void* mBindings[2]{0};     
        size_t input_size;
        size_t output_size;
        size_t batchSize = 1;
        double dataTransferTime = 0.0;
        double totalTime = 0.0;

};
/*
class CpuPipe : public Pipe{
    GpuPipe( );
    public:
};

class GpuPipe : public Pipe{

    public:
        GpuPipe(){

        };

    private:
        
};
*/