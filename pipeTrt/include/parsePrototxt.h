#ifndef PROTOTXT_PARSER_H
#define PROTOTXT_PARSER_H
#include "caffe/caffe.hpp"
#include <string>
#include <list>
#include <set>

namespace proto_parse{

    enum layerType{
        Convolution,
        Pooling,
        Relu,
        Normalization,
        SoftMax,
        FullyConnected,
        Input,
        Output
    };
    enum PoolMethod {
        MAX = 0,
        AVE = 1,
        STOCHASTIC = 2
    
    };
    struct Weights{
            Weights(const float* iValues, int iCount):values(iValues), count(iCount){};
            Weights(const Weights& w): values(w.values), count(w.count){} ;
            Weights():values(nullptr){};
            const float* values;
            size_t count;
    };
    class layerInfoBase {
        public:
            layerInfoBase(layerType itype, std::string ilayerName, int iId):type(itype), layerName(ilayerName), id(iId){};
        
            layerType type;
            std::string layerName;
            int id;
            std::vector<int> outputShape;
    };
    class CaffeParser{
        public:
            CaffeParser(const std::string& deploy, const std::string& model);
            int nbLayers(){return layersList.size();};
            std::list<layerInfoBase*> layersList;
        private:
            bool getParameters(caffe::Net<float>& caffe_net,const std::string& weightsFile);
            bool getOutShapes(const std::vector<std::vector<caffe::Blob<float>*>>& outBlobs);
            bool getBlobWeights(Weights& w, Weights& b,const caffe::shared_ptr<caffe::Layer<float>> layer, int numOutputs);
            void ParseConvLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParsePoolingLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseFullyConnectedLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseReluLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseSoftMaxLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseInputLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);

            using parsingFunction = void(CaffeParser::*)(const caffe::shared_ptr<caffe::Layer<float>> layer);
            static const std::map<std::string, parsingFunction> layerNameToParsFunc;
            int nOutputs;            
            
    };
    class convolutionLayerInfo :public layerInfoBase{
        public:
            convolutionLayerInfo (int iId, std::string iname, int iFeatureMaps, int iKernelSize, Weights iKernWeights,
              Weights iBiasWeights, int iOutputs=1): layerInfoBase(Convolution, iname, iId), nbOutputs(iOutputs), 
                                    nbFeatureMaps(iFeatureMaps), kernelSize(iKernelSize), kernelWeights(iKernWeights),
                                    biasWeights(iBiasWeights){};
            int getNbOutputs(){return nbOutputs;};
            int getNbFeatureMaps(){return nbFeatureMaps;};
            int getKernelSize(){return kernelSize;};
            Weights getKernelWeights(){return kernelWeights;};
            Weights getBiasWeights(){return biasWeights;};
        private:
            int nbOutputs;
            int nbFeatureMaps;
            int kernelSize;
            Weights kernelWeights;
            Weights biasWeights;
    };



    class reluLayerInfo :public layerInfoBase{
        public:
            reluLayerInfo(int iId, std::string iname):layerInfoBase(Relu, iname, iId){};
    }; 

    class inputLayerInfo :public layerInfoBase{
        public:
            inputLayerInfo(int iId, std::string iname, int channels, int h, int w, int n) : 
            layerInfoBase(Input, iname, iId){
                outputShape.push_back(n);
                outputShape.push_back(channels);
                outputShape.push_back(h); 
                outputShape.push_back(w);
            };
            //std::vector<int> shape;
        
            
    };

    class softMaxLayerInfo :public layerInfoBase{
        public:
            softMaxLayerInfo(int iId, std::string iname) :layerInfoBase(SoftMax, iname, iId){};
    };

    class fullyConnectedLayerInfo :public layerInfoBase {
        public:
            fullyConnectedLayerInfo(int iId, std::string iname, int fm, Weights iKernWeights,
              Weights iBiasWeights): layerInfoBase(FullyConnected, iname, iId), nbFeatureMaps(fm), 
              kernelWeights(iKernWeights), biasWeights(iBiasWeights){};
            
            int getnbFeatureMaps(){return nbFeatureMaps;};
            Weights getKernelWeights(){return kernelWeights;};
            Weights getBiasWeights(){return biasWeights;};     
        private:
            int nbFeatureMaps;
            Weights kernelWeights;
            Weights biasWeights;
    };

    class poolingLayerInfo :public layerInfoBase{
        public:
            poolingLayerInfo(int iId, std::string iname,int kernelSize, PoolMethod method = MAX): layerInfoBase(Pooling, iname, iId),
                kernelSize(kernelSize), poolingType(method){};
        int getKernelSize(){return kernelSize;};
        private:
            int kernelSize; // pool over a kernelSize * kernelSize region
            PoolMethod poolingType; // max ave stochastic
    };



 
}

#endif