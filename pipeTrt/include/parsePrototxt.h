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
            bool getOutShapes(const std::vector<caffe::Blob<float>*>& outBlobs);
            bool getBlobWeights(Weights& w, Weights& b,const caffe::shared_ptr<caffe::Layer<float>> layer, int numOutputs);
            void ParseConvLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParsePoolingLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseInnerProductLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseReluLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseSoftMaxLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);
            void ParseInputLayer(const caffe::shared_ptr<caffe::Layer<float>> layer);

            using parsingFunction = void(CaffeParser::*)(const caffe::shared_ptr<caffe::Layer<float>> layer);
            static const std::map<std::string, parsingFunction> layerNameToParsFunc;
            
            
    };


    struct Weights{
            //Weights();
            Weights(const float* iValues, int iCount):values(iValues), count(iCount){};
            Weights(const Weights& w): values(w.values), count(w.count){} ;
            Weights():values(nullptr){};
            const float* values;
            int count;
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
                shape[0] = n;
                shape[1] = channels;
                shape[2] = h;
                shape[3] = w;
            };
        private:
            int shape[4];
    };

    class softMaxLayerInfo :public layerInfoBase{
        public:
            softMaxLayerInfo(int iId, std::string iname) :layerInfoBase(SoftMax, iname, iId){};
    };

    class innerProductLayerInfo :public layerInfoBase {
        public:
            innerProductLayerInfo(int iId, std::string iname, int inOutputs, Weights iKernWeights,
              Weights iBiasWeights): layerInfoBase(FullyConnected, iname, iId), nbOutputs(inOutputs), 
              kernelWeights(iKernWeights), biasWeights(iBiasWeights){};
        
        private:
            int nbOutputs;
            Weights kernelWeights;
            Weights biasWeights;
    };

    class poolingLayerInfo :public layerInfoBase{
        public:
            poolingLayerInfo(int iId, std::string iname,int kernelSize, PoolMethod method = MAX): layerInfoBase(Pooling, iname, iId),
                kernelSize(kernelSize), poolingType(method){};
        private:
            int kernelSize; // pool over a kernelSize * kernelSize region
            PoolMethod poolingType; // max ave stochastic
    };



 
}

#endif