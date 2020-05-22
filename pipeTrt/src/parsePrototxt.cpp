#include "parsePrototxt.h"
#include <vector>

namespace proto_parse{


    CaffeParser::CaffeParser(const std::string& deploy, const  std::string& model){
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        // Instantiate the caffe net.
        caffe::Net<float> caffe_net(deploy, caffe::TEST);
        assert(getParameters(caffe_net, model));
        assert(getOutShapes(caffe_net.output_blobs()));
    }

    bool CaffeParser::getOutShapes(const std::vector<caffe::Blob<float>*>& outBlobs){
        if(outBlobs.size() != layersList.size())
            return false;
        int i=0;
        for(const auto& layer : layersList){
            layer -> outputShape = outBlobs[i] -> shape();
            ++i;
        }
        return true;
    }
    bool CaffeParser::getParameters(caffe::Net<float>& caffe_net,const std::string& weightsFile){
        caffe_net.CopyTrainedLayersFrom(weightsFile);
        const caffe::vector<caffe::shared_ptr<caffe::Layer<float>>>& layers = caffe_net.layers();
        for (const auto& layer : layers) {
            try{
            auto it = layerNameToParsFunc.find(layer->type());
            if (it == layerNameToParsFunc.end())
                throw 10;
            auto func = it->second;
            (this->*func)(layer);
            } catch(int e){
                std::cout<<"coud not parse layer "<<layer->type()<<std::endl;
                return false;
            }
        }
        return true;
    }

    bool CaffeParser::getBlobWeights(Weights& w, Weights& b, const caffe::shared_ptr<caffe::Layer<float>> layer,  int numOutputs){
        const caffe::vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = layer->blobs();        
        if(blobs.size() <2)
            return false;
        for (const auto& blob : blobs) {
            int c = blob->count();
            if (c == numOutputs){ //biases
                b.values = blob->cpu_data();
                b.count = c;
            }
            else{
                w.values = blob->cpu_data();
                w.count = c;
            }
        }
        if(w.values && b.values)
            return true;
        else 
            return false;
    }

    void CaffeParser::ParseConvLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        const caffe::ConvolutionParameter conv_param = param.convolution_param();
        int kernelSize =conv_param.kernel_size().Get(0);
        int outs = (int)conv_param.num_output(); 

        Weights biasWeights;
        Weights kernelWeights;
        assert(getBlobWeights(kernelWeights, biasWeights, layer, outs), "could not extract convolutional Weights");

        layersList.push_back(new convolutionLayerInfo((int)layersList.size(), param.name(), outs, kernelSize,
          kernelWeights, biasWeights));
    }

    void CaffeParser::ParsePoolingLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        const caffe::PoolingParameter pool_param = param.pooling_param();
        layersList.push_back(new poolingLayerInfo(layersList.size(), param.name(),pool_param.kernel_size()));
    }

    void CaffeParser::ParseInnerProductLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        const caffe::InnerProductParameter& inner_product_param = param.inner_product_param();
        int outs =inner_product_param.num_output();

        Weights biasWeights;
        Weights kernelWeights;
        assert(getBlobWeights(kernelWeights, biasWeights,layer, outs), "could not extract fully connected Weights");

        layersList.push_back(new innerProductLayerInfo (layersList.size(), param.name(),outs,
          kernelWeights, biasWeights));   
        }
    }
    void CaffeParser::ParseReluLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        layersList.push_back(new reluLayerInfo(layersList.size(), param.name()));
    }
    void CaffeParser::ParseSoftMaxLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        layersList.push_back( new softMaxLayerInfo(layersList.size(), param.name()));
    }
    void CaffeParser::ParseInputLayer(const caffe::shared_ptr<caffe::Layer<float>> layer){
        const caffe::LayerParameter& param = layer->layer_param();
        int n = param.input_param().shape(0).dim(0);
        int c = param.input_param().shape(0).dim(1);
        int h = param.input_param().shape(0).dim(2);
        int w = param.input_param().shape(0).dim(3);
        layersList.push_back( new inputLayerInfo(layersList.size(), param.name(), c, h, w, n));
    }

    const std::map<std::string, CaffeParser::parsingFunction> CaffeParser::layerNameToParsFunc = {
    { "Input",        &CaffeParser::ParseInputLayer },
    { "Convolution",  &CaffeParser::ParseConvLayer },
    { "Pooling",      &CaffeParser::ParsePoolingLayer },
    { "ReLU",         &CaffeParser::ParseReluLayer },
    { "InnerProduct", &CaffeParser::ParseInnerProductLayer },
    { "Softmax",      &CaffeParser::ParseSoftMaxLayer },
    //{ "Concat",       &CaffeParser::ParseConcatLayer },
    //{ "BatchNorm",    &CaffeParser::ParseBatchNormLayer },
    //{ "Scale",        &CaffeParser::ParseScaleLayer },
    //{ "Dropout",      &CaffeParser::ParseDropoutLayer},
    };

}
