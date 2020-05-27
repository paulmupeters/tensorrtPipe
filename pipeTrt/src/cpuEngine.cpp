#include "cpuEngine.h"




cpuEngine::cpuEngine(std::list <proto_parse::layerInfoBase*> ilayersList, int id , std::string name)
    : layersList(ilayersList), graph(id, name){
    const arm_compute::TensorShape input_shape(layersList.front()->outputShape[3],layersList.front()->outputShape[2],
        layersList.front()->outputShape[1]);
    graph << arm_compute::graph::Target::NEON << arm_compute::graph::FastMathHint::Disabled;
    createNetwork();
}
void cpuEngine::createNetwork(){   
    for(auto const& i: layersList){
        graph.add_layer(*toCLLayer(i));
    }
    graph<<arm_compute::graph::frontend::OutputLayer(get_output_accessor(layersList.back()->outputShape));
}
arm_compute::graph::frontend::ILayer* cpuEngine::toCLLayer(proto_parse::layerInfoBase* layer){
    switch(layer->type){
        case proto_parse::layerType::Convolution:{ 
         
            proto_parse::convolutionLayerInfo* info = (proto_parse::convolutionLayerInfo*)layer;
            int kernelSize = info->getKernelSize();
            arm_compute::graph::frontend::ConvolutionLayer *conv = new arm_compute::graph::frontend::ConvolutionLayer(
                kernelSize, kernelSize, 
                info->getNbFeatureMaps(),
                get_caffe_accessor(info->getKernelWeights()), 
                get_caffe_accessor(info->getBiasWeights()),
                arm_compute::PadStrideInfo(1,1,0,0)); 
            conv->set_name(info->layerName);
            return conv;
        }
        case proto_parse::layerType::Pooling:{

            proto_parse::poolingLayerInfo* info = (proto_parse::poolingLayerInfo*)layer;
            arm_compute::graph::frontend::PoolingLayer* pool = new arm_compute::graph::frontend::PoolingLayer(
                arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX,
                info->getKernelSize(), arm_compute::DataLayout::NCHW));
            pool -> set_name(info->layerName);
            return pool;
        }
        case proto_parse::layerType::Input:{

            proto_parse::inputLayerInfo* info = (proto_parse::inputLayerInfo*)layer;
            arm_compute::TensorShape shape(info->outputShape[3], info->outputShape[2], info->outputShape[1], info->outputShape[3]);
            arm_compute::graph::TensorDescriptor descriptor(shape, arm_compute::DataType::F32);
            arm_compute::graph::frontend::InputLayer* input = new arm_compute::graph::frontend::InputLayer(
                descriptor, get_input_accessor(info->outputShape));
            input -> set_name(info->layerName);
            return input;
        }
        case proto_parse::layerType::FullyConnected:{
            
            proto_parse::fullyConnectedLayerInfo* info = (proto_parse::fullyConnectedLayerInfo*)layer;
            arm_compute::graph::frontend::FullyConnectedLayer* fc = new arm_compute::graph::frontend::FullyConnectedLayer(
                info->getnbFeatureMaps(), 
                get_caffe_accessor(info->getKernelWeights()),
              get_caffe_accessor(info->getBiasWeights()));
            fc -> set_name(info->layerName);
            return fc;
        }
        case proto_parse::layerType::SoftMax:{
            arm_compute::graph::frontend::SoftmaxLayer* softMax = new arm_compute::graph::frontend::SoftmaxLayer();
            softMax -> set_name(layer->layerName);
            return softMax;
        }
        case proto_parse::layerType::Relu:{
            arm_compute::graph::frontend::ActivationLayer * relu = new arm_compute::graph::frontend::ActivationLayer(arm_compute::ActivationLayerInfo(
                arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
            relu -> set_name(layer->layerName);            
            return relu;
        }
        case proto_parse::layerType::Normalization:{
            arm_compute::graph::frontend::NormalizationLayer* norm = new arm_compute::graph::frontend::NormalizationLayer(
                arm_compute::NormalizationLayerInfo(arm_compute::NormType::IN_MAP_1D));
            
            return norm;
        }
        case proto_parse::layerType::Output:{
            arm_compute::graph::frontend::OutputLayer* output = new arm_compute::graph::frontend::OutputLayer(get_output_accessor(layer->outputShape));
            output->set_name(layer->layerName);
            return output;
        }
        default:{
            arm_compute::graph::frontend::OutputLayer* output = new arm_compute::graph::frontend::OutputLayer(get_output_accessor(layer->outputShape));
            output->set_name(layer->layerName);
            return output;
        }
    }
}
void cpuEngine::runInference(std::vector<float>& input){
 InputAccessor* inp = (InputAccessor*)graph.graph().tensors().front()->accessor();
 inp->addInput(input);
 graph.run();
}
InputAccessor::InputAccessor(std::vector<int> ishape):shape(ishape){
    inSize = ishape[1]*ishape[2]*ishape[3];
}
void InputAccessor::addInput(std::vector<float>& in){
    assert(inSize == in.size());
    inQueue.push(in);
}

bool InputAccessor::access_tensor(arm_compute::ITensor &tensor){
    if (inQueue.empty())
        return false;
    std::copy_n(inQueue.front().data(), tensor.info()->total_size(), tensor.buffer());
    inQueue.pop();
    return true;
}
OutputAccessor::OutputAccessor(std::vector<int> inshape):shape(inshape){
}
bool OutputAccessor::access_tensor(arm_compute::ITensor &tensor){
    assert(shape.size() == tensor.info()->num_dimensions());

    std::vector<float>      outputVec;
    size_t tensor_size = tensor.info()->tensor_shape().total_size();

    const auto   output_net  = reinterpret_cast<float *>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());
    outputVec.resize(tensor_size);
    std::copy(output_net, output_net + tensor_size, outputVec.begin());
    outQueue.push(outputVec);
    int max_element = arg_max(outputVec);
    std::cout<<"Predicted: " <<max_element<<std::endl;
    return true;
}


CaffeWeightsAccessor::CaffeWeightsAccessor(proto_parse::Weights iWeights):weights(iWeights){

}
bool CaffeWeightsAccessor::access_tensor(arm_compute::ITensor &tensor){
    assert(weights.count == tensor.info()->total_size());
    if(!tensor.info()->has_padding())
        std::copy_n(weights.values, tensor.info()->total_size(), tensor.buffer());
    else
        return false;
    return true;
}