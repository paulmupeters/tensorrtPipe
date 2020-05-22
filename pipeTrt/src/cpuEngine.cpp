#include "cpuEngine.h"




cpuEngine::cpuEngine(proto_parse::CaffeParser parsed, std::set<int> layerIndexes){

    const arm_compute::TensorShape input_shape = (parsed.layerList.front());
    graph << arm_compute::graph::Target::NEON << arm_compute::graph::FastMathHint::Disabled;
}





void cpuEngine::createNetwork(){
    proto_parse::CaffeParser parsed(deploy, model);
    for(auto const& i: parsed.layersList){
        if(container.find(i->id) != container.end());
        cout<<endl<<"layer name: "<<i->layerName<<"layer id: "<<i->id<<endl;
        graph.add_layer(toCLLayer(i));
    }
}


arm_compute::graph::frontend::ILayer cpuEngine::toCLLayer(proto_parse::layerInfoBase* layer){
    switch(layer->type){
        case proto_parse::layerType::Convolution:{ 
            return ConvolutionLayer(layer->kernelSize,layer->kernelSize,layer->nbOutputMaps,
              get_accessor(layer->kernelWeights ), get_accessor(layer->biasWeights)).set_name(layer->name); 
        }
        case proto_parse::layerType::Pooling:{

        }
        case proto_parse::layerType::Input:{
            arm_compute::TensorShape shape(layer->n, layer->c, layer->h, layer->w);
            TensorDescriptor descriptor(shape, arm_compute::DataType::F32);
            return InputLayer(descriptor, arm_compute::support::cpp14::make_unique<DummyAccessor>()).set_name(layer->name);
        }
        case proto_parse::layerType::FullyConnected:{
            return FullyConnectedLayer(layer->nbOutputs);
        }
        case proto_parse::layerType::SoftMax:{

        }
        case proto_parse::layerType::Relu:{

        }
        case proto_parse::layerType::Input:{

        }

    }

}

graph.add_layer()
 << common_params.target
        << common_params.fast_math_hint
        << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
        // Layer 1
        << ConvolutionLayer(
            11U, 11U, 96U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
            PadStrideInfo(4, 4, 0, 0))
        .set_name("conv1")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
        << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
        << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool1")
        // Layer 2
        << ConvolutionLayer(
            5U, 5U, 256U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
            PadStrideInfo(1, 1, 2, 2), 2)
        .set_name("conv2")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
        << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
        << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool2")
        // Layer 3
        << ConvolutionLayer(
            3U, 3U, 384U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
            PadStrideInfo(1, 1, 1, 1))
        .set_name("conv3")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3")
        // Layer 4
        << ConvolutionLayer(
            3U, 3U, 384U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
            PadStrideInfo(1, 1, 1, 1), 2)
        .set_name("conv4")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4")
        // Layer 5
        << ConvolutionLayer(
            3U, 3U, 256U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
            PadStrideInfo(1, 1, 1, 1), 2)
        .set_name("conv5")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
        << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool5")
        // Layer 6
        << FullyConnectedLayer(
            4096U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
        .set_name("fc6")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6")
        // Layer 7
        << FullyConnectedLayer(
            4096U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
        .set_name("fc7")
        << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7")
        // Layer 8
        << FullyConnectedLayer(
            1000U,
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy", weights_layout),
            get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
        .set_name("fc8")
        // Softmax
        << SoftmaxLayer().set_name("prob")
        << OutputLayer(get_output_accessor(common_params, 5));




RandomAccessor::RandomAccessor(PixelValue lower, PixelValue upper, std::random_device::result_type seed)
    : _lower(lower), _upper(upper), _seed(seed)
{
}

template <typename T, typename D>
void RandomAccessor::fill(ITensor &tensor, D &&distribution)
{
    std::mt19937 gen(_seed);

    if(tensor.info()->padding().empty() && (dynamic_cast<SubTensor *>(&tensor) == nullptr))
    {
        for(size_t offset = 0; offset < tensor.info()->total_size(); offset += tensor.info()->element_size())
        {
            const auto value                                 = static_cast<T>(distribution(gen));
            *reinterpret_cast<T *>(tensor.buffer() + offset) = value;
        }
    }
    else
    {
        // If tensor has padding accessing tensor elements through execution window.
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto value                                  = static_cast<T>(distribution(gen));
            *reinterpret_cast<T *>(tensor.ptr_to_element(id)) = value;
        });
    }
}

/** weights accessor class */
class CaffeWeightsAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] lower Lower bound value.
     * @param[in] upper Upper bound value.
     * @param[in] seed  (Optional) Seed used to initialise the random number generator.
     */
    CaffeWeightsAccessor(parse_proto::Weights caffeWeights);
    /** Allows instances to move constructed */
    CaffeWeightsAccessor(CaffeWeightsAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T, typename D>
    void fill(ITensor &tensor, D &&distribution);
};

bool CaffeWeightsAccessor::access_tensor(ITensor &tensor)
{
    switch(tensor.info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            std::uniform_int_distribution<uint8_t> distribution_u8(_lower.get<uint8_t>(), _upper.get<uint8_t>());
            fill<uint8_t>(tensor, distribution_u8);
            break;
        }
        case DataType::S8:
        {
            std::uniform_int_distribution<int8_t> distribution_s8(_lower.get<int8_t>(), _upper.get<int8_t>());
            fill<int8_t>(tensor, distribution_s8);
            break;
        }
        case DataType::U16:
        {
            std::uniform_int_distribution<uint16_t> distribution_u16(_lower.get<uint16_t>(), _upper.get<uint16_t>());
            fill<uint16_t>(tensor, distribution_u16);
            break;
        }
        case DataType::S16:
        {
            std::uniform_int_distribution<int16_t> distribution_s16(_lower.get<int16_t>(), _upper.get<int16_t>());
            fill<int16_t>(tensor, distribution_s16);
            break;
        }
        case DataType::U32:
        {
            std::uniform_int_distribution<uint32_t> distribution_u32(_lower.get<uint32_t>(), _upper.get<uint32_t>());
            fill<uint32_t>(tensor, distribution_u32);
            break;
        }
        case DataType::S32:
        {
            std::uniform_int_distribution<int32_t> distribution_s32(_lower.get<int32_t>(), _upper.get<int32_t>());
            fill<int32_t>(tensor, distribution_s32);
            break;
        }
        case DataType::U64:
        {
            std::uniform_int_distribution<uint64_t> distribution_u64(_lower.get<uint64_t>(), _upper.get<uint64_t>());
            fill<uint64_t>(tensor, distribution_u64);
            break;
        }
        case DataType::S64:
        {
            std::uniform_int_distribution<int64_t> distribution_s64(_lower.get<int64_t>(), _upper.get<int64_t>());
            fill<int64_t>(tensor, distribution_s64);
            break;
        }
        case DataType::F16:
        {
            std::uniform_real_distribution<float> distribution_f16(_lower.get<half>(), _upper.get<half>());
            fill<half>(tensor, distribution_f16);
            break;
        }
        case DataType::F32:
        {
            std::uniform_real_distribution<float> distribution_f32(_lower.get<float>(), _upper.get<float>());
            fill<float>(tensor, distribution_f32);
            break;
        }
        case DataType::F64:
        {
            std::uniform_real_distribution<double> distribution_f64(_lower.get<double>(), _upper.get<double>());
            fill<double>(tensor, distribution_f64);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
    return true;
}
