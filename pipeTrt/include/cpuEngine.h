#include "parsePrototxt.h"
#include "Utils.h"
#include <memory>
#include <string>

//#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph.h"

/**
    Class for optimized cpu inference of CNN using arm compute library

*/
class InputAccessor final : public arm_compute::graph::ITensorAccessor{
    public:
        /** Constructor
        *

        * @param input vector to initialize tensor.
        */
        InputAccessor(std::vector<int> ishape);
        /** Allows instances to move constructed */
        InputAccessor(InputAccessor &&) = default;

        // Inherited methods overriden:
        bool access_tensor(arm_compute::ITensor &tensor) override;
        void addInput(std::vector<float>& in);

    private:
        MaxQueue <std::vector<float>> inQueue;
        std::vector<int> shape;
        size_t inSize;
};
/** outputs accessor class */
class OutputAccessor final : public arm_compute::graph::ITensorAccessor{
    public:
        /** Constructor
        *

        * @param input pointer to outputlist?
        */
        OutputAccessor(std::vector<int> inshape);
        /** Allows instances to move constructed */
        OutputAccessor(OutputAccessor &&) = default;

        // Inherited methods overriden:
        bool access_tensor(arm_compute::ITensor &tensor) override;
        MaxQueue <std::vector<float>> getOutQueue(){return outQueue;};
    private:
        MaxQueue <std::vector<float>> outQueue;
        std::vector<int> shape;
};  

/** weights accessor class */
class CaffeWeightsAccessor final : public arm_compute::graph::ITensorAccessor{
    public:
        /** Constructor
        *

        * @param input weigths  Weights to initialize tensor.
        */
        CaffeWeightsAccessor(proto_parse::Weights iWeights);
        /** Allows instances to move constructed */
        CaffeWeightsAccessor(CaffeWeightsAccessor &&) = default;

        // Inherited methods overriden:
        bool access_tensor(arm_compute::ITensor &tensor) override;

    private:
        proto_parse::Weights weights;
};
inline std::unique_ptr<arm_compute::graph::ITensorAccessor> get_output_accessor(std::vector<int> inshape){
    return arm_compute::support::cpp14::make_unique<OutputAccessor>(inshape);
}
inline std::unique_ptr<arm_compute::graph::ITensorAccessor> get_caffe_accessor(proto_parse::Weights weights){
    return arm_compute::support::cpp14::make_unique<CaffeWeightsAccessor>(weights);
}
inline std::unique_ptr<arm_compute::graph::ITensorAccessor> get_input_accessor(std::vector<int> shape){
    return arm_compute::support::cpp14::make_unique<InputAccessor>(shape);
}

class cpuEngine {
    public:
        cpuEngine(std::list<proto_parse::layerInfoBase*> ilayerslist, int id, std::string name);
        void runInference(std::vector<float>& input);
        std::list<proto_parse::layerInfoBase*> layersList;
    private:
        void createNetwork();
        arm_compute::graph::frontend::Stream graph;
        arm_compute::graph::frontend::ILayer* toCLLayer(proto_parse::layerInfoBase* layer);

};