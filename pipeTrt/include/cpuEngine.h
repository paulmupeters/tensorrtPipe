#include "parsePrototxt.h"
#include "arm_compute/graph/Graph.h"


class cpuEngine : {
    void createNetwork();
    arm_compute::graph::frontend::Stream graph;
    arm_compute::graph::frontend::ILayer toCLLayer(proto_parse::layerInfoBase* layer);
    arm_compute::Tensor input;
    public:
        cpuEngine::cpuEngine();
        void runInference();
};