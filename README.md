# tensorrtPipe
Inference optimization library for tensorrt. Splits up a tensorrt network into multiple engines and uses a pipeline to improve the throughput of the network.
## Tested on:
<pre>
NVIDIA Jetson Tx2
ubuntu 18.04
CUDA 10.0
TensorRT 6.0.1.
</pre>


## Should be used as followed:
<pre><code>
nvinfer1::INetworkDefinition trtNetwork;
multiStreamTrt(trtNetwork) pptrt;

pptrt.doInference(input);

</code></pre>

An example can be found in example/
