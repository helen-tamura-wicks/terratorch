# Serving TerraTorch models with vLLM

TerraTorch models can be served using the
[vLLM](https://github.com/vllm-project/vllm) serving engine. Currently, only
models using the `SemanticSegmentationTask` or `PixelwiseRegressionTask` tasks
can be served with vLLM.

TerraTorch models can be served with vLLM in _tensor-to-tensor_ or
_image-to-image_ mode. The tensor-to-tensor mode is the default mode and is
natively enabled by vLLM. For the image-to-image mode instead, TerraTorch uses a
feature in vLLM called
[IOProcessor plugins](https://docs.vllm.ai/en/v0.13.0/design/io_processor_plugins/#writing-an-io-processor-plugin),
enabling processing and generation of data in any modality (e.g., geoTiff). In
TerraTorch we provide pre-defined IOProcessor plugins, check the list
[here](./vllm_io_plugins.md#available-terratorch-ioprocessor-plugins).

To enable your model to be served via vLLM, follow the below steps:

1. Verify the model you want to serve is either already a core model, or learn
   how to [add your model to TerraTorch](../models.md#adding-a-new-model).
2. [Prepare your model for serving with vLLM](./prepare_your_model.md).
3. If serving in image-to-image mode
   [Learn about IOProcessor plugins](./vllm_io_plugins.md), identify an existing
   one suiting your model or
   [build one yourself](https://docs.vllm.ai/en/latest/design/io_processor_plugins/).
4. Ensure your model weights and config.json are either hosted on Hugging Face,
   or stored in a local directory and accessible by your vLLM instance.
5. Start a vLLM serving instance that loads your model and perform an inference
   in [tensor-to-tensor mode](./serving_a_model_tensor.md) or in
   [image-to-image mode](./serving_a_model_image.md).
