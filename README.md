This is a mostly self-contained PyTorch implementation of Mistral 7B in fp16.
The top-level `Mistral` assumes that the original bfloat16 parameters have
been converted, with an optional upscaling factor, into fp16.

The example script at `examples/hello_finetune.py` is a working toy example that:

1.  does the bf16-to-fp16 conversion with a parameter upscaling factor of 16;
    followed by
2.  a "hello world" finetuning run with a loss or gradient scaling factor of 1024,
    where the forward + backward pass is computed on the GPU, and where the gradient
    update is applied to the fp32 parameter master copy on the host CPU.
