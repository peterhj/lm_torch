This is a mostly self-contained PyTorch implementation of Mistral 7B.

The example script at `examples/hello_finetune.py` is a working toy example
that does the following:

1.  convert the original bfloat16 parameters, with an optional upscaling
    factor, into a working dtype (fp16 by default); followed by
2.  a "hello world" finetuning run with a loss or gradient scaling factor
    of 1024, where the forward + backward pass is computed on the GPU using
    the working dtype, and where the gradient update is applied to the fp32
    parameter master copy on the host CPU.
