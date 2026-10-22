# HW3: Triton tasks

15 tasks from [@efficient_models_bot](https://t.me/efficient_models_bot), deadline 06.12.2026 00:00 MSK. One notebook per task, the links open it in Colab:

| Task id | Task |
|---|---|
| [`00_scalar_sum_two_numbers`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/00_scalar_sum_two_numbers.ipynb) | Scalar sum (warmup, dz0) |
| [`01_constant_add`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/01_constant_add.ipynb) | Constant add, one block |
| [`02_constant_add_block`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/02_constant_add_block.ipynb) | Constant add, blocks with a masked tail |
| [`03_outer_vector_add`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/03_outer_vector_add.ipynb) | Outer sum, one block |
| [`04_outer_vector_add_block`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/04_outer_vector_add_block.ipynb) | Outer sum, 2D grid |
| [`05_outer_mul_relu_block`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/05_outer_mul_relu_block.ipynb) | Outer product + ReLU |
| [`06_outer_mul_relu_block_backward`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/06_outer_mul_relu_block_backward.ipynb) | Its backward for x, `tl.atomic_add` |
| [`07_long_sum`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/07_long_sum.ipynb) | Row sum with a loop over tiles |
| [`08_long_softmax`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/08_long_softmax.ipynb) | Row softmax with a loop over tiles |
| [`09_flashattention_scalar`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/09_flashattention_scalar.ipynb) | Online-softmax attention, scalar q |
| [`10_quantize_global_int8`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/10_quantize_global_int8.ipynb) | Per-tensor int8 quantization |
| [`11_quantize_rowwise_int8`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/11_quantize_rowwise_int8.ipynb) | Per-row int8 quantization |
| [`12_matmul_int8_fused_dequant`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/12_matmul_int8_fused_dequant.ipynb) | int8 matmul with dequant and bias in the epilogue |
| [`13_quantize_global_transpose_int8`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/13_quantize_global_transpose_int8.ipynb) | Per-tensor int8 quantization written transposed |
| [`14_backward_dx_int8_fused`](https://colab.research.google.com/github/PE51K/itmo-ai-talent-hub-efficient-models-course/blob/main/hw3/14_backward_dx_int8_fused.ipynb) | dX = dY @ dequant(W)^T |

Each notebook has the task statement from the bot, the bot's signature in a cell that `%%writefile` saves to `<id>.py`, and a check against PyTorch on shapes with tails, plus some of the stated requirements (autotune config counts, `tl.atomic_add`, `exp2`). Submission starts with `/submit <id>` in the bot; the format it expects is described at the end of the seminar.

The bot queues submissions and a check can take an hour or more, so pass the notebook check before submitting. The checks are not the bot's tests.

Triton has no macOS build, so run the notebooks on a Colab T4: Runtime -> Change runtime type -> T4 GPU.
