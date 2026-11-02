# Environment for the run scripts (models are read from local paths; Hugging Face stays offline).
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # allocator only; outputs unchanged
