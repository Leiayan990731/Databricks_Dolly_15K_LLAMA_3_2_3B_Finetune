# Summary Report
1. Model Architecture<br/>
The selected model, Llama 3.2 3B Instruct, is an auto-regressive transformer optimized for multilingual dialogue and reasoning.
- Parameter Efficiency: With 3.21 billion parameters, it occupies a "sweet spot" between the lightweight 1B model and the resource-intensive 8B model, outperforming many larger open-source models on benchmarks like MMLU.
- Grouped-Query Attention (GQA): Utilizes GQA to maintain high inference scalability and throughput.
- Quantization: The model was loaded using 4-bit quantization (via bitsandbytes), compressing weights into a 4-bit format (NF4) to run on consumer-grade GPUs (e.g., NVIDIA T4/L4) while retaining near-FP16 performance.
2. Training Process & Methodology<br/>
The training pipeline leveraged the Unsloth library for optimized backpropagation and the TRL SFTTrainer for supervised fine-tuning.
- Dataset Split: To prevent data leakage, the 15,000-record Dolly dataset was split beforetraining into 95% training and 5% testing partitions using a fixed seed (4016).
- LoRA Configuration: We injected trainable rank decomposition matrices () into key transformer modules (q_proj, k_proj, v_proj, o_proj, etc.), making only ~1% of parameters trainable while keeping the pre-trained weights frozen.
- Objective Function: The model was trained using "train on responses only", masking the user instructions in the loss calculation. This ensures the model learns to generate answers rather than just modeling the structure of the prompt.
- Hyperparameters:
- Epochs: 1 (approx. 700 steps) — increased from the baseline 60 steps to ensure convergence.
- Learning Rate: 2e-4 (Linear decay).
- Batch Size: 2 (with Gradient Accumulation = 4).
3. Experiments & Results<br/>
The primary experiment compared the baseline assumptions (1B model, low step count) against the modified approach (3B model, full epoch).
- Baseline (1B, 60 steps): Failed to generalize, producing repetitive or incoherent text with ROUGE-L scores well below 0.20.
- Modified (3B, 1 Epoch): Successfully achieved ROUGE-L > 0.205. The 3B model demonstrated superior handling of complex instructions and context retention compared to the 1B variant.<br/>
Evaluation Metric:
- ROUGE-L: Measures the longest common subsequence between the reference and generated text, acting as a proxy for sentence-level structure and fluency.
(Visualization Concept: A bar chart comparing ROUGE-1, ROUGE-2, and ROUGE-L scores between the Baseline 1B and Fine-Tuned 3B models)
4. Analysis & Evaluation Conclusions
- Technique Efficacy: LoRA proved highly effective, allowing a 3B model to be fine-tuned on a single GPU. 4-bit quantization enabled the use of a larger, more capable model (3B vs 1B) within the same memory constraints, directly contributing to the success of the ROUGE metric.
- Training Duration: The jump from 60 steps to 1 epoch was critical. Instruction following requires the model to see a diverse range of examples to generalize the "assistant" persona effectively; 60 steps was insufficient for this distributional shift.
- Model Size: The 3B model offers significantly better reasoning capabilities than the 1B model, which often hallucinates or loses coherence on open-ended QA tasks found in the Dolly dataset.
