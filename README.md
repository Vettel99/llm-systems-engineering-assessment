# LLM Systems Engineering Assessment

**Candidate:** Abhishek Bhadkamkar  
**Model Used:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## 1. Design Decisions: Parser Architecture
I designed the model architecture parser utilizing a recursive tree-building approach. Instead of merely printing the `nn.Module` layers sequentially, the `build_model_tree()` function traverses the PyTorch module graph and outputs a decoupled, nested dictionary. 
* **Trade-offs:** I prioritized structural readability and data modularity over immediate execution speed. A nested dictionary format allows the parsed hierarchical output to be easily serialized and passed into downstream JSON loggers, automated testing suites, or UI visualization tools in a production environment.
* **Reusability:** By decoupling the data extraction logic (`build_model_tree`) from the display formatting logic (`print_model_tree`), the parser can ingest new, complex architectures without rewriting the core extraction loops.

## 2. Working Conditions & System Debugging
* **Hardware:** Executed in a Google Colab environment utilizing a single NVIDIA T4 GPU (16GB VRAM) and standard system RAM.
* **Timeframes:** The architecture parsing executed in seconds. The LoRA fine-tuning (restricted to 50 steps using a 500-sample slice of IMDB) completed in under 5 minutes.
* **Environment Hurdles:** Google Colab recently updated its default backend to CUDA 12.8. This initially broke the standard `bitsandbytes` 4-bit quantization because older binaries defaulted to the CPU. I resolved this system-level dependency issue by explicitly unpinning and upgrading to `bitsandbytes>=0.45.3` in the `requirements.txt`. I also explicitly suppressed hanging Weights & Biases (W&B) dashboard prompts by passing `report_to="none"` to the `TrainingArguments` to ensure the pipeline executes seamlessly from start to finish.

## 3. Extensibility & Scalability
* **Scaling to 50 Models:** If scaling to 50 models, holding them in memory sequentially would instantly exhaust system RAM. I would implement an Abstract Factory pattern with lazy loaders. Models would be pulled from remote storage (e.g., AWS S3), processed, and immediately garbage-collected before the next model is loaded into memory.
* **Handling Differing Architectures:** Hardcoding target modules (like Llama's `q_proj`) breaks when switching to architectures like GPT-2 (which uses `c_attn`). I would implement an automated mapping dictionary that links `config.model_type` to their specific attention block names, allowing the system to dynamically inject LoRA adapters regardless of the underlying architecture.

## 4. Creativity & Future Vision
If tasked with building an automated model improvement system (parse → evaluate → gap fix → merge), I would implement the following novel approaches:
1. **Automated SLERP Merging:** Instead of basic linear weight averaging, I would use Spherical Linear Interpolation (SLERP) to merge models. SLERP maintains the geometric properties and high-dimensional curvature of the model's weights much better than linear averaging, resulting in fewer "brain damage" artifacts during composition.
2. **Parser-Driven Target Selection:** The parser shouldn't just read the model; it should guide the training. I would script the parser to evaluate the variance of weights across different layers. Layers with the highest structural variance could be dynamically selected as the `target_modules` for PEFT, effectively automating the LoRA configuration rather than relying on manual guesses.
3. **Rejection Sampling on Merge (The Twist):** Before permanently committing a merge, the system should generate outputs from both the base model and the adapter. Using an LLM-as-a-judge (like GPT-4), the system would perform automated rejection sampling, only applying the merged weights to production if the semantic reasoning score mathematically outperforms the base model.

## 5. Honest Reflection
Because I have professional experience fine-tuning large language models using the Hugging Face ecosystem, setting up the `SFTTrainer` and formatting the datasets felt highly intuitive. 

The primary challenge was managing VRAM state during the composition phase (Task 3). You cannot merge LoRA adapters into a quantized (4-bit) model via `merge_and_unload()`. I had to delete the 4-bit model and reload the base model in standard `fp16`. However, simply calling `del model` causes a `NameError` if a user reruns the cell. I resolved this by writing an idempotent `try...except` block to safely clear the GPU cache. 

Ultimately, seeing the base model ramble incoherently ("The reviewer's sentiment about the acting and plot is negative...") versus the successfully merged fine-tuned model instantly adhering to the strict requested format ("The sentiment is Negative.") was highly rewarding and proved the pipeline's effectiveness.
