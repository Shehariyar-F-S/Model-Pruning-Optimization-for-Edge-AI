# Edge AI Pruning Study on MobileNetV2

This repository contains the code, experiments, and slides for an Edge AI pruning study on MobileNetV2, plus a conceptual Industry 4.0 edge deployment design.

---

## Project Overview

This project addresses two main tasks:

* **Task 1 – Practical Model Optimization:** Compress a pre-trained lightweight CNN (MobileNetV2) using pruning and evaluate its suitability for edge deployment on a CIFAR‑10 classification task.
* **Task 2 – Edge AI Concept:** Design a simple Industry 4.0 architecture showing how a factory can benefit from deploying pruned/optimized models at the edge instead of relying solely on cloud inference.

---

## Task 1: Practical Model Optimization

### Baseline Setup

These baseline metrics serve as the reference for all pruning experiments.

* **Model:** MobileNetV2 pretrained on ImageNet, last layer replaced with a 10‑class linear head for CIFAR‑10.
* **Dataset:** CIFAR‑10 (60k images, 32×32 RGB, 10 classes), resized to 64×64.
* **Training Device:** CPU (to simulate a resource‑constrained edge device).
* **Optimizer:** Adam, LR = 1e‑3 (15 Epochs).

**Baseline Performance Metrics**

| Metric | Baseline Value |
| :--- | :--- |
| **Test Accuracy** | ≈ 89.8% |
| **Parameters** | ≈ 2.24M |
| **Checkpoint Size** | ≈ 8.8 MB |
| **CPU Latency** | ≈ 1.0 ms/image |

### Pruning Heuristics



Two pruning heuristics were implemented:

* **Unstructured global magnitude pruning (weight‑level):** Applies global L1‑based pruning across all Conv2d and Linear layers. Removes individual weights with the smallest absolute value. Produces sparse weight matrices but keeps tensor shapes unchanged.
* **Structured channel pruning (filter‑level):** Applies per‑layer L2‑norm pruning on Conv2d output channels. Removes entire channels/filters (structured sparsity), reducing layer width. Produces a smaller, more hardware‑friendly architecture but is more destructive.

### Pruning Regimes: One‑Shot vs Iterative

For each heuristic, two regimes were used (creating four combinations in total):

* **One‑shot pruning:** Start from baseline, prune directly to target sparsity (30%, 50%, 70%), and fine‑tune for 5 epochs.
* **Iterative pruning:** Start from baseline, increase sparsity in steps (e.g., +10% per step), fine‑tune for a few epochs after each step, and stop once the target sparsity is reached.

### Metrics Logged

All results are stored in a `df_results` table and visualized mapping sparsity against the following:

* **Accuracy:** Test accuracy (%) on CIFAR‑10.
* **Model Size:** Total parameters, nonzero parameters (effective size), and computed sparsity (1 − nonzero/total).
* **Inference Time:** Average CPU latency in ms/image.

### Key Experimental Findings

* **Unstructured One‑Shot:** At ≈30–50% sparsity, accuracy increased to ~91.7% (surpassing baseline) and remained ~91.4% even at 69% sparsity. CPU latency remained around 0.95–1.2 ms/image. **Interpretation:** Weight‑level sparsity acts as excellent regularization, but dense CPU kernels do not fully exploit zeros for speed.
* **Structured One‑Shot:** Accuracy dropped significantly as sparsity increased (~83.7% at 29% sparsity, down to ~54.8% at 69%). Latency fluctuated with only mild improvements. **Interpretation:** Removing whole channels is much harsher on performance; accuracy collapses faster without compensating CPU latency gains.
* **Unstructured Iterative:** Often overshot desired sparsity for moderate targets (e.g., Target 0.5 → actual ≈83.6%, accuracy ≈88.7%). Extreme sparsity (~96.7%) caused accuracy to collapse to near-random (~12.4%). **Interpretation:** Iterative pruning can over‑compress and destroy performance without careful schedule design.
* **Structured Iterative:** Gentler than structured one‑shot at comparable sparsities (e.g., ~85.6% accuracy at 26.6% sparsity), but latency saw no consistent speedup. **Interpretation:** Gradual channel pruning mitigates some one-shot damage, but still hurts accuracy more than unstructured pruning.

### Task 1 Conclusion

The **best trade‑off region** is unstructured one‑shot pruning at ~30–50% sparsity. This yields a ~2× reduction in parameter count with slightly improved accuracy, though latency remains largely unchanged on a CPU. Iterative schedules require careful tuning to avoid over-compression, and structured pruning demands specialized hardware to realize latency benefits without severe accuracy loss.

---

## Task 2: Edge AI Concept for Industry 4.0

### Scenario

An Industry 4.0 visual quality inspection station on a production line where items move on a conveyor. A camera captures images, and an edge device runs a compressed CNN to classify items as “OK” or “defective” in real-time. Only summary data and periodic samples are sent to the cloud.

### System Architecture Sketch



* **Hardware Components:** Industrial RGB camera, edge compute device (ARM‑based PC, NVIDIA Jetson, or Raspberry Pi), PLC/actuator for physical rejection, and factory network (Ethernet/Wi‑Fi).
* **Edge Software Stack:** Lightweight Linux OS, inference runtime (PyTorch/ONNX Runtime/TensorRT), pruned AI model, and local application logic (acquire, preprocess, infer, command PLC, log).
* **Cloud Software Stack:** Long‑term storage, analytics dashboards, offline training/compression pipeline, and deployment services for pushing model updates.
* **Communication Paths:** High‑bandwidth local image stream (Camera → Edge), low‑latency command signals (Edge → PLC), compressed logs/samples (Edge → Cloud), and updated model weights (Cloud → Edge).

### Optimization Strategies and Their Impact

| Strategy | Energy Consumption | Latency | Accuracy |
| :--- | :--- | :--- | :--- |
| **Pruning** | Reduced via fewer parameters/FLOPs (less memory access/computation). | CPU mostly unchanged (unstructured) or modestly improved (structured). | Improved at moderate levels (30-50%); severe degradation if aggressive. |
| **Quantization** (Conceptual) | Significantly reduced due to cheaper integer arithmetic (INT8) and lower bandwidth. | 2–4× speedup on hardware with optimized integer units or NPUs. | Negligible drop with quantization-aware training; larger drop if naive. |

*Note: A realistic edge pipeline combines both: applying pruning first to shrink the model, then quantization to maximize speed/energy savings.*

### Risks of Over‑Compression

* **Severe accuracy loss and reliability breakdown:** Heavily pruned models (e.g., ~97% sparsity) devolve to random guessing, causing false negatives (missed defects) and false positives (waste). **Mitigation:** Always validate on a held-out test set, enforce minimum safety thresholds (e.g., ≥90%), and stop pruning early.
* **Loss of robustness to domain shifts:** Compressed models become brittle to lighting, noise, or new defect types. **Mitigation:** Regularly retrain on new production data, use model ensembles, and route low-confidence predictions to human inspectors.
* **Hardware mismatch:** Unstructured sparsity offers no real-world latency/energy benefit unless hardware and libraries are sparsity‑aware. 

### Recommended Edge Configuration

For this project’s constraints (CPU‑only simulation and CIFAR‑10 task):

* **Model Choice:** MobileNetV2 with one‑shot unstructured pruning at ~50% sparsity.
* **Benefits:** Higher accuracy than baseline, ~2× fewer parameters, and reduced overall size.
* **Next Conceptual Step:** Apply INT8 quantization to the pruned model for deployment on actual edge hardware with integer accelerators. 

This configuration successfully balances accuracy, model size, and practicality for Industry 4.0 deployments.
