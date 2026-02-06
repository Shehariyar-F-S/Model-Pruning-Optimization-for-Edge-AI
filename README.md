This repository contains the code, experiments, and slides for an Edge AI pruning study on MobileNetV2, plus a conceptual Industry 4.0 edge deployment design.

Project Overview
This project addresses two tasks:

Task 1 – Practical Model Optimization
Compress a pre-trained lightweight CNN (MobileNetV2) using pruning and evaluate its suitability for edge deployment on a CIFAR‑10 classification task.

Task 2 – Edge AI Concept
Design a simple Industry 4.0 architecture showing how a factory can benefit from deploying pruned/optimized models at the edge instead of relying solely on cloud inference.


Task 1: Practical Model Optimization
Baseline Setup
Model: MobileNetV2 pretrained on ImageNet, last layer replaced with a 10‑class linear head for CIFAR‑10.

Dataset: CIFAR‑10 (60k images, 32×32 RGB, 10 classes), resized to 64×64.

Training:

Device: CPU (to simulate a resource‑constrained edge device).

Optimizer: Adam, LR = 1e‑3.

Epochs: 15.

Baseline performance:

Test accuracy ≈ 89.8%.

Parameters ≈ 2.24M.

Checkpoint size ≈ 8.8 MB.

CPU latency ≈ 1.0 ms/image.

These baseline metrics serve as the reference for all pruning experiments.

Pruning Heuristics
Two pruning heuristics were implemented:

Unstructured global magnitude pruning (weight‑level)

Applies global L1‑based pruning across all Conv2d and Linear layers.

Removes individual weights with the smallest absolute value.

Produces sparse weight matrices but keeps tensor shapes unchanged.

Structured channel pruning (filter‑level)

Applies per‑layer L2‑norm pruning on Conv2d output channels.

Removes entire channels/filters (structured sparsity), reducing layer width.

Produces a smaller, more hardware‑friendly architecture but is more destructive.


Pruning Regimes: One‑Shot vs Iterative
For each heuristic, two regimes were used:

One‑shot pruning

Start from baseline.

Prune directly to target sparsity (30%, 50%, 70%).

Fine‑tune for 5 epochs.

Iterative pruning

Start from baseline.

Increase sparsity in steps (e.g., +10% per step).

Fine‑tune for a few epochs after each pruning step.

Stop once (approximate) target sparsity is reached.

This gives four combinations:
(unstructured/structured) × (one‑shot/iterative).

Metrics Logged
For every configuration the following were recorded:

Accuracy: test accuracy (%) on CIFAR‑10.

Model size:

Total parameters.

Nonzero parameters (effective size).

Computed sparsity = 1 − nonzero/total.

Inference time: average CPU latency in ms/image.

All results are stored in a df_results table and visualized as:

Accuracy vs sparsity.

Latency vs sparsity.

Nonzero parameters vs sparsity.


Key Experimental Findings
Unstructured one‑shot pruning
At ≈30–50% sparsity, accuracy increased to about 91.7%, surpassing the baseline.

Even at ≈69% sparsity, accuracy remained ≈91.4%.

CPU latency remained around 0.95–1.2 ms/image with no consistent speedup.

Interpretation: weight‑level sparsity is excellent for parameter compression and can act as regularization, but dense CPU kernels do not fully exploit zeros for speed.


Structured one‑shot pruning
At ≈29% sparsity, accuracy dropped to ~83.7%.

At ≈49% sparsity, accuracy fell to ~66.8%.

At ≈69% sparsity, accuracy degraded to ~54.8%.

Latency fluctuated around 1.0–1.25 ms/image, with only mild improvement at high sparsity.

Interpretation: removing whole channels is much harsher on performance; accuracy collapses faster, and modest CPU latency gains do not compensate for the loss.


Unstructured iterative pruning
For moderate targets, the iterative schedule often overshot the desired sparsity:

Target 0.5 → actual sparsity ≈83.6%, accuracy ≈88.7%.

At extreme sparsity (~96.7%), accuracy collapsed to ~12.4% (near random), while latency remained ≈1.27 ms/image.

Interpretation: without careful schedule design, iterative pruning can over‑compress and destroy performance while still not accelerating dense CPU inference.


Structured iterative pruning
Structured iterative pruning was gentler than structured one‑shot at comparable sparsity:

At ≈26.6% sparsity, accuracy ≈85.6%.

At ≈40.1% sparsity, accuracy ≈79.8%.

At ≈51.1% sparsity, accuracy ≈73.0%.

Latency stayed around 1.17–1.31 ms/image; no consistent speedup.

Interpretation: gradual channel pruning with intermediate fine‑tuning mitigates some damage compared to one‑shot, but still hurts accuracy more than unstructured pruning.


Overall Conclusion for Task 1
Best trade‑off region: unstructured one‑shot pruning at ~30–50% sparsity.

~2× reduction in parameter count.

Accuracy improved slightly over baseline.

Latency largely unchanged on CPU.

Structured pruning showed the expected pattern: more hardware‑friendly shape changes but much stronger accuracy degradation at the same sparsity, and only modest latency improvement in this software/hardware environment.

Iterative pruning needs careful schedule tuning to avoid overshooting sparsity; in this project it demonstrated the failure modes of over‑compression.







Task 2: Edge AI Concept for Industry 4.0
Scenario
An Industry 4.0 visual quality inspection station on a production line:

Products move on a conveyor.

A camera captures images of each item.

An edge device runs the compressed CNN to classify items as “OK” or “defective” in real time.

Only summary data and periodic samples are sent to the cloud.

System Architecture Sketch
Hardware components

Industrial RGB camera mounted above the conveyor, capturing high‑frequency images.

Edge compute device near the line, e.g.:

ARM‑based industrial PC, NVIDIA Jetson, or Raspberry Pi‑class board.

PLC / actuator to physically reject defective items.

Factory network (Ethernet/Wi‑Fi) connecting edge devices to a central server or cloud.

Software stack

On the edge device:

Lightweight Linux OS.

Inference runtime: PyTorch/ONNX Runtime/TensorRT‑style optimized engine.

AI model: pruned MobileNetV2 (e.g., one‑shot unstructured at 50% sparsity).

Application logic:

Acquire image from camera.

Preprocess and run inference.

Send decision (OK/defect) to PLC.

Log predictions, confidence scores, and occasional images.

In the cloud / central server:

Long‑term storage and analytics dashboards (defect rates, distributions).

Offline training pipeline to update and re‑compress models regularly.

Deployment service to push new pruned/quantized models to edge devices.

Communication paths

Camera → Edge: high‑bandwidth local image stream.

Edge → PLC: low‑latency command signals (e.g., reject product).

Edge ↔ Cloud:

Upstream: compressed logs and sample images.

Downstream: updated model weights and configuration.

This architecture demonstrates to a potential customer how pruning enables models to run locally on inexpensive hardware, reducing dependency on cloud connectivity and latency.

Optimization Strategies and Their Impact
Two main optimization strategies are considered:

Pruning (implemented in Task 1)

Quantization (conceptual extension)

Pruning

Energy consumption

Fewer parameters and sometimes fewer FLOPs lead to less memory access and computation, which generally reduces energy per inference on edge devices.

Latency

Unstructured pruning: parameter count drops significantly, but standard dense CPU kernels do not exploit sparsity well, so latency remains roughly the same.

Structured pruning: by shrinking channel dimensions, FLOPs are reduced and specialized kernels could accelerate inference; in this project, latency improvements on CPU were modest.

Accuracy

Moderate pruning (30–50% sparsity) improved generalization for the CIFAR‑10 task.

Aggressive pruning (e.g., high structured sparsity, or ~97% unstructured sparsity) caused severe accuracy degradation.

Quantization

Energy consumption

Moving from FP32 to INT8 weights/activations can significantly reduce energy usage due to cheaper integer arithmetic and reduced memory bandwidth.

Latency

INT8 inference can deliver 2–4× speedup on hardware with optimized integer units or NPUs.

Accuracy

With careful quantization‑aware training, accuracy degradation can be kept within a small margin; naive post‑training quantization can cause larger drops, especially for small or sensitive models.

Combined strategy

A realistic edge pipeline would apply pruning first (e.g., 50% unstructured) to shrink the model, and then quantization to maximize speed and energy savings on edge hardware.

Risks of Over‑Compression
Two key risks when compressing models too aggressively:

Severe accuracy loss and reliability breakdown

Empirical example: unstructured iterative pruning that drove sparsity to ≈96.7% led to accuracy ≈12.4%, essentially random guessing.

In a factory, this would mean:

False negatives: defects not detected, leading to quality issues.

False positives: good parts wrongly rejected, causing waste and cost.

Mitigation:

Always validate compressed models on a held‑out test set.

Enforce minimum accuracy thresholds (e.g., ≥90% for safety‑critical tasks).

Stop pruning before accuracy falls below acceptable limits.

Loss of robustness and vulnerability to domain shifts

Heavily compressed models can become brittle and fail when conditions change (lighting, camera position, new defect types, noise).

In edge settings, these shifts are common and may not be immediately visible.

Mitigation:

Regularly collect new production data and retrain/re‑compress models.

Use ensembles or fallback models when confidence is low.

Implement confidence‑based rejection and route uncertain cases to human inspection.

Additional practical risk:

Hardware mismatch

Unstructured sparsity may not yield latency or energy improvements unless the hardware and libraries are sparsity‑aware.

Over‑pruning under such conditions sacrifices accuracy for theoretical sparsity with little real‑world benefit.

Recommended Edge Configuration
For this project’s constraints (CPU‑only simulation and CIFAR‑10 task):

Model choice:

MobileNetV2 with one‑shot unstructured pruning at ~50% sparsity:

Higher accuracy than baseline.

~2× fewer parameters and reduced model size.

Similar latency on CPU.

Next step (conceptual):

Apply INT8 quantization to the pruned model for deployment on actual edge hardware with integer accelerators.

This configuration balances accuracy, model size, and practicality, and serves as a solid demonstration of model compression for Industry 4.0 edge AI
