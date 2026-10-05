# Transfer Learning with CNNs — teaching flow

## Guided live section: Fashion-MNIST (about 30 minutes)

### Goal
Move from the CNN already studied on Fashion-MNIST to the transfer-learning workflow using a pretrained ResNet-18.

### Suggested sequence
1. **Recall the previous CNN**: all convolutional filters were learned from random initialization.
2. **Inspect pretrained input requirements**: ImageNet RGB preprocessing, resizing, normalization, and the Fashion-MNIST domain mismatch.
3. **Load ResNet-18 and replace the head**: show the original 1000-class classifier and replace it with 10 outputs.
4. **Freeze the backbone**: compare total and trainable parameter counts; ask whether freezing removes forward-pass cost.
5. **Feature extraction**: train only the new head for a short run.
6. **Partial fine-tuning**: unfreeze `layer4`, use a smaller learning rate for pretrained layers and a larger one for the head.
7. **Exit discussion**: training from scratch vs feature extraction vs fine-tuning; when transfer should work well or poorly.

### Live questions
- Why must the ImageNet head be replaced?
- Why do pretrained layers normally use a smaller learning rate?
- What is saved by freezing parameters, and what computation remains?
- Why is ImageNet → Fashion-MNIST an intentionally imperfect transfer scenario?

---

## Independent CIFAR-10 activity (about 2 hours)

### Deliverable 1 — Student-designed CNN
Students design a CNN for `[3, 32, 32]` inputs, train it from scratch, and justify architecture, regularization, parameter count, and validation behavior.

### Deliverable 2 — Student-selected transfer-learning model
Students choose a TorchVision pretrained CNN, identify its weights/transforms/classification head, train it first as a frozen feature extractor, then partially fine-tune the last block. Comparison should use validation accuracy and computational considerations.

Suggested backbones: ResNet-18, EfficientNet-B0, MobileNetV3-Small. Other TorchVision classifiers are acceptable if adapted correctly.

### Deliverable 3 — Optuna study on the student CNN
Students tune a compact but meaningful search space that includes capacity, regularization, optimizer and learning-rate choices. The study uses a reduced train/validation subset, Optuna pruning, and per-trial early stopping.

Students must analyze:
- best trials, not only the winner;
- completed vs pruned vs early-stopped trials;
- parameter importance;
- at least two hyperparameter interactions or trends.

After tuning, students rebuild a fresh model with the best parameters, retrain it on the normal train/validation split, restore the best validation state, evaluate the test set once, and save/reload the checkpoint.

### Assessment emphasis
Give more weight to experimental reasoning than to absolute accuracy. Strong submissions should show:
- a valid architecture and data pipeline;
- a clear train/validation/test protocol;
- correct feature extraction and partial fine-tuning;
- a sensible Optuna search space;
- evidence-based interpretation of hyperparameter behavior;
- reproducible final-model saving.
