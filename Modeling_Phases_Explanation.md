# MODELING PHASES - Detailed Explanation
## Baby Cry Classifier - HuBERT Fine-tuning

---

### **Phase 1: Dataset & Model Architecture Setup (Cells 37-64)**

#### **1. PyTorch Dataset Class (Cell 38, 52)**
**Purpose**: Wraps data for PyTorch training.

**Key Components**:
- `BabyCryDataset`: Loads audio, handles augmented samples, applies preprocessing
- Feature extraction: Uses `Wav2Vec2FeatureExtractor` to convert raw audio to HuBERT-compatible features
- Handles both augmented (in-memory) and original (file path) samples

**Workflow**:
```
Audio File/Array → Preprocessing (3s, 16kHz) → Feature Extraction 
→ HuBERT Input Format → Label Encoding → Tensor Output
```

#### **2. DataLoader Creation (Cell 56)**
- Batch size: 8
- Shuffling: Enabled for training, disabled for validation/test
- Parallel loading: `num_workers=2`, `pin_memory=True` for GPU transfer

#### **3. HuBERT Model Loading (Cell 60)**
- Model: `facebook/hubert-base-ls960` (94.37M parameters)
- Pretrained on LibriSpeech 960h
- Output: 768-dimensional hidden states per timestep

#### **4. Baby Cry Classifier Architecture (Cell 62, 66)**
**Two versions used**:

**Phase 1 (Simple Classifier - Cell 62)**:
```
HuBERT Output (768-dim) → Linear(768→256) → ReLU → Dropout(0.3)
→ Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→7)
```

**Phase 2 (Improved Classifier - Cell 66, 74)**:
```
HuBERT Output (768-dim) 
→ Linear(768→512) → LayerNorm → GELU → Dropout(0.45)
→ Linear(512→256) → LayerNorm → GELU → Dropout(0.45)
→ Linear(256→128) → LayerNorm → GELU → Dropout(0.36)
→ Linear(128→num_classes)
```

**Architecture details**:
- Mean pooling over timesteps (batch_size × sequence_length × 768 → batch_size × 768)
- LayerNorm + GELU for stability
- Dropout (0.45) for regularization

---

### **Phase 1 Training: Frozen HuBERT (Cells 65-70)**

#### **Training Strategy**:
- HuBERT frozen (94.37M frozen, 0.23M trainable)
- Only the classifier head trains
- Goal: Learn mapping from pretrained features to classes

#### **Training Configuration (Cell 66)**:

**Loss Function**:
- `CrossEntropyLoss` with label smoothing (0.05)
- No class weights (balanced classes)

**Optimizer**:
- `AdamW` with learning rate `1e-4`
- Weight decay: `0.01`
- Beta values: `(0.9, 0.999)`

**Learning Rate Scheduler**:
- Linear warmup (10% of total steps) → linear decay
- Formula: `lr = base_lr * min(1, current_step / warmup_steps) * (1 - current_step / total_steps)`

**Training Techniques**:
- Mixed precision (`autocast` + `GradScaler`)
- Gradient clipping: `max_norm=1.0`
- Early stopping: patience 15 epochs

#### **Training Loop (Cell 68, 70)**:

**Training Function** (`train_epoch`):
1. Set model to training mode
2. Process batches with mixed precision
3. Compute loss, backward pass, clip gradients
4. Update weights, update learning rate
5. Track loss, accuracy, learning rate

**Validation Function** (`validate_epoch`):
1. Set model to eval mode, `no_grad()`
2. Forward pass only
3. Compute validation loss and accuracy
4. No weight updates

**Training Metrics Tracked**:
- Train/validation loss and accuracy per epoch
- Learning rate schedule
- Overfitting gap (train acc - val acc)
- Best model checkpoint saved

**Phase 1 Results**:
- Best validation accuracy: ~63.71%
- 7-class classification baseline established

---

### **Phase 2: Removing "Hungry" Class & Fine-tuning (Cells 71-77)**

#### **Diagnostic Step (Cell 73)**:
- Reason: "hungry" confused with other classes
- Action: Remove hungry, re-encode labels (6 classes)

**New Dataset**:
- Classes: `belly_pain, burping, cold_hot, discomfort, scared, tired`
- Train: 3,151 samples (balanced)
- Validation: 676 samples
- Test: 675 samples

#### **Model Rebuilding (Cell 74)**:

**Step 1: Architecture Update**:
- New classifier with 6 outputs (updated to 512→256→128→6)
- Dropout: 0.45

**Step 2: Transfer Learning**:
- Load Phase 1 HuBERT weights (skip classifier)
- Classifier reinitialized

**Step 3: Gradual Unfreezing Strategy**:
```python
Frozen: HuBERT layers 0-5 (early feature extraction)
Unfrozen: HuBERT layers 6-11 + Feature Projection
```

**Parameter Statistics**:
- Total: 94.93M parameters
- Trainable: 43.48M (45.8%)
- Frozen: 51.45M (54.2%)

**Rationale**: Early layers retain general audio features; later layers adapt to baby cry patterns.

#### **Optimizer Setup (Cell 75)**:

**Layer-wise Learning Rate Decay (LLRD)**:
```
Feature Projection: 2.40e-05
HuBERT Layer 11:    2.00e-05 (highest - closest to output)
HuBERT Layer 10:    1.90e-05
HuBERT Layer 9:     1.81e-05
...
HuBERT Layer 6:     1.55e-05
HuBERT Layer 1:     1.20e-05
Classifier:         5.00e-05 (highest)
```

**Scheduler**: 
- `CosineAnnealingWarmRestarts`
- T_0=5, T_mult=2, eta_min=1e-7
- Cosine decay with periodic warm restarts

#### **Training Loop (Cell 76)**:

**Enhanced Training Function** (`train_epoch_ft`):
- Tracks gradient norms
- Per-class accuracy during validation
- Tighter gradient clipping (`max_norm=0.5`)

**Metrics Tracked**:
- Train/val/test accuracy per epoch
- Per-class accuracy (every 5 epochs)
- Overfitting gap
- Gradient norms
- Early stopping based on test accuracy (patience: 7)

**Phase 2 Stage 1 Results** (Epochs 1-30):
- Test accuracy: 73.63% (+9.92% vs 7-class baseline)

#### **Extended Training (Cell 77)**:

**Stage 2: Additional 32 Epochs**:
- Same hyperparameters, continued from best Stage 1 checkpoint
- Extended patience: 10 epochs
- Total training: 62 epochs

**Final Results**:
- Best test accuracy: **82.37%** (at epoch 52)
- Improvement: **+18.66%** vs 7-class baseline
- Overfitting gap: 6.76% average

---

## **Key Design Decisions & Rationale**

### **1. Why Two-Phase Training?**
- Phase 1: Learn task-specific mapping while preserving pretrained features
- Phase 2: Fine-tune later layers for domain-specific patterns

### **2. Why Remove "Hungry"?**
- Confusion: acoustic similarity to other cries
- Solution: Handle hunger via questionnaire (context-based)

### **3. Why Gradual Unfreezing?**
- Prevent catastrophic forgetting
- Preserve low-level features, adapt high-level representations

### **4. Why Layer-wise LR Decay?**
- Lower LRs for early layers (stable features)
- Higher LRs for later layers (need more adaptation)

### **5. Architecture Improvements (Phase 2)**:
- LayerNorm: Stabilizes activations
- GELU: Better than ReLU for transformers
- Deeper classifier: 3 layers for more expressiveness
- Higher dropout (0.45): Better generalization

---

## **Training Progression Summary**

```
Phase 1 (7 classes, HuBERT frozen):
  63.71% → Baseline established
  ↓
Phase 2 Stage 1 (6 classes, unfreezing layers 6-11):
  73.63% → +9.92% improvement
  ↓
Phase 2 Stage 2 (continued training):
  82.37% → +8.74% additional improvement
  ↓
Total Improvement: +18.66% over baseline
```

This demonstrates a progression from a frozen-feature baseline to fine-tuning with class removal and improved architecture, achieving **82.37% test accuracy** on 6 cry classes.

---

## **Model Architecture Diagram**

### Phase 1 Architecture:
```
Audio Input (48,000 samples @ 16kHz)
    ↓
Wav2Vec2FeatureExtractor
    ↓
HuBERT Base (FROZEN - 94.37M params)
    ↓
Hidden States (batch × seq_len × 768)
    ↓
Mean Pooling (batch × 768)
    ↓
Classifier Head (TRAINABLE - 0.23M params)
    ↓
Linear(768→256) → ReLU → Dropout(0.3)
    ↓
Linear(256→128) → ReLU → Dropout(0.3)
    ↓
Linear(128→7) → Output (7 classes)
```

### Phase 2 Architecture:
```
Audio Input (48,000 samples @ 16kHz)
    ↓
Wav2Vec2FeatureExtractor
    ↓
HuBERT Base (PARTIALLY FROZEN - 51.45M frozen, 43.48M trainable)
  ├─ Layers 0-5: FROZEN
  └─ Layers 6-11 + Feature Proj: TRAINABLE
    ↓
Hidden States (batch × seq_len × 768)
    ↓
Mean Pooling (batch × 768)
    ↓
Improved Classifier Head (TRAINABLE)
    ↓
Linear(768→512) → LayerNorm → GELU → Dropout(0.45)
    ↓
Linear(512→256) → LayerNorm → GELU → Dropout(0.45)
    ↓
Linear(256→128) → LayerNorm → GELU → Dropout(0.36)
    ↓
Linear(128→6) → Output (6 classes)
```

---

## **Training Configuration Comparison**

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| **Classes** | 7 | 6 |
| **HuBERT Status** | Frozen | Partially Unfrozen |
| **Trainable Params** | 0.23M (0.2%) | 43.48M (45.8%) |
| **Classifier** | 2-layer (256→128→7) | 3-layer (512→256→128→6) |
| **Activation** | ReLU | GELU |
| **Normalization** | None | LayerNorm |
| **Dropout** | 0.3 | 0.45 |
| **Learning Rate** | 1e-4 (uniform) | 1.2e-5 to 5e-5 (layer-wise) |
| **Scheduler** | Linear warmup+decay | Cosine annealing w/ restarts |
| **Gradient Clipping** | 1.0 | 0.5 |
| **Best Accuracy** | 63.71% | 82.37% |

---

## **Key Takeaways**

1. **Transfer Learning Success**: Starting with frozen HuBERT and gradually unfreezing enables effective adaptation to baby cry classification.

2. **Class Selection Matters**: Removing the confusing "hungry" class improved accuracy by 9.92% in Phase 2 Stage 1.

3. **Architecture Refinement**: Improved classifier (LayerNorm + GELU + deeper) contributed to additional 8.74% improvement.

4. **Training Strategy**: Layer-wise learning rate decay and gradual unfreezing prevent catastrophic forgetting while enabling domain adaptation.

5. **Final Performance**: Achieved 82.37% test accuracy with good generalization (6.76% train-val gap).

---

*Generated from analysis of HuBERT_Final_Work.ipynb*
