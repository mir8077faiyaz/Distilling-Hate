# Distilling Hate  
### Lightweight Transformer Models for Fine-Grained Bengali Hate Speech Detection

This repository contains the official implementation of **teacher-guided knowledge distillation** for efficient Bengali hate speech detection, as described in our paper:

> **Distilling Hate: Lightweight Transformer Models for Fine-Grained Bengali Hate Speech Detection**  

We propose a **multi-objective knowledge distillation framework** that transfers both output-level and representation-level knowledge from a large BanglaBERT teacher to a compact BanglaBERT student, enabling strong performance under severe class imbalance while drastically reducing computational cost.

---

## 🔍 Overview

Hate speech detection in Bengali social media is challenged by:
- severe class imbalance,
- fine-grained hate categories,
- high computational cost of large transformer models.

Our approach addresses these challenges by combining:
- **Focal-loss–trained teacher models**
- **Logit-level knowledge distillation (KL divergence)**
- **Hidden-state alignment via CLS projection**
- **Optuna-based supervision loss and hyperparameter selection**

The resulting student model contains **only 13.7M parameters** and achieves competitive or superior **macro-F1** compared to much larger transformer baselines.

---

## 🧠 Architecture


<p align="center">
  <img src="figures/archi.png" width="85%">
</p>

**Training procedure**
1. Train a high-capacity **BanglaBERT Large** teacher using focal loss.
2. Freeze the teacher model.
3. Train a lightweight **BanglaBERT Small** student using:
   - supervised loss (cross-entropy or focal loss),
   - logit-level knowledge distillation,
   - hidden-state alignment via projected CLS representations.

---

## 📊 Key Results

### Macro-F1 Improvement via Knowledge Distillation

<p align="center">
  <img src="figures/student_macro_f1_all_datasets.png" width="85%">
</p>

**Macro-F1 comparison of student models trained with cross-entropy (baseline) and with knowledge distillation across multiple datasets.**  
Knowledge distillation consistently improves or preserves student performance, with particularly large gains observed in challenging and fine-grained settings such as **BanTH multi-label classification**.

---

## 📚 Datasets

We evaluate our framework on three benchmark datasets:

- **BanglaMultiHate (BLP-2025)**
  - Subtask-1A: Hate Type Classification
  - Subtask-1B: Target Group Classification
- **DeepHateExplainer** (native Bengali script)
- **BanTH** (transliterated Bangla, multi-label classification)

---

