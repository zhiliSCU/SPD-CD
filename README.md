# SPD-CD: A Single-Stage Physics-Driven Framework for Cross-Resolution Change Detection

This repository provides a conceptual implementation of:

**A Single-Stage Physics-Driven Framework for Cross-Resolution Change Detection in Unregistered Heterogeneous Remote Sensing Images**

---

## Overview

Cross-resolution change detection in heterogeneous remote sensing imagery suffers from a **triple coupling problem**:

1. Radiometric inconsistency induced by terrain and atmosphere
2. Geometric misalignment due to unregistered acquisition
3. Resolution discrepancy between sensors

Traditional registration-then-detection pipelines often amplify errors under these conditions.

To address this issue, we propose a **Single Physics-Driven Cross-Resolution Change Detection (SPD-CD)** framework.

---

## Core Components

### 1️⃣ PRMC — Physics-Driven Radiometric Mapping Compensation

- Inspired by the radiative transfer equation
- Models terrain-induced radiometric distortion
- Decouples high-frequency curvature components
- Applies curvature sparsity regularization
- Restores cross-temporal radiometric consistency

---

### 2️⃣ CRIFA — Cross-Resolution Implicit Feature Alignment

- Avoids explicit geometric registration
- Constructs multi-scale local feature correlation matrices
- Aligns heterogeneous-resolution features in feature space
- Resolves the registration–detection conflict

---

### 3️⃣ Differential Change Decoder

- Operates on aligned feature representations
- Produces final pixel-level change maps
- Designed for single-stage inference

---

## Important Notice

SPD-CD is an ongoing research project. 
The current release focuses on the core methodological design.

A complete reproducible version, including full training and 
evaluation protocols, will be released after the formal publication process.

This code is provided for research reference only.

---

