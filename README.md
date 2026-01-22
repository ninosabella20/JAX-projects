# Projects – 2025/2026 Winter Semester's Work

This repository contains the official semester report and project collection of:

**Student:**  
Nino (Antonino Paolo) Sabella  

**Course:**  
*Machine Learning in the Life Sciences*  

**Course Focus:**  
Machine learning methods for differential equations using **JAX**, with applications to continuous-time dynamical systems.

This repository documents the practical work carried out during the semester and serves as the formal record of the implemented models and experiments.

---

## Course Context and Final Presentation

In parallel to this repository, a final presentation is given for the course.  
Although the presentation is a separate task, it provides important conceptual insight into the structure and coherence of the work in this repository.

The presentation addresses the following guiding questions:

- Why JAX?  
- Why Equinox? (What is a neural network in this context?)  
- Why Diffrax? (What is a differential equation and why model it?)  
- What projects were done over the semester? (ODE and SDE models)  
- A selected sample project:
  - Why this particular sample was selected  
  - Dataset from the sample  
  - General selected model  
  - Goal of the model  
  - Implementation strategy (architecture, dynamics, etc.)  
  - Results  
  - Other life-science problems this type of model could solve  

This overview reflects the learning trajectory of the course and explains how the individual projects fit into a coherent framework.

---

## Folder Structure

This repository is organized into three main topics:

1. Understanding JAX  
2. Neural Ordinary Differential Equations (NODEs)  
3. Neural Stochastic Differential Equations (NSDEs)  

Each folder contains code notebooks and short conceptual presentations.

---

### 1. Understanding JAX

**Purpose:**  
Introduction to JAX and its core functionality for numerical computing and automatic differentiation.

**Contents:**
- `Jax Introduction Code`  
  Introductory code examples demonstrating basic JAX concepts.

---

### 2. Neural Ordinary Differential Equations (NODEs)

**Purpose:**  
Implementation and study of Neural ODEs for modeling continuous-time dynamical systems.

**Contents:**
- `ConceptualPresentationNODE.pdf`  
  Conceptual overview of Neural ODEs.

- `SimplestNODE-Euler.ipynb`  
  Minimal Neural ODE example using the Euler method.

- `NODEv1-LH-Covid.ipynb`  
- `NODEv1-Spirals.ipynb`  
- `NODEv1-SpiralsMissingData.ipynb`  
- `NODEv1-Sepsis.ipynb`  

These notebooks explore Neural ODEs on different datasets.

---

### 3. Neural Stochastic Differential Equations (NSDEs)

**Purpose:**  
Extension of Neural ODEs to stochastic dynamics using Neural SDEs.

**Contents:**
- `ConceptualPresentationNSDE.pdf`  
  Conceptual overview of Neural SDEs.

- `NSDEv0-NODEv1.5-Covid.ipynb`
- `NSDEv1-NODEv2-Covid.ipynb` 
- `NSDEv2-NODEv3-Covid.ipynb`  
- `NSDEv1-NODEv2-LH.ipynb`  
- `NSDEv2-NODEv3-LH.ipynb`  
- `NSDEv1-NODEv2-Spirals.ipynb`  

---

## Naming Convention

- `v` stands for **version** (model trial number).  
- Higher version numbers correspond to **later and improved model iterations**.  
- In each sequence, the **last version is the most successful model**, reflecting incremental improvements over previous trials.

---

## Requirements

Most notebooks use:

- Python 3  
- JAX  
- NumPy  
- Matplotlib  

Install dependencies as needed before running the notebooks.

---

## Notes

- This repository documents the full semester workflow and learning progression.  
- Notebooks are named to reflect dataset, model type, and version.  
- Presentations provide the theoretical background for each section.  
- The repository and final presentation together represent the complete course outcome.
