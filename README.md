# Edge AI Medical Signal Analysis Platform

A Python-based Edge AI system for medical pulse waveform analysis using digital signal processing (DSP) and real-time visualization.

---

# Project Overview

This project focuses on transforming raw pulse waveform signals into quantifiable digital features through signal processing and AI-assisted workflows.

The system integrates:
- FFT spectrum analysis
- Noise filtering
- Real-time visualization
- Automated report generation
- Local Edge AI deployment

The platform is designed for offline execution to ensure privacy and low-latency processing.

---

# Features

- FFT-based frequency spectrum analysis
- Butterworth low-pass filtering
- Signal detrending and noise reduction
- HRV and spectral feature extraction
- Real-time waveform visualization
- Streamlit interactive dashboard
- Local LLM-assisted workflow
- Edge deployment using Ollama

---

# System Architecture

```text
Raw Pulse Signal
        ↓
Preprocessing
(Filter / Detrend)
        ↓
FFT Analysis
        ↓
Feature Extraction
        ↓
AI Workflow
        ↓
Visualization Dashboard


