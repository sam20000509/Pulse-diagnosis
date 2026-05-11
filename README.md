# Edge AI-Assisted Medical Signal Analysis Platform
*(Pulse-diagnosis System)*

## 💡 Project Overview
This project is an edge-computing medical auxiliary system that combines Digital Signal Processing (DSP) and Large Language Models (LLM). It automatically reads and analyzes raw pulse wave data, quantifying subjective medical observations into objective physical metrics.

## 🛠️ Tech Stack
* **Signal Processing:** Python, SciPy, NumPy (FFT, Butterworth Filter)
* **AI & Agent:** LangChain, LangGraph, Ollama (Qwen2.5-14B), ChromaDB (RAG)
* **Data Visualization:** Streamlit, Plotly

## 🚀 Key Features
1. **Advanced Signal Filtering:** Implemented 50Hz Butterworth low-pass filters and detrending techniques to eliminate hardware noise.
2. **Feature Extraction (FFT):** Applied Fast Fourier Transform to extract core frequency-domain parameters.
3. **Multi-Agent Workflow:** Architected an AI workflow via LangGraph for logic verification and safety auditing, preventing AI hallucinations.
4. **Edge Deployment:** 100% offline edge computing using Ollama for maximum data privacy.
