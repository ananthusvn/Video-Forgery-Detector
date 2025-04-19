# 🎥 Video Forgery Detector

A Python-based tool for detecting signs of tampering, re-encoding, and double compression in video files. This project uses frame-level analysis, structural similarity (SSIM), metadata inspection, and DCT-based artifact detection to assess potential video forgery.

---

## 🔍 Features

- ✅ Metadata inspection (creation vs modification time)
- ✅ Frame-to-frame difference analysis
- ✅ Structural Similarity Index (SSIM) scoring
- ✅ Double compression detection using DCT
- ✅ Visual summary of analysis results
- ✅ Final verdict with confidence level

---

## 🛠 How It Works

The tool performs multiple layers of analysis:

1. **Metadata Analysis**  
   Checks for inconsistencies between file creation and modification times.

2. **Frame Difference Analysis**  
   Compares brightness differences and structural similarity between sampled frames.

3. **Double Compression Detection**  
   Uses DCT (Discrete Cosine Transform) on the Y-channel to detect quantization artifacts.

4. **Visualizations**  
   Generates plots for differences, SSIM scores, metadata summary, and result overview.

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.x
- OpenCV
- NumPy
- scikit-image
- Matplotlib

### 📦 Installation

```bash
pip install -r requirements.txt
