# 🚀 Asphalt Crack Detection and Classification
A deep learning-based **multi-task YOLOv11 model** for **asphalt distress detection and classification**.  
This project enables **automated identification and categorization of cracks** in road surfaces to aid in **infrastructure maintenance and inspection**.

## 📌 Project Overview
This project utilizes **YOLOv11** with a **custom multi-task architecture** that performs:
- **Object Detection:** Identifies and localizes cracks in asphalt.
- **Classification:** Categorizes detected cracks into different types (e.g., linear cracks, alligator cracks, transverse cracks, etc.).

The system is designed for **high-accuracy, real-time detection**, making it useful for **road maintenance automation, transportation engineering, and smart city initiatives**.

---

## 🏗️ Features
✅ **YOLOv11 Multi-Task Model** – Detects and classifies asphalt cracks in one forward pass.  
✅ **Custom Dataset Integration** – Supports annotated datasets with bounding boxes and crack categories.  
✅ **Modular & Scalable** – Flexible architecture for future enhancements.  
✅ **Optimized for Performance** – Trained on multiple YOLOv11 sizes (`n, s, m, l, x`) to find the best-performing model.  
✅ **Azure DevOps Integration** – Supports version control and model retraining workflows.

---

## 🔬 Model Modifications
This project **extends YOLOv11** by adding **a classification head** alongside the existing **detection head**.  

🔹 **New Multi-Task Architecture**
- **Backbone & Neck:** Uses YOLOv11’s default feature extractor.  
- **Detection Head:** Predicts bounding boxes and confidence scores.  
- **Classification Head:** Predicts crack type for each detected region.

🔹 **Custom Multi-Task Loss**
- Combines **YOLO object detection loss** with **classification loss**.
- Adjusted with hyperparameters **α (detection weight)** and **β (classification weight)**.

---

## 📊 Model Training
1️⃣ **Prepare the dataset** (`images + annotations + class labels`).  
2️⃣ **Modify YOLOv11 to support multi-task learning** (`model.py`, `trainer.py`, `dataset.py`).  
3️⃣ **Train YOLOv11 on multiple model sizes** (`n, s, m, l, x`) using:
   ```bash
   python train.py --data datasets/dataset.yaml --epochs 50 --imgsz 640

---

## 📈 Model Evaluation Metrics
We evaluate models using:

- mAP (mean Average Precision)
- Precision / Recall / F1-score
- Inference Speed (FPS)
- Model Size & Latency
