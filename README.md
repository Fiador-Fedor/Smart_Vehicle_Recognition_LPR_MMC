# Smart_Vehicle_Recognition_LPR_MMC 🚓  
License Plate Recognition (LPR) & Make, Model, Colour (MMC) System  

An intelligent **License Plate Recognition (LPR)** and **Make, Model & Colour (MMC)** classification system developed by **Prince Benneh** & **Kelvin Fedor**.  
The system leverages **computer vision, deep learning, and IoT integration** with a **Raspberry Pi + Web App** to detect, recognize, and manage vehicles in real-time.  

---

## ✨ Features  

- 📸 **License Plate Recognition (LPR)** – Detects and recognizes license plates from vehicle images.  
- 🚘 **Make & Model Classification** – Identifies vehicle manufacturer and model.  
- 🎨 **Colour Recognition** – Detects and classifies vehicle colour using CNN models.  
- 🍓 **Raspberry Pi Integration** – Pi runs the detection scripts and pushes results directly to Firebase.  
- 💻 **Web App (React + Wouter + Firebase)** – Fetches data from Firebase and displays real-time results.  
- ⚡️ **Real-Time Alerts** – Flags suspicious or blacklisted vehicles instantly.  

---

## 🏗 System Architecture  

### Data Sources  
- **Stanford Cars Dataset** → Make & Model recognition  
  🔗 https://ai.stanford.edu/~jkrause/cars/car_dataset.html  
- **VCoR Dataset** → Colour classification  
  🔗 https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset  
- **ANPR Datasets** → License plate recognition  
  - https://github.com/detectRecog/CCPD  
  - https://universe.roboflow.com/automatic-license-plate-recognition-kxxzn/license-plate-detection-ee9ca/dataset/7  
  - https://universe.roboflow.com/luther-eyandi/ghanaian-license-plates/dataset/1  
  - https://universe.roboflow.com/selorm/numberplatesdetect/dataset/2  
- **OCR Dataset** → OCR training  
  🔗 https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/data.zip  

### Deep Learning Models  
- **LPR:** YOLOv8 + Custom OCR  
- **Make & Model:** EfficientNet / ResNet  
- **Colour Detection:** CNN trained on VCoR  

### Pipeline  
1. Raspberry Pi captures vehicle image  
2. Runs detection + classification script  
3. Pushes result to Firebase  
4. Web App fetches and displays data in real-time  

---

## 📂 Project Structure  

```

LPR-MMC/
│── Application_Code/   # Raspberry Pi detection + Firebase push scripts, web dashboard
│── Models/             # Trained ML models (LPR, MMC)
│── Datasets/           # Dataset references
│── Docs/               # Documentation
│── README.md           # Project documentation

````

---

## 🚀 Getting Started  

### 🔧 Prerequisites  
- Raspberry Pi 4 (or higher)  
- Python 3.9+  
- Node.js 16+  
- Firebase project setup  

### 🖥 Setup Raspberry Pi Scripts  
```bash
cd Application_Code
pip install -r requirements.txt
python run_System_implementation_Scipt.py
````

This script runs detection and pushes results to Firebase.

### 🌐 Setup Web App

```bash
cd web_app
npm install
npm run dev
```

The web app fetches and displays data from Firebase in real time.

---

## 📊 Example Vehicle Scan

```json
{
  "id": "scan-001",
  "licensePlate": "GE4589-25",
  "make": "Toyota",
  "model": "Corolla 2019",
  "color": "White",
  "flagged": true,
  "status": "stolen",
  "scannedAt": "2025-09-30T14:32:00Z",
  "scannedBy": "raspberry-pi-01",
  "scanSource": "pi_camera"
}
```

---

## 🌐 Web App Features

* 📊 **Dashboard** – Displays recent scans
* 🔍 **Search & Filter** – Find vehicles by plate, make, or status
* 🚨 **Flagged Vehicles** – Highlight stolen or suspicious cars
* 🔒 **Firebase Authentication** – Secure access for admins

---

## 📌 Roadmap

* ✅ LPR + MMC pipeline on Pi
* ✅ Firebase integration
* ✅ React + Wouter web app
* ⏳ Add analytics dashboard
* ⏳ Deploy full production system

---

## 🤝 Contributors

👨🏽‍💻 **Kelvin Fiador** – Computer Vision (Plate Detector & OCR), Hardware
👨🏾‍💻 **Prince Benneh** – Computer Vision (Make/Model Detector & Color Detector) Web App, Firebase Integration
