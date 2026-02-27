# 🚀 AI-Driven Luxury Marketing Content Generator

An advanced analytical tool designed to synthesize sophisticated marketing copy using the **Mistral-7B-Instruct** Large Language Model (LLM). This project demonstrates the integration of state-of-the-art Natural Language Processing (NLP) techniques with a functional web interface.

## 🔑 Key Functionalities
* **LLM Integration**: Leverages Mistral-7B with **4-bit Quantization** (via BitsAndBytes) to optimize performance on consumer-grade GPU environments.
* **Intelligent Prompt Engineering**: Implements structured system-level constraints to ensure the output maintains a high-end, professional persona.
* **Automated Fact-Checking**: A built-in validation layer that cross-references generated prices and discounts to ensure data integrity and prevent information leakage.
* **Sentiment & Tone Analysis**: Utilizes a secondary **DistilBERT** pipeline to provide real-time feedback on the emotional resonance of the generated content.
* **Responsive GUI**: A streamlined web interface developed using **Flask** and modern CSS for seamless user interaction.

## 🛠️ Technical Specifications
* **Core Language**: Python 3.x
* **Framework**: Flask (Backend) / HTML5 & CSS3 (Frontend)
* **Model Architectures**: 
  * Mistral-7B-Instruct-v0.2 (Causal Language Modeling)
  * DistilBERT-base (Sequence Classification)
* **Hardware Acceleration**: Optimized for NVIDIA T4 GPU via CUDA and Hugging Face `accelerate`.

## 📂 Project Structure
* `app.py`: The central engine containing model initialization, inference logic, and API routing.
* `templates/index.html`: The presentation layer defining the user interface.
* `requirements.txt`: Comprehensive list of dependencies for environment replication.
* `README.md`: Formal project documentation.

## 🚀 Deployment Instructions (Google Colab)
1.  **Environment Setup**: Ensure the Runtime type is set to **T4 GPU**.
2.  **Asset Upload**: Upload `app.py` and the `templates/` directory to the root path.
3.  **Dependency Installation**: Execute `!pip install flask-ngrok transformers accelerate bitsandbytes`.
4.  **Execution**: Run `app.py`. Access the application via the securely tunneled URL provided by Ngrok.
