Here's the revised README reflecting the **pretraining on Wikitext** and **fine-tuning on the Shakespeare dataset**:

---

# **EECS 595 Project: RetNet Study and Implementation**

## 📋 **Overview**  
This project, part of the **EECS 595 course**, explores the **RetNet** model architecture and implements models from scratch, inspired by **nanoGPT**. We performed **pretraining** on the **Wikitext-2** and **Wikitext-103** datasets, followed by **fine-tuning** on the **Shakespeare dataset** to adapt the model for literary text generation.

### 🔗 **Collaborators**
- [@keaganjp](https://github.com/keaganjp)  
- [@anmolmansingh](https://github.com/anmolmansingh)  

### 📂 **Repository Links**  
- [RetNet Implementation](https://github.com/fkodom/yet-another-retnet)  
- [nanoGPT (Baseline GPT Model)](https://github.com/karpathy/nanoGPT)  
- [Wikitext Datasets](https://huggingface.co/datasets/wikitext)  
- [Shakespeare Dataset](https://huggingface.co/datasets/shakespeare)

---

## 🎯 **Objective**
The project aims to:
- **Analyze** the RetNet architecture’s performance for sequential data modeling.
- **Pretrain models** on large corpora like **Wikitext-2** and **Wikitext-103** to build a robust language model.
- **Fine-tune** the pretrained models on the **Shakespeare dataset** to test adaptability to literary text.
- **Compare** the performance of RetNet with simpler transformer architectures (nanoGPT).

---

## 🔄 **Workflow and Process**  
1. **Research and Model Selection**  
   - Explored RetNet architecture with [open-source implementation](https://github.com/fkodom/yet-another-retnet).  
   - Implemented **nanoGPT** as a baseline for comparison.

2. **Dataset Usage**  
   - **Pretraining**: Used **Wikitext-2** and **Wikitext-103** datasets to build a robust language model.
   - **Fine-tuning**: Adapted the model using the **Shakespeare dataset** to specialize it for literary text generation.

3. **Training and Fine-Tuning Process**  
   - Pretrained both RetNet and nanoGPT models with **Wikitext datasets**.
   - Fine-tuned the pretrained models on the Shakespeare dataset using adjusted hyperparameters.

4. **Evaluation and Comparison**  
   - Evaluated the models' performance using perplexity scores.
   - Generated text samples to compare the literary capabilities of both models.
   
5. **Report and Findings**  
   - Documented insights and key results in the attached **report** (`report.pdf`).

---

## 💻 **Programming Languages and Tools Used**
- **Python**: Core programming language.
- **PyTorch**: Framework for implementing and training models.
- **Hugging Face Datasets**: For loading Wikitext and Shakespeare datasets.
- **Jupyter Notebooks**: For experimentation and visualization.
- **Git**: For version control and collaboration.

---

## 📊 **Results and Insights**  
- **Pretrained RetNet and nanoGPT** on Wikitext corpora.
- **Fine-tuned models on Shakespeare**, successfully generating meaningful literary text.
- Compared models in terms of **perplexity scores** and **text generation quality**.
- The RetNet model demonstrated better adaptability but required careful hyperparameter tuning.

For a detailed breakdown, refer to the **attached report** (`report.pdf`).

---

## 🚀 **Getting Started**  
### Prerequisites
- Python 3.8+  
- PyTorch 2.0+  
- Install dependencies:
  ```bash
  pip install torch datasets
  ```

### Clone the Repository
```bash
git clone https://github.com/keaganjp/EECS595-RetNet-Study
cd EECS595-RetNet-Study
```

### Run Pretraining and Fine-Tuning  
1. **Pretrain** the model with Wikitext datasets:
   ```bash
   python train.py --dataset wikitext-2
   ```

2. **Fine-tune** the pretrained model using the Shakespeare dataset:
   ```bash
   python finetune.py --dataset shakespeare
   ```

---

## 📄 **Report**  
A comprehensive report detailing the experimental setup, training process, and evaluation is available in the attached **`report.pdf`**.

---

## 🤝 **Contributions and Acknowledgments**  
Special thanks to [@anmolmansingh](https://github.com/anmolmansingh) for his contributions. This project builds on **open-source implementations** and datasets mentioned above.

---

## 📝 **License**  
This project is distributed under the MIT License. See `LICENSE` for more details.

---