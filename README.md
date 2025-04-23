# Human-like-LLM-benchmark

# Bridging Human and Artificial Cognition in Large Language Models

This project investigates **three critical dimensions** of human-like cognition in Large Language Models (LLMs):  
**Humor Understanding**, **Emotional Intelligence**, and **Commonsense Reasoning**.

Our goal is to systematically evaluate how well current LLMs emulate these human cognitive abilities, and to identify fundamental limitations in their understanding of nuanced social and affective cues.

---

## 🧠 Dimensions of Human-Like Cognition

### 😂 Humor Understanding
We evaluate LLMs on their ability to understand and generate humor using **Chumor**, a challenging Chinese humor dataset.  
Our findings show that current models **perform poorly**, struggling to detect or produce contextually appropriate humor.  
We also present analysis of typical failure cases to highlight where and why models fall short.

### 💬 Emotional Intelligence
We assess how LLMs respond to human emotions across diverse scenarios.  
Although most models show basic **empathy** in simple contexts, they fail to:
- Handle **emotional transitions**
- Capture **implicit affective cues**
- Recognize **cultural and situational context**

This reveals significant weaknesses in **perspective-taking** and **relationship-aware** reasoning.

### 🧠 Commonsense Reasoning
To measure commonsense capabilities, we benchmark LLMs using:
- **CommonsenseQA** (factual reasoning)
- **Social IQa** (social scenarios)
- **PIQA** (physical interaction understanding)

We analyze how models deal with everyday reasoning tasks, uncovering both strengths and blind spots in different reasoning modalities.

---

## 📊 Datasets

| Dimension              | Dataset         | Language  | Focus                         |
|------------------------|-----------------|-----------|-------------------------------|
| Humor Understanding    | Chumor          | Chinese   | Humor comprehension/generation |
| Emotional Intelligence | Custom Scenarios| English   | Emotion detection & response |
| Commonsense Reasoning  | CommonsenseQA, Social IQa, PIQA | English | Factual, social, and physical reasoning |

---

## 📈 Key Findings

- LLMs lack **robust understanding** of humor, often misinterpreting punchlines or failing to detect irony.
- Emotional responses are **surface-level**, with poor generalization to complex interpersonal dynamics.
- While commonsense benchmarks show **reasonable performance**, models struggle with **contextual adaptation**.

---

## 🚀 Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/humanlike-llm-eval.git
cd humanlike-llm-eval
pip install -r requirements.txt
