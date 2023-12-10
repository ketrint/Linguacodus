# Machine Learning Pipelines Synthesis with Large Language Models

## Abstract

In the realm of machine learning, seamlessly translating natural language descriptions into compilable code is a longstanding challenge. This paper presents a novel framework that addresses this challenge by introducing a pipeline capable of iteratively transforming natural language task descriptions into code through high-level machine learning instructions. Central to this framework is the fine-tuning of the Llama large language model, enabling it to rank different solutions for various problems and select an appropriate fit for a given task. We also cover the fine-tuning process and provide insights into transforming natural language descriptions into code. Our approach marks a significant step towards automating code generation, bridging the gap between task descriptions and executable code, and holds promise for advancing machine learning applications across diverse domains. We showcase the effectiveness of our framework through experimental evaluations and discuss its potential applications in various domains, highlighting its implications for advancing the field of machine learning.

## Main Contributions

1. **A Controllable Transformation Framework**: We introduce a framework for the controlled transformation of ML task natural language descriptions into suitable high-level solution instructions. The framework involves fine-tuning the Llama-2 model using pairs of ML task descriptions and instructions retrieved with GPT-3.5.

2. **Instruction-Based Sequential Generation**: We demonstrate that executing instructions for sequential generation leads to producing compilable code backed by promising results based on evaluation metrics.

## Getting Started

To use our framework, follow the steps outlined below:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

```bash
git clone https://github.com/Deltax2016/NL4ML.git
```

2. Run the Framework: Execute the inference notebook to initiate the pipeline for transforming natural language task descriptions into code.

## Potential Applications

Our framework has promising applications in various domains, including:

- Automated Machine Learning (AutoML)
- Natural Language Processing (NLP)
- Data Science and Analytics
