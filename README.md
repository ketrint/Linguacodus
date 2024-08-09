This repository provides the official implementation for the **Linguacodus: A synergistic framework for transformative code generation in machine learning pipelines** instruction generation part. Linguacodus leverages large language models to automate the process of converting natural language descriptions of ML tasks into executable Python code.


## Features
- **Instruction Generation**: Creates detailed instructions for machine learning task solution using provided data.
- **Instruction Improvement**: Analyzes and refines the generated instructions to ensure they are logical and error-free.
- **Flexible Configuration**: Easily customizable prompts and options to suit various machine learning tasks.


## Getting Started

To get started with the framework, follow these steps:

### 1. Clone the Repository
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/ketrint/Linguacodus.git
cd Linguacodus
```

### 2. Create a Virtual Environment
Set up a virtual environment to manage dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### 3. Install Required Packages
Install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key
To enable instruction refinement using OpenAI's GPT models, set up your API key:

1. Create a `.env` file in the root directory of the project.
2. Add your OpenAI API key to the `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

The repository provides several scripts for different functionalities:

### **1. Llama 2 Fine-Tuning**
Fine-tune the Llama 2 model to generate instructions based on natural language task descriptions:

```bash
python llamafinetune.py
```

The script intputs input_prompts.csv files, containing pre-processed descriptions of machine learning tasks.

### **2. Llama 2 Inference**
Generate the top-3 instruction sets for a given machine learning task using the fine-tuned Llama 2 model:

```bash
python llamainference.py
```

### **3. Instruction Refinement**
Choose the best instruction from the generated options and refine it using OpenAI's GPT model. **Note:** This step requires an OpenAI API key.

```bash
python instruction_improver.py
```

The instructions are further converted to code with the sequential LLM propmting scheme.

## Potential Applications

This framework has broad applications across several domains, including:

- Automated Machine Learning (AutoML)
- Natural Language Processing (NLP)
- Data Science and Analytics

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

