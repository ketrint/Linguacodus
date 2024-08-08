# Machine Learning Pipelines Synthesis with Large Language Models

Paper official repository.

Automatic transformation of Machine Learning (ML) task natural description into Python code through the textual instruction.

## Features
- **Instruction Generation**: Creates detailed instructions for machine learning task solution using provided data.
- **Instruction Improvement**: Analyzes and refines the generated instructions to ensure they are logical and error-free.
- **Flexible Configuration**: Easily customizable prompts and options to suit various machine learning tasks.


## Getting Started

To use our framework, follow the steps outlined below:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

```bash
git clone https://github.com/ketrint/Linguacodus.git

cd Linguacodus
```

2. **Create a Virtual Environment**
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. **Install Required Packages**
```bash
pip install -r requirements.txt
```

4. **Set Up OpenAI API Key**
- Create a .env file in the root directory of the project.
- Add your OpenAI API key to the .env file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

**Llama 2 fine-tune**: fine-tune Llama 2 model to generate instructions by natural language tasks decriptions.
```bash
python llamafinetune.py
```

**Llama 2 inference**: infer top-3 instructions for a given ML task .
```bash
python llamainference.py
```

**Instructions refinement**: choose the best out of three generated instructions and improve it.
Requires OPENAI_API_KEY

```bash
instruction_improver.py
```

## Potential Applications

- Automated Machine Learning (AutoML)
- Natural Language Processing (NLP)
- Data Science and Analytics

This project is licensed under the MIT License.
