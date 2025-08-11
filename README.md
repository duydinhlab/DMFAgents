# DropMicroFluidAgents (DMFAs) 🧠🔎🌐🧑‍💻

This repository accompanies the manuscript:  
**DropMicroFluidAgents (DMFAs): Autonomous Droplet Microfluidic Research Framework Through Large Language Model Agents**  
*Dinh-Nguyen Nguyen, Raymond Kai-Yu Tong, Ngoc-Duy Dinh*  
Department of Biomedical Engineering, The Chinese University of Hong Kong

**DropMicroFluidAgents (DMFAs)** is a novel multi-agent framework that leverages large language model (LLM) agents to automate and optimize research workflows in droplet microfluidics. It integrates scientific reasoning, machine learning, and CAD design automation into an intelligent and user-friendly system.


▶️ [DEMO DMFAgents](https://github.com/duydinhlab/DMFAgents/blob/main/DEMO/DEMO-DMFAs.mp4)

## 📬 Contact

- **Prof Ngoc-Duy Dinh** (Corresponding Author)  
  Email: ngocduydinh@cuhk.edu.hk
  
  Address: BME office, Room 1120, 11/F, William M.W. Mong Engineering Building, or 
  Room 208, Ho Sin Hang Engineering Building (SHB), The Chinese University of Hong Kong, Shatin, N.T., Hong Kong


## 🔍 Overview

DMFAs includes two main agents:
- **Scientific Mentor**: Provides accurate, domain-specific responses using Retrieval-Augmented Generation (RAG) and multiple LLMs (LLAMA 3.1, MISTRAL, GEMMA2).
- **Automation Designer**: Automatically generates machine learning models and CAD scripts for droplet microfluidic devices.


## 📈 Key Features

- **Accurate Q&A** using a custom microfluidics knowledge base
- **Performance Tuning** via prompt engineering, embedding models, chunking strategies, and decoding parameters
- **CAD Design Generation** with validated AutoLISP scripts for AutoCAD
- **Machine Learning Automation** for droplet rate prediction (R² = 0.96)
- **Interactive GUI** for end-user accessibility

## 📊 Performance

| Model | QA Accuracy | ML Model R² | CAD Support |
|-------|-------------|-------------|-------------|
| LLAMA 3.1 + DMFA | 76.15% | 0.96 | ✅ |
| GEMMA2 + DMFA | +34.47% improvement | – | ✅ |
| MISTRAL + DMFA | 72.00% | – | ✅ |

## 📁 Repository Structure

```
DMFAgents/
├── data/               # Microfluidics knowledge base, Q&A dataset
├── src/
│   ├── mentor/         # Scientific Mentor logic
│   ├── designer/       # Automation Designer logic (ML & CAD)
│   └── utils/          # Prompting, embedding, and evaluation tools
├── gui/                # Streamlit-based UI for users
├── models/             # Pretrained models and evaluation results
└── README.md
```

## 🧪 Evaluation

Evaluation includes:
- Accuracy and F1-score comparison of LLM and agent-based systems
- Performance across prompt strategies, embedding models, chunk sizes, and decoding parameters (top-p, top-k, temperature)
- Human expert verification of Q&A and CAD outputs


# DropMicroFluidAgents 🧠🔎🌐🧑‍💻 Scientific Mentor

This project implements a Scientific Mentor using LangChain, ChromaDB, SentenceTransformer embeddings, and a local Large Language Model (LLM) via Ollama. It processes PDF documents, chunks them into vector embeddings, and uses a modular LangGraph flow to answer scientific questions, particularly for droplet-based microfluidics research.

---

## Quick Start (Download and Run)

You do **not** need to clone the entire repo! Simply download the ` ScientificMentor_GUI.py ` file from GitHub and run it directly.


# Features

Load and chunk PDF documents using LangChain.
Embed and store documents in a local Chroma vectorstore.
Use a local LLM (e.g., llama3.1) for answer generation, grading, and routing.
Modular LangGraph flow including:
•	Question routing (vectorstore vs. web search)
•	Document relevance grading
•	Hallucination grading
•	Answer utility grading
Web search fallback with Tavily API

### 1. Download the Script

- Click the `Raw` button on the GitHub file view for ` ScientificMentor_GUI py `  
- Save the file to your local machine

### 2. Install Required Dependencies
you can install the required packages manually:
pip install langchain chromadb sentence-transformers PyPDF2 tavily-python nest_asyncio python-dotenv 
Note:
Python ≥ 3.9
Ollama installed and running locally (for models like llama3.1)
Tavily API key for web search

### 3. Create .env File (Optional)
If you want to customize environment variables, create a .env file in the same directory as ScientificMentor.py

LLAMA_CLOUD_API_KEY=your_llama_api_key_here

COHERE_API_KEY=your_cohere_api_key_here

TAVILY_API_KEY=your_tavily_api_key_here

USER_AGENT=myagent 

If you skip this, the default USER_AGENT environment variable will be set by the app automatically.

### 4. PDF Setup
You may edit this in the script:
pdf_folder_path = '/your/path/to/RAG_droplet_downloaded_papers'

chroma_db_path = '/your/path/to/DATABASE'

The script will:
•	Extract text from all PDFs

•	Split into retrievable chunks

•	Store them in a persistent Chroma vector database

### 5. Run the App
Run the Streamlit app from your terminal:
streamlit run ScientificMentor_GUI.py

This will launch a local web server and open the app in your default browser.
Troubleshooting

No module named X → Install missing module via pip install X

Ollama not running → Make sure ollama is running locally

Missing .env → Ensure .env is loaded and includes all necessary keys



# DropMicroFluidAgents 🧠🔎🌐🧑‍💻 Automation Designer

A Design Automation web app powered by **LLM-agents** that automates code generation, review, improvement, and evaluation for AutoLisp and Python code.  This application leverages LangGraph, LangChain, and the ChatOllama local LLM model (`llama3.1`) to create an intelligent code design assistant workflow.

---

## Features

- Generate code snippets from natural language questions  
- Review generated code for potential bugs and adherence to best practices  
- Improve and optimize code based on reviewer feedback  
- Rate and compare original vs improved code snippets  
- Classify unresolved feedback for iterative improvement  
- Supports AutoLisp and Python code languages  
- Runs locally using a lightweight, local LLM (ChatOllama) for privacy and speed  
- Simple Streamlit interface for easy user interaction  

---

## Quick Start (Download and Run)

You do **not** need to clone the entire repo! Simply download the ` AutomationDesigner_GUI.py ` file from GitHub and run it directly.

### 1. Download the Script

- Click the `Raw` button on the GitHub file view for ` AutomationDesigner_GUI.py `
-   
- Save the file to your local machine

### 2. Install Required Dependencies
you can install the required packages manually:
pip install streamlit python-dotenv langgraph langchain-core langchain-community typing-extensions
Note:
•	You must have Python 3.8 or higher installed.

•	You need to have a local Ollama LLM model (llama3.1) installed and running. See the Ollama docs for setup instructions.

### 3. Create .env File (Optional)
If you want to customize environment variables, create a .env file in the same directory as DesignAutomation.py

LLAMA_CLOUD_API_KEY=your_llama_api_key_here

USER_AGENT =myagent

If you skip this, the default USER_AGENT environment variable will be set by the app automatically.
✅ Make sure your .env file is saved and loaded before running the script.

### 4. Run the App
Run the Streamlit app from your terminal:
streamlit run AutomationDesigner_GUI.py

This will launch a local web server and open the app in your default browser.
How to Use

1.	Select the programming language (AutoLisp or Python) from the dropdown.
2.	Enter your code-related question or request in the text area.
3.	Click the Run button to start the automated pipeline.
4.	View the outputs:
   
o	Original generated code
o	Final improved code after iterative reviews and improvements
o	Review history detailing feedback and changes

Code Structure Overview
•	Streamlit UI: Simple user interface for input and output

•	Prompt Templates: Custom prompts tailored for AutoLisp and Python code tasks (generation, review, improvement, rating, classification)

•	LangGraph Workflow: Defines nodes and transitions between code generator, reviewer, improver, and result stages

•	Local LLM (ChatOllama): Powered by a local LLM for text generation and JSON parsing

•	Environment loading: Uses python-dotenv to load environment variables

Requirements

•	Python 3.8+
•	Streamlit
•	python-dotenv
•	LangGraph
•	LangChain Core
•	LangChain Community
•	Typing Extensions
•	Local Ollama LLM with llama3.1 model installed and running

Troubleshooting

•	"ModuleNotFoundError": Make sure all required packages are installed via pip.

•	LLM model errors: Ensure you have the llama3.1 model properly installed locally and Ollama daemon/service is running.

•	Streamlit UI issues: Clear browser cache or try running in an incognito/private window.

•	Environment variables not loaded: Confirm your .env file is in the same directory or manually set variables in your OS environment.


## 🔗 GitHub

Visit the project repository: [https://github.com/duydinhlab/DMFAgents](https://github.com/duydinhlab/DMFAgents)


## 🧠 Citation

Please cite our work if you find this useful:

```bibtex
@article{nguyen2025dmfas,
  title={DropMicroFluidAgents: Autonomous Droplet Microfluidic Research Framework Through Large Language Model Agents},
  author={Nguyen, Dinh-Nguyen and Tong, Raymond Kai-Yu and Dinh, Ngoc-Duy},
  journal={arXiv.2501.14772},
  year={2025},
  note={https://doi.org/10.48550/arXiv.2501.14772}
}



