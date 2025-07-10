
import os
from dotenv import load_dotenv

#Load langchain API
load_dotenv()
os.environ['USER_AGENT'] = 'myagent'
local_llm = "llama3.1"# 'gemma2', 'mistral', 'llama3.1'


from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

prompt_generator = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks.
    ONLY provie answer relevant to the question. DO NOT add anything else.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {Question} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["Question"],
)

llm = ChatOllama(model=local_llm, temperature=0.05)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt_generator | llm | StrOutputParser()







import csv
import pandas as pd

# Load the evaluation dataset

qa_evaluation_dataset = pd.read_csv('your .csv file')

# Initialize list to store results
results = []

# Set the starting row index
start_row = 0
for _, item in qa_evaluation_dataset.iterrows():
    question = item['question']
    print("question:\n", question)
    reference_answer = item.get('answer', 'N/A')  # Retrieve reference answer if available
    # Extract the generated answer and the retrieved context used for answering
    generated_answer = rag_chain.invoke({"Question": question})
    print("generated_answer:\n", generated_answer)

    # Append result to list
    results.append({
        "question": question,
        "answer": generated_answer,
        "ground_truth": reference_answer
    })

# Save results to CSV
output_folder = 'your output folder'
os.makedirs(output_folder, exist_ok=True)
csv_file_path = output_folder + "/" + f"file_name.csv"
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["question", "contexts", "answer", "ground_truth"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_file_path}")