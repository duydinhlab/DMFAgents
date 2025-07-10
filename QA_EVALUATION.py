
import os
from dotenv import load_dotenv

#Load langchain API
load_dotenv()
os.environ['USER_AGENT'] = 'myagent'

import re
import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


import csv
import pandas as pd
EVALUATION_PROMPT = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
Evaluate the given response based on the question, answer, ground truth.
1. Write feedback that assesses the response quality strictly per the scale.
2. After writing feedback, provide a score between 0 and 100 as per scale.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} SCORE: {{an integer number between 0 and 100}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include SCORE in your output.

###Score scale:
# Score 0-5: The response is irrelevant, nonsensical, or incoherent; no effort to address the question.
# Score 6-10: An attempt is made, but the response is entirely unrelated or meaningless.
# Score 11-15: Contains vague or random terms, but lacks clarity, relevance, and logic.
# Score 16-20: Slight relevance, but the response is mostly incorrect, unclear, and fails to address the question.
# Score 21-25: Marginal effort to address the question, but lacks clarity and coherence; mostly irrelevant.
# Score 26-30: Some fragments of relevance, but the response is poorly structured and fails to convey useful information.
# Score 31-35: Displays minimal understanding, with major inaccuracies and a lack of focus on the question.
# Score 36-40: Partially aligned with the question but overly verbose, unclear, or dominated by factual errors.
# Score 41-45: Demonstrates basic understanding  with significant errors or irrelevant details.
# Score 46-50: Covers core aspects but is imprecise, verbose, or unclear; lacks depth or includes notable inaccuracies.
# Score 51-55: Partially correct, with omissions or minor inaccuracies; somewhat clear and relevant.
# Score 56-60: Mostly relevant and clear, but lacks focus or includes unnecessary details; broadly acceptable.
# Score 61-65: Addresses the question clearly  with minor inaccuracies or slight verbosity.
# Score 66-70: A solid, clear response that aligns with the question and ground truth.
# Score 71-75: Comprehensive, clear, and precise, with only minor omissions or redundant elements.
# Score 76-80: Nearly flawless response; highly clear and relevant, with only slight room for improvement.
# Score 81-85: A thorough and precise response; clear and directly addresses the question with full relevance.
# Score 86-90: Excellent, highly clear, and succinct answer; covers all aspects comprehensively and precisely.
# Score 91-95: Virtually flawless; exceptional clarity and relevance, with added depth where appropriate.
# Score 96-100: Perfect response; unparalleled clarity, precision, and focus on the question, with no room for improvement.
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    Question: {Question} 
    Answer: {Answer} 
    Ground Truth Answer: {Ground Truth Answer}
    Feedback:""
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=[ "Question", "Answer","Ground Truth Answer"],
)

qa_evaluation_dataset = pd.read_csv('your .csv file')

local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0.0)
EVALUATION_PROMPT_Start = EVALUATION_PROMPT | llm | StrOutputParser()

# Folder where results will be saved
output_folder = 'your ouput folder'
os.makedirs(output_folder, exist_ok=True)



# File path for saving results
output_file_path = 'output .csv file'

# Initialize list to store results
results = []
# Max score for percentage calculation
MAX_SCORE = 100
for _, item in qa_evaluation_dataset.iterrows():
    question = item['question']
    reference_answer = item.get('ground_truth', 'N/A')  # Retrieve reference answer if available
    generated_answer = item.get('answer', 'N/A')

    # Invoke the LLM with the evaluation prompt
    response = EVALUATION_PROMPT_Start.invoke(
        {"Question": question, "Answer": generated_answer, "Ground Truth Answer": reference_answer}
    )
    # print("response:\n", response)

    # Extract Feedback and SCORE using regex
    feedback_match = re.search(r'Feedback:\s*(.*)', response, re.DOTALL)
    score_match = re.search(r'SCORE:\s*(\d{1,3})', response)

    feedback = feedback_match.group(1).strip() if feedback_match else "N/A"
    score = int(score_match.group(1)) if score_match else None

    # Calculate percentage
    percentage = (score / MAX_SCORE) * 100 if score is not None else None

    # Append result to list
    results.append({
        "question": question,
        "answer": generated_answer,
        "ground_truth": reference_answer,
        "feedback": feedback,
        "score": score,
        "percentage": percentage
    })

    # Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Calculate statistics for 'score' and 'percentage'
valid_scores = results_df['score'].dropna().astype(int)
mean_score = valid_scores.mean()
std_error = valid_scores.std() / np.sqrt(len(valid_scores))

valid_percentages = results_df['percentage'].dropna()
mean_percentage = valid_percentages.mean()
std_error_percentage = valid_percentages.std() / np.sqrt(len(valid_percentages))

# Save results with scores and percentages to a CSV file
results_df.to_csv(output_file_path, index=False)

