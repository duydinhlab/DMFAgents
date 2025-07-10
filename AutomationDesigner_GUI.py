import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOllama
from typing_extensions import TypedDict
from typing import Optional

# Load environment
load_dotenv()
os.environ['USER_AGENT'] = 'myagent'
local_llm = "llama3.1"

### ----- Streamlit UI -----


st.set_page_config(page_title="DropMicroFluidAgents 🧠🔎🌐🧑‍💻 Automation Designer", layout="centered")
st.title("DropMicroFluidAgents 🧠🔎🌐🧑‍💻 Automation Designer")

st.markdown("Powered by **LLM-agents**")

# Dropdown to select language
code_program = st.selectbox("Choose Code Language:", ["AutoLisp", "Python"], index=0)

question = st.text_area("Enter your code-related question/request:", height=200)

if not question:
    st.info("Enter a question and click the button to run the pipeline.")

### ----- Prompt Templates -----
def get_prompt(template_type):
    prompts = {
        "AutoLisp": {
            "generator": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in writing code in AutoLisp.\n
        Generate the proper lines of code based on question.\n
        Only output the code and nothing else.\n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the question:\n\n{question}\n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            "reviewer": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a code reviewer specialized in AutoLisp.\n
    You need to review the given code and potential bugs \n
    and point out issues as bullet list.\n
    ONLY output the bullet list and nothing else
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the code:  \n\n {code} \n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
            "improver": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in improving AutoLisp code.\n
        Improve the given code given the following guidelines.\n
        ONly output the improved code and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the guidelines:  \n\n {guidelines} \n\n
        Here is the given code:  \n\n {code} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "rater": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in evaluating AutoLisp code.\n
        Rate the skills of the coder on a scale of 10 given the following code review circle with a short reason.\n
        ONly output the single score with the key 'score' in JSON format and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the guidelines:  \n\n {guidelines} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "comparator": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in evaluating AutoLisp code.\n
        Compare the two code snippets and rate on a scale of 10 to both..\n
        ONLY output score with the key 'original_code_score' for original code and \n
        the key 'improved_code_score' for revised code in JSON format and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the original code:  \n\n {original code} \n\n
        Here is the improved code:  \n\n {improved code} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "classifier": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in classfiying feedback relating to AutoLisp codes.\n
        Are all feedback mentioned resolved in the code? Output feedback not resolved in JSON format with single key 'not resolved feedback'.\n
        Count the number of not resolved feedbacks and ouput it with single key 'number of not resolved feedback'.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the code:  \n\n {code} \n\n
        Here is the feedback:  \n\n {feedback} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        },


        "Python": {
            "generator": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in writing code in Python.\n
        Generate the proper lines of code based on question.\n
        Only output the code and nothing else.\n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the question:\n\n{question}\n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            "reviewer": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a code reviewer specialized in 'Python'.\n
        You need to review the given code following PEP8 guidelines and potential bugs \n
        and point out issues as bullet list.\n
        ONLY output the bullet list and nothing else
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the code:  \n\n {code} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "improver": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in improving python code.\n
        Improve the given code given the following guidelines.\n
        ONly output the improved code and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the guidelines:  \n\n {guidelines} \n\n
        Here is the given code:  \n\n {code} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "rater": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in evaluating python code.\n
        Rate the skills of the coder on a scale of 10 given the following code review circle with a short reason.\n
        ONly output the single score with the key 'score' in JSON format and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the guidelines:  \n\n {guidelines} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "comparator": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in evaluating python code.\n
        Compare the two code snippets and rate on a scale of 10 to both..\n
        ONLY output score with the key 'original_code_score' for original code and \n
        the key 'improved_code_score' for revised code in JSON format and nothing else.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the original code:  \n\n {original code} \n\n
        Here is the improved code:  \n\n {improved code} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
            "classifier": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert in classfiying feedback relating to python codes.\n
        Are all feedback mentioned resolved in the code? Output feedback not resolved in JSON format with single key 'not resolved feedback'.\n
        Count the number of not resolved feedbacks and ouput it with single key 'number of not resolved feedback'.\n
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the code:  \n\n {code} \n\n
        Here is the feedback:  \n\n {feedback} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        }
    }
    return prompts[code_program][template_type]

### ----- LangChain + LangGraph Setup -----
llm = ChatOllama(model=local_llm, temperature=0)
llm_json = ChatOllama(model=local_llm, format="json", temperature=0)

class GraphState(TypedDict):
    question: Optional[str]
    feedback: Optional[str]
    history: Optional[str]
    code: Optional[str]
    specialization: Optional[str]
    rating: Optional[str]
    iterations: Optional[int]
    code_compare: Optional[str]
    actual_code: Optional[str]

# Chain definitions
code_generator_start = PromptTemplate(
    template=get_prompt("generator"), input_variables=["question"]
) | llm | StrOutputParser()

reviewer_start = PromptTemplate(
    template=get_prompt("reviewer"), input_variables=["code"]
) | llm | StrOutputParser()

code_improver_start = PromptTemplate(
    template=get_prompt("improver"), input_variables=["guidelines", "code"]
) | llm | StrOutputParser()

code_rater_start = PromptTemplate(
    template=get_prompt("rater"), input_variables=["guidelines"]
) | llm_json | JsonOutputParser()

code_comparison_start = PromptTemplate(
    template=get_prompt("comparator"),
    input_variables=["original code", "improved code"]
) | llm_json | JsonOutputParser()

feedback_classifier_start = PromptTemplate(
    template=get_prompt("classifier"),
    input_variables=["code", "feedback"]
) | llm_json | JsonOutputParser()

### ----- Handlers -----
def handle_code_generator(state):
    question = state["question"]
    print("Code Generator working...")
    actual_code = code_generator_start.invoke({"question": question})
    code = actual_code
    history = code

    return {'question': question, 'actual_code': actual_code, 'code': code, 'history': history}

def handle_code_reviewer(state):
    history = state["history"]
    code = state["code"]
    iterations = state["iterations"]
    print("Reviewer working...")

    feedback = reviewer_start.invoke({"code": code})
    return {'history': history + "\n REVIEWER:\n" + feedback, 'feedback': feedback, 'iterations': iterations + 1}

def handle_code_improver(state):
    history = state["history"]
    feedback = state["feedback"]
    code = state["code"]
    print("Code Improver working...")
    code = code_improver_start.invoke({"guidelines": feedback, "code": code})
    return {'history': history + '\n CODER:\n' + code, 'code': code}

def handle_result(state):
    print("Review done...")

    history = state["history"]
    code1 = state["code"]
    code2 = state["actual_code"]
    try:
        rating = code_rater_start.invoke({"guidelines": history})
        code_compare = code_comparison_start.invoke({"original code": code1, "improved code": code2})
        print("history:\n", history)
    except TypeError as e:
        print(f"Error in handle_result: {e}")
        return {}
    return {'rating': rating, 'code_compare': code_compare}

#Add nodes to workflow
workflow = StateGraph(GraphState)
workflow.add_node("Code Generator",handle_code_generator)
workflow.add_node("Code Reviewer",handle_code_reviewer)
workflow.add_node("Code Improver",handle_code_improver)
workflow.add_node("Result",handle_result)
#Defining conditional edges
def is_result_ready(state):
    ready_result = 0
    numberOfFeedback = feedback_classifier_start.invoke({"code": state["code"], "feedback": state["feedback"]})
    if numberOfFeedback==0:
        ready_result = 1
    else:
        ready_result = 0
    total_iterations = 1 if state['iterations']>5 else 0

    if ready_result or total_iterations:
        return "Result"
    return "Code Improver"

workflow.add_conditional_edges(
    "Code Reviewer",
    is_result_ready,
    {
        "Result": "Result",
        "Code Improver": "Code Improver"
    }
)

workflow.set_entry_point("Code Generator")
workflow.add_edge("Code Generator", "Code Reviewer")
workflow.add_edge("Code Improver", "Code Reviewer")
workflow.add_edge("Result", END)

app = workflow.compile()

### ----- Streamlit Button -----
if st.button("Run "):
    with st.spinner("Running... Please wait."):
        final_state = app.invoke({"question": question,"iterations":0})

        st.success("Run complete!")

        st.subheader("🔷 Original Code")
        st.code(final_state.get("actual_code", ""), language=code_program.lower())

        st.subheader("🛠 Final Improved Code")
        st.code(final_state.get("code", ""), language=code_program.lower())

        st.subheader("🧾 Review History")
        st.text(final_state.get("history", ""))

