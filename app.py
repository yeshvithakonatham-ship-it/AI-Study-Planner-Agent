import os
import gradio as gr

from typing import TypedDict

from langchain_groq import ChatGroq


from langgraph.graph import StateGraph, START, END


# =====================================
# GROQ API KEY
# =====================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =====================================
# GROQ LLM
# =====================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)


# =====================================
# RAG KNOWLEDGE BASE
# =====================================

study_documents = [

    """
    Pomodoro Technique:
    Study for 25 minutes and take a 5-minute break.
    After completing four study sessions, take a longer break.
    This technique helps students maintain focus and avoid burnout.
    """,

    """
    Active Recall:
    Instead of simply rereading notes, test yourself by trying to
    remember information without looking at the material.
    Flashcards and self-questioning are useful active recall techniques.
    """,

    """
    Spaced Repetition:
    Review study material multiple times with increasing intervals
    between each revision session.
    This helps improve long-term memory retention.
    """,

    """
    Time Management:
    Break large study tasks into smaller manageable tasks.
    Prioritize difficult subjects and topics that require more practice.
    Avoid studying too many difficult topics continuously.
    """,

    """
    Revision Strategy:
    Reserve time before examinations for revision.
    Focus revision sessions on important concepts, weak topics,
    previous mistakes, and practice questions.
    """
]

documents = study_documents
# =====================================
# LANGGRAPH STATE
# =====================================

class StudyState(TypedDict):

    subjects: str
    days: int
    hours_per_day: float
    difficult_subjects: str

    analysis: str
    rag_tips: str
    study_plan: str


# =====================================
# AGENT 1: STUDY ANALYZER
# =====================================

def study_analyzer(state: StudyState):

    prompt = f"""
You are an AI Study Analyzer Agent.

Analyze the student's study requirements.

Student Information:

Subjects:
{state['subjects']}

Days remaining until exam:
{state['days']}

Available study hours per day:
{state['hours_per_day']}

Difficult subjects:
{state['difficult_subjects']}


Your tasks:

1. Assign a priority level to each subject.
2. Give difficult subjects higher priority.
3. Calculate the total available study hours.
4. Suggest how study time should be distributed among subjects.
5. Keep the analysis clear and concise.

Return the result in a structured format.
"""

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }


# =====================================
# AGENT 2: LIGHTWEIGHT RAG RETRIEVAL
# =====================================

def rag_retrieval(state: StudyState):

    query = (
        state["subjects"] + " " +
        state["difficult_subjects"]
    ).lower()

    query_words = query.split()

    relevant_docs = []

    for document in documents:

        content = document.lower()

        if any(word in content for word in query_words):
            relevant_docs.append(document)

    if not relevant_docs:
        relevant_docs = documents

    rag_content = "\n\n".join(
        document.strip()
        for document in relevant_docs[:3]
    )

    return {
        "rag_tips": rag_content
    }

# =====================================
# AGENT 3: STUDY PLAN GENERATOR
# =====================================

def study_plan_generator(state: StudyState):

    prompt = f"""
You are an AI Study Plan Generator Agent.

Create a personalized and practical day-wise study plan.

Student Information:

Subjects:
{state['subjects']}

Days remaining:
{state['days']}

Available study hours per day:
{state['hours_per_day']}

Difficult subjects:
{state['difficult_subjects']}


Study Analysis:
{state['analysis']}


Relevant Study Tips from Knowledge Base:
{state['rag_tips']}


Rules:

1. Create a plan for every day from Day 1 to Day {state['days']}.
2. Do not exceed {state['hours_per_day']} study hours per day.
3. Give more time to difficult subjects.
4. Include all subjects.
5. Use relevant study techniques from the provided study tips.
6. Include revision time.
7. Include short breaks where appropriate.
8. Keep the plan realistic and easy to follow.

Return a clean and well-structured day-wise study plan.
"""

    response = llm.invoke(prompt)

    return {
        "study_plan": response.content
    }


# =====================================
# LANGGRAPH WORKFLOW
# =====================================

workflow = StateGraph(StudyState)


# Add nodes

workflow.add_node(
    "study_analyzer",
    study_analyzer
)

workflow.add_node(
    "rag_retrieval",
    rag_retrieval
)

workflow.add_node(
    "study_plan_generator",
    study_plan_generator
)


# Connect nodes

workflow.add_edge(
    START,
    "study_analyzer"
)

workflow.add_edge(
    "study_analyzer",
    "rag_retrieval"
)

workflow.add_edge(
    "rag_retrieval",
    "study_plan_generator"
)

workflow.add_edge(
    "study_plan_generator",
    END
)


# Compile workflow

study_planner_app = workflow.compile()


# =====================================
# GRADIO FUNCTION
# =====================================

def run_study_planner(
    subjects,
    days,
    hours_per_day,
    difficult_subjects
):

    # Validate input

    if not subjects.strip():
        return (
            "Please enter at least one subject.",
            "",
            ""
        )

    if not difficult_subjects.strip():
        difficult_subjects = "None specified"


    result = study_planner_app.invoke({

        "subjects": subjects,

        "days": int(days),

        "hours_per_day": float(hours_per_day),

        "difficult_subjects": difficult_subjects
    })


    return (

        result["analysis"],

        result["rag_tips"],

        result["study_plan"]
    )


# =====================================
# GRADIO USER INTERFACE
# =====================================

with gr.Blocks(
    title="AI Study Planner Agent"
) as demo:


    gr.Markdown(
        """
# 🤖 AI Study Planner Agent

### Personalized Study Planning using LangGraph + Lightweight RAG + Groq
"""
    )


    with gr.Row():


        # =============================
        # INPUT SECTION
        # =============================

        with gr.Column():


            subjects = gr.Textbox(

                label="📚 Subjects",

                placeholder=
                "Example: Operating Systems, Machine Learning, Angular"
            )


            days = gr.Number(

                label="📅 Days Remaining",

                value=7
            )


            hours_per_day = gr.Number(

                label="⏰ Study Hours Per Day",

                value=4
            )


            difficult_subjects = gr.Textbox(

                label="🔥 Difficult Subjects",

                placeholder=
                "Example: Operating Systems, Machine Learning"
            )


            submit_button = gr.Button(

                "🚀 Generate Study Plan"
            )


        # =============================
        # OUTPUT SECTION
        # =============================

        with gr.Column():


            analysis_output = gr.Textbox(

                label="📊 Study Analysis",

                lines=12
            )


            rag_output = gr.Textbox(

                label="📚 RAG Study Tips",

                lines=10
            )


            plan_output = gr.Textbox(

                label="🗓️ Personalized Study Plan",

                lines=20
            )


# =====================================
# BUTTON ACTION
# =====================================

submit_button.click(

    fn=run_study_planner,

    inputs=[

        subjects,

        days,

        hours_per_day,

        difficult_subjects
    ],

    outputs=[

        analysis_output,

        rag_output,

        plan_output
    ]
)


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )
