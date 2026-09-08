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
# STUDY KNOWLEDGE BASE
# =====================================

study_tips = """
Pomodoro Technique:
Study for 25 minutes and take a 5-minute break.
After completing four study sessions, take a longer break.

Active Recall:
Instead of simply rereading notes, test yourself by trying to
remember information without looking at the material.
Flashcards and self-questioning are useful.

Spaced Repetition:
Review study material multiple times with increasing intervals
between revision sessions. This improves long-term memory retention.

Time Management:
Break large study tasks into smaller manageable tasks.
Prioritize difficult subjects and topics that require more practice.
Avoid studying too many difficult topics continuously.

Revision Strategy:
Reserve time before examinations for revision.
Focus revision sessions on important concepts, weak topics,
previous mistakes, and practice questions.
"""


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

Subjects:
{state['subjects']}

Days remaining:
{state['days']}

Study hours per day:
{state['hours_per_day']}

Difficult subjects:
{state['difficult_subjects']}

Tasks:

1. Assign priority levels to subjects.
2. Give difficult subjects higher priority.
3. Calculate total available study hours.
4. Suggest time distribution.
5. Keep the answer clear and concise.
"""

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }


# =====================================
# AGENT 2: STUDY TIPS RETRIEVAL
# =====================================

def rag_retrieval(state: StudyState):

    return {
        "rag_tips": study_tips
    }


# =====================================
# AGENT 3: STUDY PLAN GENERATOR
# =====================================

def study_plan_generator(state: StudyState):

    prompt = f"""
You are an AI Study Plan Generator Agent.

Create a personalized day-wise study plan.

Subjects:
{state['subjects']}

Days remaining:
{state['days']}

Study hours per day:
{state['hours_per_day']}

Difficult subjects:
{state['difficult_subjects']}

Study Analysis:
{state['analysis']}

Study Tips:
{state['rag_tips']}

Rules:

1. Create a plan from Day 1 to Day {state['days']}.
2. Do not exceed {state['hours_per_day']} hours per day.
3. Give difficult subjects more time.
4. Include all subjects.
5. Include revision.
6. Include breaks.
7. Keep the plan realistic.
"""

    response = llm.invoke(prompt)

    return {
        "study_plan": response.content
    }


# =====================================
# LANGGRAPH WORKFLOW
# =====================================

workflow = StateGraph(StudyState)

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


study_planner_app = workflow.compile()


# =====================================
# RUN STUDY PLANNER
# =====================================

def run_study_planner(
    subjects,
    days,
    hours_per_day,
    difficult_subjects
):

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

### Personalized Study Planning using LangGraph + Study Knowledge Base + Groq
"""
    )


    with gr.Row():

        with gr.Column():

            subjects = gr.Textbox(
                label="📚 Subjects",
                placeholder="Example: Operating Systems, Machine Learning, Angular"
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
                placeholder="Example: Operating Systems, Machine Learning"
            )

            submit_button = gr.Button(
                "🚀 Generate Study Plan"
            )


        with gr.Column():

            analysis_output = gr.Textbox(
                label="📊 Study Analysis",
                lines=12
            )

            rag_output = gr.Textbox(
                label="📚 Study Tips",
                lines=10
            )

            plan_output = gr.Textbox(
                label="🗓️ Personalized Study Plan",
                lines=20
            )


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
            os.environ.get("PORT", 10000)
        )
    )
