import time
import sys
import os
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mistralai import ChatMistralAI
from tools.utility import tools
from utils.utils import AgentState

dotenv.load_dotenv()


class Agent:
    def __init__(self):
        self.mistral_model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

        Tool = tools()
        self.tools = Tool.toolkit()
        self.model_with_tool = self.mistral_model.bind_tools(self.tools)

    def system_prompt(self) -> SystemMessage:
        """
        System prompt for Speech Therapy agent.
        Ensures the agent uses the MyProsody tool first,
        and gracefully falls back to manual analysis if the tool fails.
        Supports the full dashboard: fluency, accuracy, grammar, filler words, etc.
        """
        return SystemMessage(
            content=(
                "You are SpeechyBot, an AI speech-therapy assistant.\n"
                "\n"
                "Your job is to evaluate a user's speech and generate structured output "
                "that supports the following dashboard categories:\n"
                "1. Speech Rate Over Time (words per minute)\n"
                "2. Grammar Errors (tense, subject-verb, preposition, article)\n"
                "3. Filler Words (um, like, you know, uh)\n"
                "4. Pronunciation Accuracy (percentage score)\n"
                "5. Fluency Score (speech flow, rhythm, pacing)\n"
                "6. Week-to-week improvement metrics\n"
                "\n"
                "TOOL USAGE RULES:\n"
                "- Always attempt to use the `myprosody` analysis tool first.\n"
                "- If the tool fails, errors, or returns unusable output, "
                "you must fall back to analyzing the raw text directly.\n"
                "\n"
                "MANUAL FALLBACK BEHAVIOR (if tool fails):\n"
                "- Detect fluency based on sentence flow, pauses, rhythm, and cohesion.\n"
                "- Detect pronunciation quality from phonetic patterns, clarity markers, "
                "and overall ease of articulation.\n"
                "- Detect grammar errors and classify them into the four dashboard types.\n"
                "- Count filler words.\n"
                "- Estimate speech rate if timestamps are provided; otherwise infer clarity.\n"
                "- Produce a JSON-friendly structured response with all dashboard fields.\n"
                "\n"
                "STYLE RULES:\n"
                "- Your response must be structured, analytical, and useful for a UI.\n"
                "- Never output tool errors; instead, fall back to your own reasoning.\n"
                "- Always return clean JSON-like dictionaries ready for front-end parsing.\n"
                "\n"
                "Your output MUST contain the following keys:\n"
                "{\n"
                "  'speech_rate': [...],\n"
                "  'grammar_errors': {\n"
                "       'tense': int,\n"
                "       'subject_verb': int,\n"
                "       'preposition': int,\n"
                "       'article': int\n"
                "  },\n"
                "  'filler_words': {\n"
                "       'um': int,\n"
                "       'like': int,\n"
                "       'you_know': int,\n"
                "       'uh': int\n"
                "  },\n"
                "  'fluency_score': str,  # e.g. 'Excellent speech flow and rhythm'\n"
                "  'pronunciation_accuracy': str,  # e.g. 'Good pronunciation clarity'\n"
                "  'improvement': { 'week1': int, 'week2': int, 'week3': int, 'week4': int }\n"
                "}\n"
                "\n"
                "If any metric cannot be reliably computed, provide your best estimate.\n"
            )
        )


    def run_agent(self, state: AgentState, config: RunnableConfig) -> dict:
        """Runs the Helix Agent on the provided state and configuration."""
        time.sleep(3)

        response = self.model_with_tool.invoke(
            [self.system_prompt()] + state["messages"],
            config
        )

        return {"messages": [response]}
