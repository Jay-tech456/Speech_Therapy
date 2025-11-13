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
        """System prompt for speech thereapy agent."""
        return SystemMessage(
            content=(
                " You are a speaech therapy AI assistant named SpeechyBot"
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
