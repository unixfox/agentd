from openai import BaseModel
from pydantic import Field

from astra_assistants.tools.tool_interface import ToolInterface


class Done(BaseModel):
    task: str = Field(..., description="Look at the previous conversation and come up with a detailed and specific "
                                       "description of the task that the user is trying to achieve, include success"
                                       "criteria.")
    next_steps: str = Field(..., description="Pending steps or doubts that still need to be answered to complete the task. "
                                             "Can be a list but as things get completed remove them from the list.")
    is_complete: bool = Field(..., description="Whether the task is complete. "
                                               "Only True if you are very sure of the answer by now.")

    class Config:
        schema_extra = {
            "example1": {
                "task": (
                    "Problem: The user asked to refactor the function to use list comprehension instead of a for loop:"
                ),
                "next_steps": "Refactor the function to use comprehension",
                "is_complete": False
            },
            "example5": {
                "task": (
                    "Problem: So far the assistant has made progress in refactoring the code to use a comprehension"
                ),
                "next_steps": "Ensure you are using tools to write the code to a file.",
                "is_complete": False
            },
            "example6": {
                "Task": (
                    "Problem: Based on the conversation the user needed to "
                    "refactor the function to use list comprehension instead of a for loop"
                ),
                "doubts": "",
                "is_complete": True
            },
        }

    def to_string(self):
        return (
            f"task: {self.task}\n"
            f"next_steps: {self.next_steps}\n"
            f"is_complete: {self.is_complete}\n"
        )


# Define the chain-of-thought tool
class DoneTool(ToolInterface):
    def __init__(self):
        self.chain = []
        self.task = None

    def set_initial_thought(self, thought: Done):
        """Initialize the chain of thought."""
        self.current_thought = thought
        self.chain.append(thought)

    def call(self, args: Done):
        instructions = (
            f"## Context:\n"
            f"{args.to_string()}\n"
        )

        if not args.is_complete:
            instructions += (f"## Instructions: proceed with the task\n")

        print(f"Providing instructions: \n{instructions}")
        return {'output': instructions, 'args': args, 'tool': self.__class__.__name__}