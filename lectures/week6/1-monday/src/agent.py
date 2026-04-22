from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model


def get_text(content) -> str:
    """Extract text from model response content, whether it's a str or list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


class EmailState(TypedDict):
    email_text: str
    sender: str
    classification: Literal["spam", "inquiry", "complaint", "urgent", "unknown"]
    priority: Literal["low", "medium", "high"]
    response: str
    escalate: bool


def classify_email(state: EmailState) -> dict:
    model = init_chat_model("us.amazon.nova-2-lite-v1:0", model_provider="bedrock")

    result = model.invoke(
        f"""Classify this email into exactly one category: spam, inquiry, complaint, or urgent.

Email from <senders-email>{state['sender']}</senders-email>:
<senders-email-content>
{state['email_text']}
</senders-email-context>

Respond with only the category name in plain text/string format."""
    )

    classification = get_text(result.content).strip().lower()
    print(f"Here's my classification from the model: {classification}")
    if classification not in ("spam", "inquiry", "complaint", "urgent"):
        classification = "unknown"

    return {"classification": classification}


def route_by_classification(state: EmailState) -> str:
    classification = state["classification"]
    if classification == "spam":
        return "handle_spam"
    elif classification == "urgent":
        return "handle_urgent"
    elif classification == "complaint":
        return "handle_complaint"
    else:
        return "handle_inquiry"


def handle_spam(state: EmailState) -> dict:
    return {
        "priority": "low",
        "response": "[SPAM FILTERED - No response sent]",
        "escalate": False,
    }


def handle_urgent(state: EmailState) -> dict:
    model = init_chat_model("us.amazon.nova-2-lite-v1:0", model_provider="bedrock")
    result = model.invoke(
        f"""Draft a brief acknowledgment for this urgent email.
Be professional and indicate immediate attention.

Email: {state['email_text']}"""
    )
    return {
        "priority": "high",
        "response": get_text(result.content),
        "escalate": True,
    }


def handle_complaint(state: EmailState) -> dict:
    model = init_chat_model("us.amazon.nova-2-lite-v1:0", model_provider="bedrock")
    result = model.invoke(
        f"""Draft a professional response to this complaint.
Acknowledge the issue and express intent to resolve.

Complaint: {state['email_text']}"""
    )
    return {
        "priority": "high",
        "response": get_text(result.content),
        "escalate": True,
    }


def handle_inquiry(state: EmailState) -> dict:
    model = init_chat_model("us.amazon.nova-2-lite-v1:0", model_provider="bedrock")
    result = model.invoke(
        f"""Draft a helpful response to this inquiry.
Be friendly and informative.

Inquiry: {state['email_text']}"""
    )
    return {
        "priority": "medium",
        "response": get_text(result.content),
        "escalate": False,
    }


workflow = StateGraph(EmailState)

workflow.add_node("classify", classify_email)
workflow.add_node("handle_spam", handle_spam)
workflow.add_node("handle_urgent", handle_urgent)
workflow.add_node("handle_complaint", handle_complaint)
workflow.add_node("handle_inquiry", handle_inquiry)

workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "handle_spam": "handle_spam",
        "handle_urgent": "handle_urgent",
        "handle_complaint": "handle_complaint",
        "handle_inquiry": "handle_inquiry",
    },
)

workflow.add_edge("handle_spam", END)
workflow.add_edge("handle_urgent", END)
workflow.add_edge("handle_complaint", END)
workflow.add_edge("handle_inquiry", END)

graph = workflow.compile()
