import pickle
from typing import TypedDict

from langgraph.graph import END, StateGraph

import __main__
from gmail_api import authenticate_gmail, send_routed_email
from nlp_core import ScratchMultinomialNB, ScratchTfidfVectorizer, preprocess_text

__main__.ScratchTfidfVectorizer = ScratchTfidfVectorizer
__main__.ScratchMultinomialNB = ScratchMultinomialNB


with open("models/custom_tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("models/custom_naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Authenticate Gmail
gmail_service = authenticate_gmail()


# LangGraph State and Routing Logic
class TicketState(TypedDict):
    original_sender: str
    subject: str
    body: str
    predicted_category: str
    destination_email: str
    status: str


DEPARTMENT_MAP = {
    "Technical": "shravanmore777@student.sfit.ac.in",
    "Billing": "manavlakhani96@student.sfit.ac.in",
    "Account": "dswanand14@student.sfit.ac.in",
    "Logistics": "mewadajash94@student.sfit.ac.in",
}


def classify_ticket(state: TicketState):
    tokens = preprocess_text(state["body"])
    vec = vectorizer.transform([tokens])
    prediction = model.predict(vec)[0]
    return {"predicted_category": prediction}


def map_department(state: TicketState):
    category = state.get("predicted_category", "Technical")
    return {"destination_email": DEPARTMENT_MAP.get(category, "general@yourdomain.com")}


def dispatch_email(state: TicketState):
    success = send_routed_email(
        gmail_service,
        state["destination_email"],
        state["subject"],
        state["original_sender"],
        state["body"],
        state["predicted_category"],
    )
    return {"status": "Dispatched" if success else "Failed"}


workflow = StateGraph(TicketState)
workflow.add_node("classifier", classify_ticket)
workflow.add_node("mapper", map_department)
workflow.add_node("dispatcher", dispatch_email)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "mapper")
workflow.add_edge("mapper", "dispatcher")
workflow.add_edge("dispatcher", END)


router_app = workflow.compile()
