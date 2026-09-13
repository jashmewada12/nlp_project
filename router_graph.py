import pickle
import __main__
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Import your custom classes from your modular file
from nlp_core import preprocess_text, ScratchTfidfVectorizer, ScratchMultinomialNB
from gmail_api import authenticate_gmail, send_routed_email

# THE PICKLE FIX
__main__.ScratchTfidfVectorizer = ScratchTfidfVectorizer
__main__.ScratchMultinomialNB = ScratchMultinomialNB

# Load models using the correct path to your models/ folder
with open('models/custom_tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/custom_naive_bayes_model.pkl', 'rb') as f:
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
    "Technical": "tech-support@yourdomain.com",
    "Billing": "billing@yourdomain.com",
    "Account": "accounts@yourdomain.com",
    "Logistics": "logistics@yourdomain.com"
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
        gmail_service, state["destination_email"], state["subject"], 
        state["original_sender"], state["body"], state["predicted_category"]
    )
    return {"status": "Dispatched" if success else "Failed"}

# Build and Compile the Graph
workflow = StateGraph(TicketState)
workflow.add_node("classifier", classify_ticket)
workflow.add_node("mapper", map_department)
workflow.add_node("dispatcher", dispatch_email)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "mapper")
workflow.add_edge("mapper", "dispatcher")
workflow.add_edge("dispatcher", END)

# This is the variable that app.py is looking for!
router_app = workflow.compile()