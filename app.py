import json
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

from router_graph import router_app
from gmail_api import authenticate_gmail, fetch_unread_emails

# --- 1. GLOBAL SERVICE & BACKGROUND AUTOMATION ---
gmail_service = authenticate_gmail()
HISTORY_FILE = "routing_history.json"

async def auto_check_inbox():
    """Continuously checks the inbox every 10 seconds in the background."""
    print(">>> Background automation started. Monitoring inbox... <<<")
    while True:
        await asyncio.to_thread(process_inbox)
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifecycle manager (Replaces the deprecated on_event)."""
    # Boot up the background automation loop on startup
    asyncio.create_task(auto_check_inbox())
    yield

# --- 2. FASTAPI APP INITIALIZATION ---
app = FastAPI(title="NLP Ticket Router API", lifespan=lifespan)

# --- 3. DATABASE LOGGING ---
def log_ticket(sender, subject, category, destination):
    log_entry = {
        "sender": sender,
        "subject": subject,
        "category": category,
        "destination": destination
    }
    
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            
    history.insert(0, log_entry) 
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- 4. INBOX PROCESSING ---
def process_inbox():
    # --- STRICT TRIGGER CONTROL ---
    # The exact email address required to trigger the router
    AUTHORIZED_SENDER = "jashmewada8@gmail.com" 
    
    emails = fetch_unread_emails(gmail_service, target_email=AUTHORIZED_SENDER)
    
    if not emails:
        return # Silent return to avoid terminal clutter when no emails exist
    
    print(f"\n[!] Found {len(emails)} new ticket(s) from {AUTHORIZED_SENDER}. Triggering LangGraph...")
    
    for email in emails:
        initial_state = {
            "original_sender": email["sender"],
            "subject": email["subject"],
            "body": email["body"],
            "predicted_category": "",
            "destination_email": "",
            "status": ""
        }
        
        final_state = router_app.invoke(initial_state)
        
        log_ticket(
            email["sender"], 
            email["subject"], 
            final_state["predicted_category"], 
            final_state["destination_email"]
        )
        print(f" -> Routed ticket '{email['subject']}' to [ {final_state['predicted_category'].upper()} ]")

# --- 5. API ENDPOINTS ---
@app.get("/")
def home():
    # Automatically redirect visitors from root to the dashboard
    return RedirectResponse(url="/dashboard")

@app.get("/trigger-router")
def trigger_router(background_tasks: BackgroundTasks):
    """Manual trigger endpoint."""
    background_tasks.add_task(process_inbox)
    return {"message": "Manual inbox processing triggered."}

@app.get("/api/history")
def get_routing_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    if os.path.exists("dashboard.html"):
        with open("dashboard.html", "r") as f:
            return f.read()
    return "<h1>Error: dashboard.html not found!</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)