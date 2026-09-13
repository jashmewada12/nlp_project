import os.path
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://mail.google.com/']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def fetch_unread_emails(service, target_email):
    search_query = f"is:unread from:{target_email}"
    results = service.users().messages().list(userId='me', q=search_query).execute()
    messages = results.get('messages', [])
    
    email_data = []
    if not messages:
        return email_data

    for msg in messages:
        msg_id = msg['id']
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        
        headers = message['payload']['headers']
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
        sender = next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown')
        body = message.get('snippet', '')
        
        email_data.append({"id": msg_id, "sender": sender, "subject": subject, "body": body})
        
        # Mark as READ
        service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
        
    return email_data

def send_routed_email(service, to_email, subject, original_sender, body, category):
    msg = EmailMessage()
    msg.set_content(f"--- ROUTED TICKET ---\nFrom: {original_sender}\nCategory: {category}\n\n{body}")
    msg['To'] = to_email
    msg['From'] = "me"
    msg['Subject'] = f"[{category.upper()} TICKET] {subject}"
    
    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        service.users().messages().send(userId="me", body={'raw': encoded_message}).execute()
        return True
    except Exception as e:
        print(f"Failed to send: {e}")
        return False