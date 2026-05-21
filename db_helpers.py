"""
=============================================================================
DATABASE & STORAGE HELPERS (Supabase + Google Docs)
=============================================================================

This module provides utility functions to interact with Supabase (for structured
metadata/JSON storage) and Google Drive/Docs (for human-readable document syncing).

It includes robust fallbacks so that if credentials are missing or incorrect,
the pipeline prints warning logs but does NOT crash, making local testing
and dry runs extremely smooth.
=============================================================================
"""

import os
import json
import base64
import uuid
from dotenv import load_dotenv

# Optional external SDKs
try:
    from supabase import create_client, Client
    SUPABASE_INSTALLED = True
except ImportError:
    SUPABASE_INSTALLED = False

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

# Ensure environment variables are loaded
load_dotenv()

# ===========================================================================
# 1. Supabase Initialization & Database Operations
# ===========================================================================

def get_supabase_client() -> Client:
    """Initialise and return the Supabase client if credentials exist."""
    if not SUPABASE_INSTALLED:
        return None

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        print(f"  ⚠️  Failed to initialise Supabase Client: {e}")
        return None


def create_call_record(title: str = "Untitled Call Analysis") -> str:
    """
    Creates an initial row in the Supabase 'calls' table and returns the call UUID.
    If Supabase is unconfigured, it returns a local mock UUID.
    """
    client = get_supabase_client()
    if not client:
        mock_id = str(uuid.uuid4())
        print(f"  📌  Supabase unconfigured. Created local session ID: {mock_id}")
        return mock_id

    try:
        data = {"title": title}
        response = client.table("calls").insert(data).execute()
        
        # Parse output row ID
        if response.data and len(response.data) > 0:
            call_id = response.data[0].get("id")
            print(f"  ✅  Supabase call record created with ID: {call_id}")
            return call_id
    except Exception as e:
        print(f"  ⚠️  Supabase insert failed (create_call_record): {e}")
    
    # Fallback to local mock ID if database fails
    mock_id = str(uuid.uuid4())
    print(f"  📌  Supabase creation failed. Using local session ID: {mock_id}")
    return mock_id


def update_layer1_data(call_id: str, doc_id: str, transcript: str, speakers: list) -> bool:
    """Updates the Supabase record with Layer 1 Transcription data."""
    client = get_supabase_client()
    if not client or "-" not in call_id:  # Basic UUID check or check if we are in mock mode
        print("  📌  Mock update for Layer 1 Supabase data.")
        return True

    try:
        update_data = {
            "l1_google_doc_id": doc_id,
            "l1_transcript": transcript,
            "l1_speakers": speakers,
        }
        client.table("calls").update(update_data).eq("id", call_id).execute()
        print("  ✅  Layer 1 data successfully pushed to Supabase.")
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to update Layer 1 data in Supabase: {e}")
        return False


def update_layer2_data(call_id: str, doc_id: str, summary: str) -> bool:
    """Updates the Supabase record with Layer 2 Breakdown summary data."""
    client = get_supabase_client()
    if not client or "-" not in call_id:
        print("  📌  Mock update for Layer 2 Supabase data.")
        return True

    try:
        update_data = {
            "l2_google_doc_id": doc_id,
            "l2_summary": summary,
        }
        client.table("calls").update(update_data).eq("id", call_id).execute()
        print("  ✅  Layer 2 data successfully pushed to Supabase.")
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to update Layer 2 data in Supabase: {e}")
        return False


def update_layer3_data(call_id: str, doc_id: str, classification: dict, quality_score: float, metadata: dict = None) -> bool:
    """Updates the Supabase record with Layer 3 Classification & Doc Quality data."""
    client = get_supabase_client()
    if not client or "-" not in call_id:
        print("  📌  Mock update for Layer 3 Supabase data.")
        return True

    try:
        update_data = {
            "l3_google_doc_id": doc_id,
            "l3_classification": classification,
            "l3_doc_quality_score": quality_score,
        }
        if metadata:
            update_data["pipeline_metadata"] = metadata

        client.table("calls").update(update_data).eq("id", call_id).execute()
        print("  ✅  Layer 3 data successfully pushed to Supabase.")
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to update Layer 3 data in Supabase: {e}")
        return False


# ===========================================================================
# 2. Google Drive & Docs API Operations
# ===========================================================================

def get_google_credentials():
    """Resolves and loads the Google Service Account credentials."""
    if not GOOGLE_INSTALLED:
        return None

    # Option A: Loaded from a Base64-encoded string (recommended for Railway)
    b64_creds = os.getenv("GOOGLE_CREDENTIALS_B64")
    if b64_creds:
        try:
            creds_json = json.loads(base64.b64decode(b64_creds).decode("utf-8"))
            return service_account.Credentials.from_service_account_info(
                creds_json, scopes=["https://www.googleapis.com/auth/drive"]
            )
        except Exception as e:
            print(f"  ⚠️  Failed to parse Base64 Google credentials: {e}")

    # Option B: Loaded from a local JSON path (for local dev)
    local_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if local_path and os.path.exists(local_path):
        try:
            return service_account.Credentials.from_service_account_file(
                local_path, scopes=["https://www.googleapis.com/auth/drive"]
            )
        except Exception as e:
            print(f"  ⚠️  Failed to load Google credentials file: {e}")

    return None


def create_gdoc(title: str, content: str) -> str:
    """
    Creates a new Google Doc inside the shared Drive folder and inserts the content.
    Returns the google_doc_id, or a mock string if unconfigured.
    """
    creds = get_google_credentials()
    if not creds:
        dummy_id = f"mock-doc-{uuid.uuid4().hex[:12]}"
        print(f"  📌  Google APIs unconfigured. Created dummy Doc: {title} (ID: {dummy_id})")
        return dummy_id

    try:
        # Build API services
        drive_service = build("drive", "v3", credentials=creds)
        docs_service = build("docs", "v1", credentials=creds)

        # File metadata (assign folder parent if set)
        file_metadata = {
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
        }
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        if folder_id:
            file_metadata["parents"] = [folder_id]

        # 1. Create the Doc
        file = drive_service.files().create(body=file_metadata, fields="id").execute()
        doc_id = file.get("id")
        
        # 2. Write the content to the Doc
        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": content
                }
            }
        ]
        docs_service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests}
        ).execute()

        print(f"  📄  Created Google Doc: '{title}' (ID: {doc_id})")
        return doc_id

    except Exception as e:
        print(f"  ⚠️  Google API doc creation failed: {e}")
        dummy_id = f"failed-doc-{uuid.uuid4().hex[:12]}"
        print(f"  📌  Falling back to dummy doc ID: {dummy_id}")
        return dummy_id
