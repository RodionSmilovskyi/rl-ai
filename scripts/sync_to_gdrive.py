#!/usr/bin/env python3
"""
Sync DESIGN.md and README.md to Google Drive 'Drone engineering' folder.
Supports:
1. Cloud Google Drive API sync (no local Google Drive installation needed).
2. Local Google Drive folder sync (if mounted).
3. Local staging fallback (output/google_drive_sync/).
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESIGN_FILE = PROJECT_ROOT / "DESIGN.md"
README_FILE = PROJECT_ROOT / "README.md"
FALLBACK_SYNC_DIR = PROJECT_ROOT / "output" / "google_drive_sync"

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"


def load_env_var(var_name: str) -> str | None:
    val = os.environ.get(var_name)
    if val:
        return val.strip()

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{var_name}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return None


# ==============================================================================
# Cloud Google Drive API Implementation (Zero external dependencies)
# ==============================================================================

def get_cloud_access_token() -> str | None:
    token = load_env_var("GOOGLE_DRIVE_ACCESS_TOKEN")
    if token:
        return token

    # Check gcloud CLI
    try:
        res = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    return None


def execute_drive_request(req: urllib.request.Request) -> dict | None:
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_msg)
            message = err_json.get("error", {}).get("message", err_msg)
            code = err_json.get("error", {}).get("code", e.code)
            status = err_json.get("error", {}).get("status", "")
        except Exception:
            message, code, status = err_msg, e.code, ""
        raise RuntimeError(f"Google Drive API error ({code} {status}): {message}") from e


def find_or_create_cloud_folder(token: str, folder_name: str = "Drone engineering") -> str:
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    url = f"{DRIVE_API_BASE}/files?q={urllib.parse.quote(query)}&fields=files(id,name)&spaces=drive"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = execute_drive_request(req)
    files = data.get("files", []) if data else []

    if files:
        folder_id = files[0]["id"]
        print(f"[INFO] Cloud Google Drive: Found existing folder '{folder_name}' (ID: {folder_id})")
        return folder_id

    # Create folder
    print(f"[INFO] Cloud Google Drive: Creating folder '{folder_name}'...")
    create_url = f"{DRIVE_API_BASE}/files"
    meta = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    req = urllib.request.Request(
        create_url,
        data=json.dumps(meta).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        method="POST",
    )
    res = execute_drive_request(req)
    folder_id = res["id"]
    print(f"[SUCCESS] Cloud Google Drive: Created folder '{folder_name}' (ID: {folder_id})")
    return folder_id


def upload_cloud_file(token: str, folder_id: str, file_path: Path) -> bool:
    file_name = file_path.name
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Check if file exists in folder
    query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
    url = f"{DRIVE_API_BASE}/files?q={urllib.parse.quote(query)}&fields=files(id,name)&spaces=drive"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    data = execute_drive_request(req)
    existing_files = data.get("files", []) if data else []

    if existing_files:
        # Update existing file content
        file_id = existing_files[0]["id"]
        patch_url = f"{DRIVE_UPLOAD_BASE}/files/{file_id}?uploadType=media"
        req = urllib.request.Request(
            patch_url,
            data=file_bytes,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/markdown; charset=UTF-8",
            },
            method="PATCH",
        )
        execute_drive_request(req)
        print(f"[SUCCESS] Cloud Google Drive: Updated '{file_name}' ({len(file_bytes)} bytes) [ID: {file_id}]")
        return True
    else:
        # Create new file using multipart upload
        boundary = "-------" + uuid.uuid4().hex
        meta_json = json.dumps({"name": file_name, "parents": [folder_id]})
        
        body = (
            f"--{boundary}\r\n"
            "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{meta_json}\r\n"
            f"--{boundary}\r\n"
            "Content-Type: text/markdown; charset=UTF-8\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        post_url = f"{DRIVE_UPLOAD_BASE}/files?uploadType=multipart&fields=id,name"
        req = urllib.request.Request(
            post_url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
            method="POST",
        )
        res = execute_drive_request(req)
        file_id = res["id"]
        print(f"[SUCCESS] Cloud Google Drive: Uploaded '{file_name}' ({len(file_bytes)} bytes) [ID: {file_id}]")
        return True


def sync_to_cloud_drive() -> bool:
    token = get_cloud_access_token()
    if not token:
        print("[NOTICE] No Google Cloud / Drive authentication token found.")
        return False

    try:
        folder_id = find_or_create_cloud_folder(token, "Drone engineering")
        upload_cloud_file(token, folder_id, DESIGN_FILE)
        upload_cloud_file(token, folder_id, README_FILE)
        return True
    except RuntimeError as e:
        print(f"[WARN] Cloud Google Drive sync failed: {e}")
        if "insufficient authentication scopes" in str(e).lower() or "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
            print("[ACTION NEEDED] Your gcloud account lacks the Google Drive scope.")
            print("To authorize cloud Google Drive sync, run this command once:")
            print("\n    gcloud auth login --enable-gdrive-access\n")
            print("After logging in, cloud sync will work automatically without any local Drive client.")
        return False
    except Exception as e:
        print(f"[WARN] Cloud Google Drive sync unexpected error: {e}")
        return False


# ==============================================================================
# Local Mount Search (Fallback if Drive for Desktop is mounted)
# ==============================================================================

def find_local_drone_folder() -> Path | None:
    env_path = load_env_var("GOOGLE_DRIVE_DRONE_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

    candidate_paths = [
        Path("/mnt/g/My Drive/Drone engineering"),
        Path("/mnt/g/Drone engineering"),
        Path("/mnt/d/My Drive/Drone engineering"),
        Path("/mnt/c/Users/Rodion Smilovskyi/Google Drive/Drone engineering"),
        Path.home() / "Google Drive" / "Drone engineering",
        Path.home() / "google-drive" / "Drone engineering",
    ]

    for cand in candidate_paths:
        if cand.exists() and cand.is_dir():
            return cand

    return None


def sync_files() -> bool:
    if not DESIGN_FILE.exists():
        print(f"[ERROR] Source file not found: {DESIGN_FILE}", file=sys.stderr)
        return False
    if not README_FILE.exists():
        print(f"[ERROR] Source file not found: {README_FILE}", file=sys.stderr)
        return False

    print("[STEP 1] Attempting Cloud Google Drive API sync...")
    if sync_to_cloud_drive():
        print("[SUCCESS] Files successfully synchronized to Cloud Google Drive folder 'Drone engineering'!")
        return True

    print("\n[STEP 2] Checking for local Google Drive mount...")
    local_target = find_local_drone_folder()
    if local_target:
        print(f"[INFO] Target local Google Drive folder identified at: {local_target}")
        shutil.copy2(DESIGN_FILE, local_target / "DESIGN.md")
        shutil.copy2(README_FILE, local_target / "README.md")
        print(f"[SUCCESS] Copied DESIGN.md -> {local_target / 'DESIGN.md'}")
        print(f"[SUCCESS] Copied README.md -> {local_target / 'README.md'}")
        return True

    print("\n[STEP 3] Local staging fallback...")
    FALLBACK_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DESIGN_FILE, FALLBACK_SYNC_DIR / "DESIGN.md")
    shutil.copy2(README_FILE, FALLBACK_SYNC_DIR / "README.md")
    print(f"[INFO] Staged files locally for sync at: {FALLBACK_SYNC_DIR}")
    print(f"       - {FALLBACK_SYNC_DIR / 'DESIGN.md'}")
    print(f"       - {FALLBACK_SYNC_DIR / 'README.md'}")
    return False


if __name__ == "__main__":
    success = sync_files()
    sys.exit(0 if success else 1)
