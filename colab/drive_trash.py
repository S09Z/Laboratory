"""Permanently remove old checkpoints that Colab's Drive mount moved to the Drive Trash.

Deleting through /content/drive only *trashes* files, and trashed files still count
against the Drive quota (~1.7 GB per 500 training steps). This deletes, permanently,
only files that are (a) already in the Trash, (b) named like an SBV2 checkpoint, and
(c) were inside the given folders. Nothing else in the user's Trash is touched.
"""
from __future__ import annotations

import re

CHECKPOINT_NAME = re.compile(r"^(?:(?:G|D|WD|DUR)_\d+\.pth|.+_e\d+_s\d+\.safetensors)$")


def drive_service():
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    return build("drive", "v3", cache_discovery=False)


def folder_id(service, path_parts: list[str]) -> str:
    """Drive folder id for MyDrive/<parts...>."""
    parent = "root"
    for name in path_parts:
        q = (f"name = '{name}' and '{parent}' in parents and "
             "mimeType = 'application/vnd.google-apps.folder' and trashed = false")
        files = service.files().list(q=q, fields="files(id)", pageSize=10).execute().get("files", [])
        if not files:
            raise FileNotFoundError("/".join(path_parts))
        parent = files[0]["id"]
    return parent


def purge_trashed_checkpoints(service, folder_ids: set[str]) -> int:
    q = "trashed = true and (name contains '.pth' or name contains '.safetensors')"
    token, n = None, 0
    while True:
        resp = service.files().list(q=q, fields="nextPageToken, files(id, name, parents)",
                                    pageSize=200, pageToken=token).execute()
        for f in resp.get("files", []):
            if CHECKPOINT_NAME.match(f["name"]) and set(f.get("parents", [])) & folder_ids:
                service.files().delete(fileId=f["id"]).execute()
                n += 1
        token = resp.get("nextPageToken")
        if not token:
            return n
