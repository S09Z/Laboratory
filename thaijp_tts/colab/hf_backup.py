"""Back up / restore training checkpoints on the Hugging Face Hub (private repo).

Kaggle's /kaggle/working is lost when the session ends, so each new checkpoint set
(G/D/WD .pth + inference .safetensors + config + style vectors) is pushed to HF and
restored at the start of the next session. A set is only usable once its ``DONE``
marker exists, so an interrupted upload is never restored. Backup problems must never
stop training: ``push`` swallows errors and reports them.
"""
from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .ckpt import PARTS, complete_steps, safetensors_steps

PREFIX = "ckpt"
DEFAULT_REPO_NAME = "thaijp-tts-ckpt"
EXTRA_FILES = ("config.json", "style_vectors.npy")


def get_secret(name: str) -> str | None:
    """Kaggle Secret (Add-ons > Secrets, attached to the notebook) or environment variable."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 — not on Kaggle / secret missing
        return os.environ.get(name)


class HfBackup:
    def __init__(self, repo_id: str, models_dir: Path, assets_dir: Path, token: str | None = None,
                 api=None, download: Callable | None = None, keep: int = 2,
                 sleep: Callable[[float], None] = time.sleep, wait_s: float = 120, settle_s: float = 20):
        self.repo_id = repo_id
        self.models_dir = Path(models_dir)
        self.assets_dir = Path(assets_dir)
        self.keep = keep
        self.sleep = sleep
        self.wait_s = wait_s
        self.settle_s = settle_s
        self.last_pushed = 0
        if api is None:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi(token=token)
            download = download or (lambda repo_id, filename, local_dir: hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=local_dir, token=token))
        self.api = api
        self.download = download

    # ---- push ----
    def _safetensors(self, step: int) -> Path | None:
        return next((p for s, p in safetensors_steps(self.assets_dir) if s == step), None)

    def push(self) -> bool:
        """Upload the newest complete checkpoint set. True if something was uploaded; never raises."""
        try:
            return self._push()
        except Exception as e:  # noqa: BLE001 — backup must never stop training
            print("⚠️ สำรองขึ้น Hugging Face ไม่สำเร็จ (จะลองใหม่รอบหน้า):", e, flush=True)
            return False

    def _push(self) -> bool:
        steps = complete_steps(self.models_dir)
        if not steps or steps[-1] <= self.last_pushed:
            return False
        step = steps[-1]
        waited = 0.0
        while self._safetensors(step) is None and waited < self.wait_s:   # trainer writes it right after G/D/WD
            self.sleep(10)
            waited += 10
        self.sleep(self.settle_s)   # let the last writes of the .pth files finish
        self.api.create_repo(self.repo_id, private=True, exist_ok=True)
        files = [self.models_dir / f"{p}_{step}.pth" for p in PARTS]
        st = self._safetensors(step)
        if st is not None:
            files.append(st)
        files += [self.assets_dir / n for n in EXTRA_FILES if (self.assets_dir / n).exists()]
        for f in files:
            self.api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{PREFIX}/{step}/{f.name}",
                                 repo_id=self.repo_id)
        self.api.upload_file(path_or_fileobj=b"ok", path_in_repo=f"{PREFIX}/{step}/DONE", repo_id=self.repo_id)
        self.last_pushed = step
        self._prune_remote()
        print(f"☁️ สำรอง checkpoint step {step} ขึ้น {self.repo_id} แล้ว ({len(files)} ไฟล์)", flush=True)
        return True

    def _remote_sets(self) -> dict[int, list[str]]:
        sets: dict[int, list[str]] = {}
        for f in self.api.list_repo_files(self.repo_id):
            parts = f.split("/")
            if len(parts) == 3 and parts[0] == PREFIX and parts[1].isdigit():
                sets.setdefault(int(parts[1]), []).append(f)
        return sets

    def _prune_remote(self) -> None:
        sets = self._remote_sets()
        for step in sorted(sets)[:-self.keep]:
            for f in sets[step]:
                self.api.delete_file(path_in_repo=f, repo_id=self.repo_id)

    # ---- restore ----
    def restore(self) -> int | None:
        """Download the newest DONE set into models/ and model_assets/. None if local files exist or nothing to restore."""
        if complete_steps(self.models_dir):
            return None   # local checkpoints win
        try:
            sets = {s: fs for s, fs in self._remote_sets().items() if f"{PREFIX}/{s}/DONE" in fs}
        except Exception as e:  # noqa: BLE001 — repo missing on first run
            print("ไม่พบ repo สำรอง (ปกติถ้าเป็นการเทรนครั้งแรก):", e, flush=True)
            return None
        if not sets:
            return None
        step = max(sets)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(sets[step]):
            name = f.rsplit("/", 1)[1]
            if name == "DONE":
                continue
            dest = self.models_dir if name.endswith(".pth") else self.assets_dir
            got = Path(self.download(repo_id=self.repo_id, filename=f, local_dir=str(dest)))
            target = dest / name
            if got != target:   # hf_hub_download(local_dir=...) keeps repo subfolders (ckpt/<step>/..) → flatten
                shutil.move(str(got), str(target))
        print(f"☁️ กู้ checkpoint step {step} จาก {self.repo_id}", flush=True)
        return step


def make_backup(repo_id: str, models_dir: Path, assets_dir: Path, token: str | None = None,
                api=None, **kw) -> HfBackup | None:
    """None (with a hint) when no HF_TOKEN is available; repo defaults to '<your HF name>/thaijp-tts-ckpt'."""
    token = token or get_secret("HF_TOKEN")
    if not token:
        print("⚠️ ไม่มี HF_TOKEN → ไม่สำรอง checkpoint ขึ้น Hugging Face (ถ้า session หลุด งานจะหาย) — "
              "ใส่ที่ Add-ons > Secrets แล้วเปิดใช้กับ notebook นี้", flush=True)
        return None
    if not repo_id:
        who = api
        if who is None:
            from huggingface_hub import HfApi

            who = HfApi(token=token)
        repo_id = f"{who.whoami()['name']}/{DEFAULT_REPO_NAME}"
    return HfBackup(repo_id, models_dir, assets_dir, token=token, api=api,
                    download=getattr(api, "download", None), **kw)
