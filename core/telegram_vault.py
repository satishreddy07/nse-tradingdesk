"""
========================================================================================
Telegram Cloud Master Vault (`core/telegram_vault.py`)
========================================================================================
Backs up and restores the SINGLE MASTER FILE (`data/swing_trades.db`) containing:
1. All Active Positions & Closed Trade History (with full YYYY-MM-DD HH:MM:SS IST timestamps)
2. All 800+ Stocks' 2-Year Historical OHLCV Bars (`ohlcv_master_store`)
3. All Screened Signals, Scan Logs & System Snapshots (`latest_signals`, `system_kv_store`)

Whenever Render restarts or cold-boots, `auto_restore_on_cold_boot()` retrieves the pinned
`#VAULT_INDEX_JSON:...` message from Telegram and restores the entire Single Master File (`swing_trades.db`).
========================================================================================
"""
import os
import gzip
import json
import shutil
import sqlite3
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import requests

from core.config import (
    DB_PATH,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from core.data_quality import format_ist

logger = logging.getLogger("TelegramVault")


class TelegramVaultManager:
    """
    Manages zero-data-loss cloud persistence of the Single Master File (`data/swing_trades.db`)
    to Telegram Cloud (`quant_master_vault.db.gz`).
    """

    def __init__(self):
        self._lock = threading.RLock()

    def get_credentials(self) -> Dict[str, Any]:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN).strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID).strip()
        enabled = bool(bot_token and chat_id and ":" in bot_token)

        masked_token = f"{bot_token[:6]}...{bot_token[-4:]}" if len(bot_token) > 12 else "NOT_SET"
        masked_chat = f"{chat_id[:3]}...{chat_id[-3:]}" if len(chat_id) > 5 else chat_id

        return {
            "enabled": enabled,
            "bot_token": bot_token,
            "chat_id": chat_id,
            "token_masked": masked_token,
            "chat_id_masked": masked_chat
        }

    def get_manifest(self) -> Dict[str, Any]:
        from execution.state_engine import state_engine
        default_manifest = {
            "last_master_backup": None,
            "history": []
        }
        m = state_engine.get_kv("vault_manifest", default_manifest)
        if not isinstance(m, dict):
            return default_manifest
        return m

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        from execution.state_engine import state_engine
        state_engine.set_kv("vault_manifest", manifest)

    def _delete_telegram_message(self, creds: Dict[str, Any], message_id: Optional[int]) -> None:
        if not message_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{creds['bot_token']}/deleteMessage",
                data={"chat_id": creds["chat_id"], "message_id": int(message_id)},
                timeout=8
            )
        except Exception:
            pass

    def _get_pinned_message_info(self, creds: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the currently pinned message object from Telegram (`getChat`)."""
        try:
            chat_resp = requests.get(
                f"https://api.telegram.org/bot{creds['bot_token']}/getChat",
                params={"chat_id": creds["chat_id"]},
                timeout=12
            ).json()
            if chat_resp.get("ok"):
                return chat_resp.get("result", {}).get("pinned_message", {}) or {}
        except Exception as e:
            logger.debug(f"Error inspecting Telegram pinned message: {e}")
        return {}

    def fetch_cloud_index_from_telegram(self) -> Optional[Dict[str, Any]]:
        """
        Queries Telegram's live pinned `quant_master_vault.db.gz` message (`getChat` -> `pinned_message`)
        directly so Telegram acts as the authoritative cloud source of truth across restarts.
        """
        creds = self.get_credentials()
        if not creds["enabled"]:
            return None

        pinned = self._get_pinned_message_info(creds)
        if not pinned:
            return None

        doc = pinned.get("document") or {}
        doc_fid = doc.get("file_id")
        text = pinned.get("caption") or pinned.get("text") or ""

        parsed: Dict[str, Any] = {}
        if "#VAULT_INDEX_JSON:" in text:
            try:
                json_part = text.split("#VAULT_INDEX_JSON:", 1)[1].strip().split("\n")[0]
                parsed = json.loads(json_part)
            except Exception:
                parsed = {}

        if doc_fid:
            parsed["master_file_id"] = doc_fid
            parsed["db_file_id"] = doc_fid
            parsed["message_id"] = pinned.get("message_id")
            return parsed

        if parsed.get("master_file_id") or parsed.get("db_file_id"):
            parsed["message_id"] = pinned.get("message_id")
            return parsed

        return None

    def discover_cloud_master_file_id(self) -> Optional[str]:
        """
        Discovers the latest `master_file_id` from Telegram's live pinned message FIRST,
        falling back to the local DB manifest only if Telegram is unreachable.
        """
        cloud_idx = self.fetch_cloud_index_from_telegram()
        if cloud_idx:
            fid = cloud_idx.get("master_file_id") or cloud_idx.get("db_file_id")
            if fid:
                return fid

        manifest = self.get_manifest()
        last_b = manifest.get("last_master_backup") or manifest.get("last_db_backup") or {}
        if last_b.get("file_id"):
            return last_b["file_id"]
        return None

    # --------------------------------------------------------------------------
    # SINGLE MASTER FILE BACKUP & RESTORE (IN-PLACE UPDATE ON TELEGRAM)
    # --------------------------------------------------------------------------
    def backup_master_file_to_cloud(self, custom_note: str = "Auto Master File Sync") -> Dict[str, Any]:
        """
        Creates an ACID-safe SQLite snapshot of the Single Master File (`data/swing_trades.db`),
        compresses it with gzip (`quant_master_vault.db.gz`), and UPDATES THE EXISTING PINNED
        MASTER FILE IN-PLACE in Telegram using `editMessageMedia` (or replaces & deletes the old
        message if >48h old), ensuring ONLY ONE `quant_master_vault.db.gz` file ever exists in Telegram!
        """
        from execution.state_engine import state_engine

        creds = self.get_credentials()
        if not creds["enabled"]:
            return {"status": "SKIPPED", "message": "Telegram Vault not enabled."}

        if not DB_PATH.exists():
            return {"status": "ERROR", "message": "Master database file does not exist yet."}

        with self._lock:
            tmp_db = None
            tmp_gz = None
            try:
                ohlcv_stats = state_engine.get_ohlcv_stats()
                stocks_cnt = ohlcv_stats.get("stocks_count", 0)
                latest_bar = ohlcv_stats.get("latest_bar_date", "--")
                active_cnt = len(state_engine.get_active_positions())
                closed_cnt = len(state_engine.get_trade_history(limit=5000))
                ts_str = format_ist()

                # Pre-update manifest timestamp inside SQLite before taking snapshot
                manifest = self.get_manifest()
                prev_msg_id = (manifest.get("last_master_backup") or {}).get("message_id")
                pre_record = {
                    "status": "SUCCESS",
                    "filename": "quant_master_vault.db.gz",
                    "stocks_count": stocks_cnt,
                    "latest_bar_date": latest_bar,
                    "active_trades": active_cnt,
                    "closed_trades": closed_cnt,
                    "note": custom_note,
                    "timestamp": ts_str
                }
                manifest["last_master_backup"] = {**(manifest.get("last_master_backup") or {}), **pre_record}
                self._save_manifest(manifest)

                with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f_db:
                    tmp_db = Path(f_db.name)

                # Online ACID backup of the Single Master File
                src_conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
                dst_conn = sqlite3.connect(str(tmp_db))
                with dst_conn:
                    src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()

                tmp_gz = Path(str(tmp_db) + ".gz")
                with open(tmp_db, "rb") as f_in, gzip.open(tmp_gz, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)

                size_kb = round(tmp_gz.stat().st_size / 1024.0, 1)

                index_meta = {
                    "stocks_count": stocks_cnt,
                    "latest_bar_date": latest_bar,
                    "active_trades": active_cnt,
                    "closed_trades": closed_cnt,
                    "note": custom_note,
                    "updated_at": ts_str
                }

                caption = (
                    f"📌 <b>QUANT SINGLE MASTER VAULT (DO NOT DELETE)</b>\n"
                    f"📝 <b>Last Update:</b> {custom_note}\n"
                    f"📊 <b>Trades:</b> {active_cnt} Active | {closed_cnt} Closed\n"
                    f"📦 <b>OHLCV Universe:</b> {stocks_cnt} Stocks (Latest Bar: {latest_bar})\n"
                    f"💾 <b>Master Size:</b> {size_kb} KB | 🕒 <b>Synced:</b> {ts_str}\n"
                    f"<code>#VAULT_INDEX_JSON:{json.dumps(index_meta)}</code>"
                )

                # Check if we already have a pinned document message in Telegram to update in-place!
                pinned = self._get_pinned_message_info(creds)
                pinned_msg_id = pinned.get("message_id") or prev_msg_id
                pinned_has_doc = bool(pinned.get("document"))

                resp = None
                updated_in_place = False

                if pinned_msg_id and pinned_has_doc:
                    media_payload = json.dumps({
                        "type": "document",
                        "media": "attach://document",
                        "caption": caption,
                        "parse_mode": "HTML"
                    })
                    with open(tmp_gz, "rb") as doc_file:
                        edit_resp = requests.post(
                            f"https://api.telegram.org/bot{creds['bot_token']}/editMessageMedia",
                            data={
                                "chat_id": creds["chat_id"],
                                "message_id": int(pinned_msg_id),
                                "media": media_payload
                            },
                            files={"document": ("quant_master_vault.db.gz", doc_file, "application/gzip")},
                            timeout=90
                        ).json()
                    if edit_resp.get("ok"):
                        resp = edit_resp
                        updated_in_place = True

                # If no pinned document existed yet (or edit window expired), send once, pin it, and delete old message
                if not updated_in_place:
                    with open(tmp_gz, "rb") as doc_file:
                        resp = requests.post(
                            f"https://api.telegram.org/bot{creds['bot_token']}/sendDocument",
                            data={"chat_id": creds["chat_id"], "caption": caption, "parse_mode": "HTML"},
                            files={"document": ("quant_master_vault.db.gz", doc_file, "application/gzip")},
                            timeout=90
                        ).json()

                    if not resp or not resp.get("ok"):
                        err = (resp or {}).get("description", "Unknown Telegram error")
                        logger.warning(f"Telegram Master Vault upload failed: {err}")
                        return {"status": "ERROR", "message": err}

                    new_msg_id = resp["result"]["message_id"]
                    requests.post(
                        f"https://api.telegram.org/bot{creds['bot_token']}/pinChatMessage",
                        data={"chat_id": creds["chat_id"], "message_id": new_msg_id, "disable_notification": True},
                        timeout=10
                    )
                    # Delete previous message so only ONE master file remains in Telegram
                    if pinned_msg_id and int(pinned_msg_id) != int(new_msg_id):
                        self._delete_telegram_message(creds, int(pinned_msg_id))
                    if prev_msg_id and int(prev_msg_id) not in (int(new_msg_id), int(pinned_msg_id or 0)):
                        self._delete_telegram_message(creds, int(prev_msg_id))

                file_id = resp["result"]["document"]["file_id"]
                msg_id = resp["result"]["message_id"]
                record = {
                    "status": "SUCCESS",
                    "mode": "UPDATED_IN_PLACE" if updated_in_place else "CREATED_AND_PINNED",
                    "message_id": msg_id,
                    "file_id": file_id,
                    "filename": "quant_master_vault.db.gz",
                    "stocks_count": stocks_cnt,
                    "latest_bar_date": latest_bar,
                    "active_trades": active_cnt,
                    "closed_trades": closed_cnt,
                    "size_kb": size_kb,
                    "note": custom_note,
                    "timestamp": ts_str
                }

                manifest = self.get_manifest()
                manifest["last_master_backup"] = record
                manifest["last_db_backup"] = record
                manifest["last_ohlcv_backup"] = record
                hist = manifest.get("history", [])
                hist.insert(0, {"type": "MASTER_VAULT", **record})
                manifest["history"] = hist[:25]
                self._save_manifest(manifest)

                logger.info(
                    f"☁️ Single Master File {'UPDATED IN-PLACE' if updated_in_place else 'PINNED'} in Telegram Cloud "
                    f"(msg_id={msg_id}, {stocks_cnt} stocks, {active_cnt} open trades, {closed_cnt} closed trades, {size_kb} KB)"
                )
                return record

            except Exception as e:
                logger.error(f"Master Vault backup exception: {e}")
                return {"status": "ERROR", "message": str(e)}
            finally:
                if tmp_db and tmp_db.exists():
                    tmp_db.unlink(missing_ok=True)
                if tmp_gz and tmp_gz.exists():
                    tmp_gz.unlink(missing_ok=True)

    # Aliases so existing callers all write to the Single Master File backup
    def backup_db_to_cloud(self, custom_note: str = "Trade Update Sync") -> Dict[str, Any]:
        return self.backup_master_file_to_cloud(custom_note=custom_note)

    def backup_ohlcv_to_cloud(self, custom_note: str = "OHLCV Gap-Fill Sync") -> Dict[str, Any]:
        return self.backup_master_file_to_cloud(custom_note=custom_note)

    def restore_master_file_from_cloud(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Restores the Single Master File (`data/swing_trades.db`) from Telegram Cloud."""
        from execution.state_engine import state_engine

        creds = self.get_credentials()
        if not creds["enabled"]:
            return {"status": "SKIPPED", "message": "Telegram Vault not enabled."}

        target_file_id = file_id or self.discover_cloud_master_file_id()
        if not target_file_id:
            return {"status": "SKIPPED", "message": "No cloud Master File ID found in Telegram."}

        try:
            meta = requests.get(
                f"https://api.telegram.org/bot{creds['bot_token']}/getFile",
                params={"file_id": target_file_id},
                timeout=15
            ).json()
            if not meta.get("ok"):
                return {"status": "ERROR", "message": f"getFile failed: {meta.get('description')}"}

            file_path = meta["result"]["file_path"]
            dl = requests.get(f"https://api.telegram.org/file/bot{creds['bot_token']}/{file_path}", timeout=90)
            if dl.status_code != 200:
                return {"status": "ERROR", "message": f"HTTP {dl.status_code} downloading Master File"}

            raw_bytes = dl.content
            if raw_bytes[:2] == b"\x1f\x8b":
                raw_bytes = gzip.decompress(raw_bytes)

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(raw_bytes)

            conn = sqlite3.connect(str(tmp_path))
            chk = conn.execute("PRAGMA integrity_check;").fetchone()
            conn.close()
            if not chk or chk[0] != "ok":
                tmp_path.unlink(missing_ok=True)
                return {"status": "ERROR", "message": "Restored file failed SQLite integrity check."}

            with self._lock:
                shutil.copy2(tmp_path, DB_PATH)
                state_engine._ohlcv_mem_cache.clear()
                state_engine._init_db()
            tmp_path.unlink(missing_ok=True)

            ohlcv_stats = state_engine.get_ohlcv_stats()
            active_cnt = len(state_engine.get_active_positions())
            closed_cnt = len(state_engine.get_trade_history(limit=5000))
            size_kb = round(len(raw_bytes) / 1024.0, 1)
            now_ts = format_ist()

            # Sync local manifest with the restored Telegram file_id so local DB knows it matches Telegram
            manifest = self.get_manifest()
            restored_rec = {
                "status": "SUCCESS",
                "file_id": target_file_id,
                "filename": "quant_master_vault.db.gz",
                "stocks_count": ohlcv_stats.get("stocks_count", 0),
                "latest_bar_date": ohlcv_stats.get("latest_bar_date", "--"),
                "active_trades": active_cnt,
                "closed_trades": closed_cnt,
                "size_kb": size_kb,
                "note": "Restored from Telegram Cloud Authority",
                "timestamp": now_ts
            }
            manifest["last_master_backup"] = restored_rec
            manifest["last_db_backup"] = restored_rec
            manifest["last_ohlcv_backup"] = restored_rec
            self._save_manifest(manifest)

            logger.info(
                f"✅ Restored Single Master File from Telegram Cloud "
                f"({size_kb} KB, {ohlcv_stats.get('stocks_count', 0)} OHLCV stocks, "
                f"{active_cnt} active trades, {closed_cnt} closed trades)"
            )
            return {
                "status": "SUCCESS",
                "size_kb": size_kb,
                "stocks_count": ohlcv_stats.get("stocks_count", 0),
                "latest_bar_date": ohlcv_stats.get("latest_bar_date", "--"),
                "active_trades": active_cnt,
                "closed_trades": closed_cnt,
                "file_id": target_file_id,
                "restored_at": now_ts
            }
        except Exception as e:
            logger.error(f"Restore Master File from cloud failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    def restore_db_from_cloud(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        return self.restore_master_file_from_cloud(file_id=file_id)

    def restore_ohlcv_from_cloud(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        return {"status": "SUCCESS", "message": "Included in Single Master File restore."}

    def auto_restore_on_cold_boot(self) -> Dict[str, Any]:
        """
        Called automatically on server startup (e.g. Render container boot):
        Always queries Telegram's pinned `#VAULT_INDEX_JSON` first so Telegram acts
        as the 100% authoritative data manager. If Telegram has a newer `master_file_id`
        or `updated_at` timestamp than the local working copy (or local DB is missing),
        automatically downloads and restores the Single Master File from Telegram Cloud.
        """
        from execution.state_engine import state_engine

        creds = self.get_credentials()
        if not creds["enabled"]:
            return {"status": "DISABLED"}

        cloud_idx = self.fetch_cloud_index_from_telegram()
        cloud_fid = (cloud_idx or {}).get("master_file_id") or (cloud_idx or {}).get("db_file_id")
        cloud_ts = (cloud_idx or {}).get("updated_at", "")

        stats = state_engine.get_ohlcv_stats()
        local_manifest = self.get_manifest()
        local_last = local_manifest.get("last_master_backup") or local_manifest.get("last_db_backup") or {}
        local_fid = local_last.get("file_id", "")
        local_ts = local_last.get("timestamp", "")

        needs_restore = False
        reason = ""

        if not DB_PATH.exists() or stats.get("stocks_count", 0) < 50:
            needs_restore = True
            reason = "Local Master File missing or empty"
        elif cloud_fid and cloud_fid != local_fid:
            # If Telegram has a newer timestamp or local has no recorded file_id
            if not local_ts or (cloud_ts and cloud_ts[:19] >= local_ts[:19]):
                needs_restore = True
                reason = f"Telegram Cloud has newer Master Vault (Cloud: {cloud_ts} vs Local: {local_ts or 'None'})"

        if not needs_restore:
            return {
                "status": "IN_SYNC_WITH_TELEGRAM",
                "stocks_count": stats.get("stocks_count", 0),
                "file_id": local_fid or cloud_fid
            }

        logger.info(f"🔄 Telegram Cloud Sync on Boot ({reason}): Restoring Single Master File from Telegram...")
        return self.restore_master_file_from_cloud(file_id=cloud_fid)

    def send_alert_and_sync(self, title: str, message: str, sync_db: bool = True, sync_ohlcv: bool = False):
        """Dispatches a Telegram alert and immediately syncs the Single Master File in a background thread."""
        creds = self.get_credentials()
        if not creds["enabled"]:
            return

        def _worker():
            try:
                html = (
                    f"🔔 <b>{title}</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{message}\n\n"
                    f"🕒 <i>{format_ist()}</i>"
                )
                requests.post(
                    f"https://api.telegram.org/bot{creds['bot_token']}/sendMessage",
                    data={"chat_id": creds["chat_id"], "text": html, "parse_mode": "HTML"},
                    timeout=12
                )
            except Exception as e:
                logger.debug(f"Telegram alert error: {e}")

            if sync_db or sync_ohlcv:
                self.backup_master_file_to_cloud(custom_note=title)

        threading.Thread(target=_worker, daemon=True).start()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status of the Single Master File & Telegram Cloud Vault."""
        from execution.state_engine import state_engine

        creds = self.get_credentials()
        manifest = self.get_manifest()
        last_m = manifest.get("last_master_backup") or manifest.get("last_db_backup") or {}
        ohlcv_stats = state_engine.get_ohlcv_stats()
        stocks_cnt = ohlcv_stats.get("stocks_count", 0)
        latest_date = ohlcv_stats.get("latest_bar_date", "--")
        db_size_mb = round(DB_PATH.stat().st_size / (1024 * 1024), 2) if DB_PATH.exists() else 0.0

        return {
            "status": "ONLINE" if creds["enabled"] else "DISABLED",
            "enabled": creds["enabled"],
            "master_file_path": "data/swing_trades.db",
            "master_file_size_mb": db_size_mb,
            "token_masked": creds["token_masked"],
            "chat_id": creds["chat_id_masked"],
            "local_ohlcv_stocks": stocks_cnt,
            "local_ohlcv_files": stocks_cnt,
            "latest_local_ohlcv_date": latest_date,
            "latest_bar_date": latest_date,
            "last_db_backup_ist": last_m.get("timestamp", "Synced"),
            "last_master_backup": last_m,
            "last_db_backup": last_m,
            "last_ohlcv_backup": last_m,
            "history": manifest.get("history", [])[:10]
        }

    def get_vault_status(self) -> Dict[str, Any]:
        return self.get_status()


# Singleton Instance
telegram_vault = TelegramVaultManager()
