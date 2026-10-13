"""
========================================================================================
Telegram Cloud Storage Vault — Trade DB + Historical OHLCV Cache Persistence
========================================================================================
Ensures 100% zero data loss across Render container restarts / spin-downs by syncing:
1. SQLite Trade Database (`data/swing_trades.db` -> `swing_trades_latest.db.gz`)
   - Active Positions, Closed Trade History, Signals, Scan Execution Logs
2. Historical 2-Year OHLCV Cache (`data/historical_data/*.pkl` -> `ohlcv_cache_vault.pkl.gz`)
   - Compact float32 compressed bundle (~2-4 MB for 800+ stocks x 500 daily bars)
   - Supports Smart Incremental Gap-Filling: on cold boot or scan, restores cached OHLCV
     from Telegram Cloud and fetches only the missing delta bars up to today's date.
3. Cloud Pinned Index Pointer (`pinChatMessage` + `getChat`):
   - Stores `db_file_id` and `ohlcv_file_id` in the Telegram chat's pinned message so
     cold-boot containers can always discover and download the latest snapshots even if
     local disk is wiped and `getUpdates` 24h window has expired.
========================================================================================
"""

import os
import io
import json
import time
import gzip
import shutil
import pickle
import sqlite3
import hashlib
import logging
import tempfile
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd
import numpy as np

from core.data_quality import IST, now_ist, format_ist
from core.config import (
    DATA_DIR, HISTORICAL_DATA_DIR, DB_PATH,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    VAULT_MANIFEST_PATH, VAULT_CONFIG_PATH
)

logger = logging.getLogger("TelegramVault")


class TelegramVaultManager:
    """
    Manages two-way cloud synchronization of both SQLite Trade State and
    Historical 2-Year OHLCV DataFrames via Telegram Bot API.
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # --------------------------------------------------------------------------
    # CREDENTIALS & HANDSHAKE
    # --------------------------------------------------------------------------
    def get_credentials(self) -> Dict[str, Any]:
        """Retrieves Telegram Bot Token & Chat ID from env, config file, or default."""
        cfg = {}
        if VAULT_CONFIG_PATH.exists():
            try:
                with open(VAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

        bot_token = (
            os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
            or str(cfg.get("bot_token", "")).strip()
            or TELEGRAM_BOT_TOKEN
        )
        chat_id = (
            os.environ.get("TELEGRAM_CHAT_ID", "").strip()
            or str(cfg.get("chat_id", "")).strip()
            or TELEGRAM_CHAT_ID
        )
        enabled = cfg.get("enabled", True)

        return {
            "bot_token": bot_token,
            "chat_id": chat_id,
            "enabled": bool(enabled and bot_token and chat_id),
            "token_masked": f"{bot_token[:6]}...{bot_token[-4:]}" if len(bot_token) > 12 else ("Set" if bot_token else "Not Set"),
            "chat_id_masked": str(chat_id)
        }

    def save_credentials(self, bot_token: str, chat_id: str, enabled: bool = True) -> Dict[str, Any]:
        """Saves custom Telegram credentials to data/vault_config.json."""
        payload = {
            "bot_token": str(bot_token).strip(),
            "chat_id": str(chat_id).strip(),
            "enabled": bool(enabled),
            "updated_at": format_ist()
        }
        with open(VAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return {"status": "SUCCESS", "credentials": self.get_credentials()}

    # --------------------------------------------------------------------------
    # MANIFEST & CLOUD PINNED INDEX DISCOVERY
    # --------------------------------------------------------------------------
    def get_manifest(self) -> Dict[str, Any]:
        """Reads local vault_manifest.json or returns empty structure."""
        default_manifest = {
            "last_db_backup": None,
            "last_ohlcv_backup": None,
            "history": []
        }
        if VAULT_MANIFEST_PATH.exists():
            try:
                with open(VAULT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    default_manifest.update(data)
            except Exception:
                pass
        return default_manifest

    def _save_manifest(self, manifest: Dict[str, Any]):
        try:
            with open(VAULT_MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write vault_manifest.json: {e}")

    def _pin_cloud_index(self):
        """
        Sends and pins a compact Cloud Index message in the Telegram chat containing
        the latest `db_file_id` and `ohlcv_file_id` JSON metadata.
        On Render cold-boot, `getChat(chat_id)` reads `pinned_message` to recover both IDs!
        """
        creds = self.get_credentials()
        if not creds["enabled"]:
            return

        manifest = self.get_manifest()
        db_rec = manifest.get("last_db_backup") or {}
        ohlcv_rec = manifest.get("last_ohlcv_backup") or {}

        index_payload = {
            "db_file_id": db_rec.get("file_id"),
            "db_timestamp": db_rec.get("timestamp"),
            "ohlcv_file_id": ohlcv_rec.get("file_id"),
            "ohlcv_timestamp": ohlcv_rec.get("timestamp"),
            "ohlcv_stocks_count": ohlcv_rec.get("stocks_count", 0),
            "ohlcv_latest_date": ohlcv_rec.get("latest_bar_date", ""),
            "updated_at": format_ist()
        }
        index_json_str = json.dumps(index_payload)

        text_html = (
            "📌 <b>QUANT TRADING DESK — CLOUD VAULT MASTER INDEX</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🗄️ <b>Trade DB Snapshot:</b> <code>{db_rec.get('filename', 'None')}</code> ({db_rec.get('timestamp', '--')})\n"
            f"📊 <b>OHLCV Cloud Cache:</b> <code>{ohlcv_rec.get('stocks_count', 0)} stocks</code> (Up to {ohlcv_rec.get('latest_bar_date', '--')})\n"
            f"🕒 <b>Synced At:</b> {index_payload['updated_at']}\n\n"
            f"<code>#VAULT_INDEX_JSON:{index_json_str}</code>"
        )

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{creds['bot_token']}/sendMessage",
                data={"chat_id": creds["chat_id"], "text": text_html, "parse_mode": "HTML"},
                timeout=12
            ).json()
            if resp.get("ok"):
                msg_id = resp["result"]["message_id"]
                requests.post(
                    f"https://api.telegram.org/bot{creds['bot_token']}/pinChatMessage",
                    data={"chat_id": creds["chat_id"], "message_id": msg_id, "disable_notification": True},
                    timeout=10
                )
        except Exception as e:
            logger.debug(f"Cloud index pin notice: {e}")

    def discover_cloud_file_ids(self) -> Dict[str, Optional[str]]:
        """
        Discovers the latest `db_file_id` and `ohlcv_file_id` from:
        1. Telegram `getChat` pinned message (`#VAULT_INDEX_JSON:...`)
        2. Local `vault_manifest.json`
        3. Telegram `getUpdates` recent documents
        """
        manifest = self.get_manifest()
        db_id = (manifest.get("last_db_backup") or {}).get("file_id")
        ohlcv_id = (manifest.get("last_ohlcv_backup") or {}).get("file_id")

        creds = self.get_credentials()
        if not creds["bot_token"] or not creds["chat_id"]:
            return {"db_file_id": db_id, "ohlcv_file_id": ohlcv_id}

        # Check Telegram getChat pinned_message first (survives container wipes & 24h getUpdates expiry)
        try:
            chat_res = requests.get(
                f"https://api.telegram.org/bot{creds['bot_token']}/getChat",
                params={"chat_id": creds["chat_id"]},
                timeout=12
            ).json()
            if chat_res.get("ok"):
                pinned = chat_res.get("result", {}).get("pinned_message") or {}
                pinned_text = pinned.get("text") or pinned.get("caption") or ""
                if "#VAULT_INDEX_JSON:" in pinned_text:
                    json_part = pinned_text.split("#VAULT_INDEX_JSON:", 1)[1].strip()
                    idx_data = json.loads(json_part)
                    if idx_data.get("db_file_id"):
                        db_id = idx_data["db_file_id"]
                    if idx_data.get("ohlcv_file_id"):
                        ohlcv_id = idx_data["ohlcv_file_id"]
        except Exception as e:
            logger.debug(f"getChat pinned index lookup note: {e}")

        # Fallback to getUpdates if either is still missing
        if not db_id or not ohlcv_id:
            try:
                upd_res = requests.get(
                    f"https://api.telegram.org/bot{creds['bot_token']}/getUpdates",
                    timeout=12
                ).json()
                if upd_res.get("ok"):
                    for u in reversed(upd_res.get("result", [])):
                        msg = u.get("message") or u.get("channel_post") or {}
                        doc = msg.get("document") or {}
                        fname = doc.get("file_name", "")
                        if not db_id and fname.startswith("swing_trades") and (fname.endswith(".db") or fname.endswith(".db.gz")):
                            db_id = doc.get("file_id")
                        if not ohlcv_id and fname.startswith("ohlcv_cache_vault") and fname.endswith(".pkl.gz"):
                            ohlcv_id = doc.get("file_id")
            except Exception as e:
                logger.debug(f"getUpdates lookup note: {e}")

        return {"db_file_id": db_id, "ohlcv_file_id": ohlcv_id}

    # --------------------------------------------------------------------------
    # 1. SQLITE TRADE DATABASE BACKUP & RESTORE
    # --------------------------------------------------------------------------
    def backup_db_to_cloud(self, custom_note: str = "Automated State Sync") -> Dict[str, Any]:
        """Creates an ACID SQLite snapshot, compresses with gzip, and uploads to Telegram Cloud."""
        creds = self.get_credentials()
        if not creds["enabled"]:
            return {"status": "SKIPPED", "message": "Telegram Vault not enabled."}

        target_db = Path(DB_PATH)
        if not target_db.exists():
            return {"status": "ERROR", "message": "Database file does not exist yet."}

        now = now_ist()
        ts_slug = now.strftime("%Y%m%d_%H%M%S")
        date_display = format_ist(now)
        filename = f"swing_trades_{ts_slug}.db.gz"

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db_path = Path(tmp.name)

        try:
            with self._lock:
                src_conn = sqlite3.connect(str(target_db), timeout=30.0)
                dst_conn = sqlite3.connect(str(tmp_db_path))
                src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()

            with open(tmp_db_path, "rb") as f:
                raw_bytes = f.read()

            gz_bytes = gzip.compress(raw_bytes, compresslevel=6)
            sha256_hash = hashlib.sha256(gz_bytes).hexdigest()
            size_kb = round(len(gz_bytes) / 1024.0, 1)

            # Gather quick counts for caption
            p1_open, p2_open, closed_cnt = 0, 0, 0
            try:
                conn = sqlite3.connect(str(target_db))
                cur = conn.cursor()
                for pt, cnt in cur.execute("SELECT portfolio_type, COUNT(*) FROM active_positions GROUP BY portfolio_type").fetchall():
                    if "NIFTY" in str(pt).upper():
                        p2_open += cnt
                    else:
                        p1_open += cnt
                row_c = cur.execute("SELECT COUNT(*) FROM trade_history").fetchone()
                closed_cnt = row_c[0] if row_c else 0
                conn.close()
            except Exception:
                pass

            caption_html = (
                "💾 <b>QUANT DESK — TRADE DATABASE CLOUD SNAPSHOT</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 <b>Timestamp:</b> {date_display}\n"
                f"📦 <b>File:</b> <code>{filename}</code> ({size_kb} KB)\n"
                f"💼 <b>P1 Active:</b> {p1_open} | <b>P2 Active:</b> {p2_open} | <b>Closed Trades:</b> {closed_cnt}\n"
                f"📝 <b>Trigger:</b> {custom_note}"
            )

            files = {"document": (filename, io.BytesIO(gz_bytes), "application/gzip")}
            data = {"chat_id": creds["chat_id"], "caption": caption_html, "parse_mode": "HTML"}
            resp = requests.post(
                f"https://api.telegram.org/bot{creds['bot_token']}/sendDocument",
                data=data,
                files=files,
                timeout=60
            ).json()

            if not resp.get("ok"):
                return {"status": "ERROR", "message": f"Telegram upload error: {resp.get('description')}"}

            doc_obj = resp.get("result", {}).get("document", {})
            file_id = doc_obj.get("file_id")
            msg_id = resp.get("result", {}).get("message_id")

            record = {
                "type": "DB",
                "timestamp": date_display,
                "filename": filename,
                "file_id": file_id,
                "message_id": msg_id,
                "size_kb": size_kb,
                "checksum": sha256_hash,
                "note": custom_note
            }
            manifest = self.get_manifest()
            manifest["last_db_backup"] = record
            history = manifest.get("history", [])
            history.insert(0, record)
            manifest["history"] = history[:30]
            self._save_manifest(manifest)
            self._pin_cloud_index()

            logger.info(f"☁️ Uploaded SQLite Trade DB backup to Telegram Cloud: {filename} ({size_kb} KB)")
            return {"status": "SUCCESS", "file_id": file_id, "filename": filename, "size_kb": size_kb, "timestamp": date_display}
        except Exception as e:
            logger.error(f"DB cloud backup error: {e}")
            return {"status": "ERROR", "message": str(e)}
        finally:
            if tmp_db_path.exists():
                try:
                    tmp_db_path.unlink()
                except Exception:
                    pass

    def restore_db_from_cloud(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Downloads and verifies the latest SQLite Trade DB from Telegram Cloud."""
        creds = self.get_credentials()
        if not creds["bot_token"]:
            return {"status": "ERROR", "message": "Telegram Bot Token not configured."}

        target_file_id = file_id or self.discover_cloud_file_ids().get("db_file_id")
        if not target_file_id:
            return {"status": "SKIPPED", "message": "No Trade DB backup found in Telegram Cloud."}

        try:
            meta = requests.get(
                f"https://api.telegram.org/bot{creds['bot_token']}/getFile",
                params={"file_id": target_file_id},
                timeout=15
            ).json()
            if not meta.get("ok"):
                return {"status": "ERROR", "message": f"getFile failed: {meta.get('description')}"}

            file_path = meta["result"]["file_path"]
            dl = requests.get(f"https://api.telegram.org/file/bot{creds['bot_token']}/{file_path}", timeout=60)
            if dl.status_code != 200:
                return {"status": "ERROR", "message": f"HTTP {dl.status_code} downloading DB"}

            raw_bytes = dl.content
            if raw_bytes[:2] == b"\x1f\x8b":
                raw_bytes = gzip.decompress(raw_bytes)

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(raw_bytes)

            # Integrity check
            conn = sqlite3.connect(str(tmp_path))
            chk = conn.execute("PRAGMA integrity_check;").fetchone()
            conn.close()
            if not chk or chk[0] != "ok":
                tmp_path.unlink(missing_ok=True)
                return {"status": "ERROR", "message": "Restored DB failed SQLite integrity check."}

            with self._lock:
                shutil.copy2(tmp_path, DB_PATH)
            tmp_path.unlink(missing_ok=True)

            size_kb = round(len(raw_bytes) / 1024.0, 1)
            logger.info(f"✅ Restored Trade DB from Telegram Cloud ({size_kb} KB)")
            return {"status": "SUCCESS", "size_kb": size_kb, "file_id": target_file_id, "restored_at": format_ist()}
        except Exception as e:
            logger.error(f"Restore DB from cloud failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    # --------------------------------------------------------------------------
    # 2. HISTORICAL OHLCV CACHE ARCHIVE BACKUP & RESTORE
    # --------------------------------------------------------------------------
    def backup_ohlcv_to_cloud(self, custom_note: str = "OHLCV Gap-Fill Sync") -> Dict[str, Any]:
        """
        Packs all cached ticker DataFrames in `data/historical_data/*.pkl` into a compact
        gzipped bundle (`ohlcv_cache_vault.pkl.gz`) and uploads it to Telegram Cloud.
        Uses float32 for OHLC and int64 for Volume, keeping 800 stocks x 500 bars under ~3.5 MB!
        """
        creds = self.get_credentials()
        if not creds["enabled"]:
            return {"status": "SKIPPED", "message": "Telegram Vault not enabled."}

        pkl_files = list(HISTORICAL_DATA_DIR.glob("*.pkl"))
        if not pkl_files:
            return {"status": "SKIPPED", "message": "No local OHLCV pickle files to back up."}

        bundle: Dict[str, pd.DataFrame] = {}
        latest_date_seen = ""

        for fpath in pkl_files:
            sym = fpath.stem.upper()
            try:
                df = pd.read_pickle(fpath)
                if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                    df_clean = df[["Open", "High", "Low", "Close", "Volume"]].tail(500).copy()
                    for col in ["Open", "High", "Low", "Close"]:
                        df_clean[col] = df_clean[col].astype("float32")
                    df_clean["Volume"] = df_clean["Volume"].fillna(0).astype("int64")
                    bundle[sym] = df_clean
                    bar_dt = str(df_clean.index[-1])[:10]
                    if bar_dt > latest_date_seen:
                        latest_date_seen = bar_dt
            except Exception:
                continue

        if not bundle:
            return {"status": "SKIPPED", "message": "No valid OHLCV DataFrames found."}

        now = now_ist()
        date_display = format_ist(now)
        filename = "ohlcv_cache_vault.pkl.gz"

        try:
            raw_pkl = pickle.dumps(
                {
                    "created_at": date_display,
                    "latest_bar_date": latest_date_seen,
                    "stocks_count": len(bundle),
                    "ohlcv": bundle
                },
                protocol=pickle.HIGHEST_PROTOCOL
            )
            gz_bytes = gzip.compress(raw_pkl, compresslevel=6)
            size_kb = round(len(gz_bytes) / 1024.0, 1)
            size_mb = round(size_kb / 1024.0, 2)

            caption_html = (
                "📊 <b>QUANT DESK — HISTORICAL OHLCV CLOUD VAULT</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 <b>Synced At:</b> {date_display}\n"
                f"📈 <b>Stocks Cached:</b> <code>{len(bundle)} NSE Tickers</code>\n"
                f"🗓️ <b>Latest Bar Date:</b> <code>{latest_date_seen}</code>\n"
                f"📦 <b>Archive Size:</b> {size_mb} MB ({size_kb} KB gzip)\n"
                f"📝 <b>Note:</b> {custom_note}\n\n"
                "<i>Used for instant cold-boot recovery & incremental gap-filling across Render restarts.</i>"
            )

            files = {"document": (filename, io.BytesIO(gz_bytes), "application/gzip")}
            data = {"chat_id": creds["chat_id"], "caption": caption_html, "parse_mode": "HTML"}
            resp = requests.post(
                f"https://api.telegram.org/bot{creds['bot_token']}/sendDocument",
                data=data,
                files=files,
                timeout=120
            ).json()

            if not resp.get("ok"):
                return {"status": "ERROR", "message": f"Telegram OHLCV upload error: {resp.get('description')}"}

            doc_obj = resp.get("result", {}).get("document", {})
            file_id = doc_obj.get("file_id")
            msg_id = resp.get("result", {}).get("message_id")

            record = {
                "type": "OHLCV",
                "timestamp": date_display,
                "filename": filename,
                "file_id": file_id,
                "message_id": msg_id,
                "stocks_count": len(bundle),
                "latest_bar_date": latest_date_seen,
                "size_kb": size_kb,
                "note": custom_note
            }
            manifest = self.get_manifest()
            manifest["last_ohlcv_backup"] = record
            history = manifest.get("history", [])
            history.insert(0, record)
            manifest["history"] = history[:30]
            self._save_manifest(manifest)
            self._pin_cloud_index()

            logger.info(f"☁️ Uploaded OHLCV Cloud Cache ({len(bundle)} stocks, up to {latest_date_seen}, {size_mb} MB) to Telegram!")
            return {
                "status": "SUCCESS",
                "file_id": file_id,
                "stocks_count": len(bundle),
                "latest_bar_date": latest_date_seen,
                "size_kb": size_kb,
                "timestamp": date_display
            }
        except Exception as e:
            logger.error(f"OHLCV cloud backup failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    def restore_ohlcv_from_cloud(self, file_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Downloads `ohlcv_cache_vault.pkl.gz` from Telegram Cloud and extracts all ticker
        DataFrames into `data/historical_data/{SYM}.pkl`.
        """
        creds = self.get_credentials()
        if not creds["bot_token"]:
            return {"status": "ERROR", "message": "Telegram Bot Token not configured."}

        target_file_id = file_id or self.discover_cloud_file_ids().get("ohlcv_file_id")
        if not target_file_id:
            return {"status": "SKIPPED", "message": "No OHLCV archive found in Telegram Cloud."}

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
                return {"status": "ERROR", "message": f"HTTP {dl.status_code} downloading OHLCV archive"}

            raw_bytes = dl.content
            if raw_bytes[:2] == b"\x1f\x8b":
                raw_bytes = gzip.decompress(raw_bytes)

            payload = pickle.loads(raw_bytes)
            ohlcv_map: Dict[str, pd.DataFrame] = payload.get("ohlcv", {}) if isinstance(payload, dict) else {}
            latest_bar_date = payload.get("latest_bar_date", "")

            HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            restored_count = 0
            for sym, df in ohlcv_map.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    clean_sym = str(sym).replace(".NS", "").replace(".BO", "").strip().upper()
                    pkl_path = HISTORICAL_DATA_DIR / f"{clean_sym}.pkl"
                    df.to_pickle(pkl_path)
                    restored_count += 1

            logger.info(f"✅ Restored {restored_count} stock OHLCV histories from Telegram Cloud (latest bar: {latest_bar_date})!")
            return {
                "status": "SUCCESS",
                "stocks_restored": restored_count,
                "latest_bar_date": latest_bar_date,
                "file_id": target_file_id,
                "restored_at": format_ist()
            }
        except Exception as e:
            logger.error(f"Restore OHLCV from cloud failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    # --------------------------------------------------------------------------
    # 3. COLD-BOOT AUTO-RESTORE (RENDER CONTAINER STARTUP)
    # --------------------------------------------------------------------------
    def auto_restore_on_cold_boot(self) -> Dict[str, Any]:
        """
        Automatically executed when the server boots up (e.g. Render spin-up/redeploy):
        1. If local SQLite DB has 0 trades/signals AND Telegram Cloud has a DB snapshot, restores it.
        2. If local `data/historical_data` has < 50 cached stocks AND Telegram Cloud has an OHLCV bundle,
           restores all cached stocks so incremental gap-filling works immediately!
        """
        results = {"db_restore": "SKIPPED", "ohlcv_restore": "SKIPPED"}
        creds = self.get_credentials()
        if not creds["enabled"]:
            return results

        discovered = self.discover_cloud_file_ids()

        # Check SQLite DB
        db_needs_restore = False
        if not Path(DB_PATH).exists() or Path(DB_PATH).stat().st_size == 0:
            db_needs_restore = True
        else:
            try:
                conn = sqlite3.connect(str(DB_PATH))
                c_pos = conn.execute("SELECT COUNT(*) FROM active_positions").fetchone()[0]
                c_hist = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0]
                c_sig = conn.execute("SELECT COUNT(*) FROM latest_signals").fetchone()[0]
                conn.close()
                if c_pos == 0 and c_hist == 0 and c_sig == 0:
                    db_needs_restore = True
            except Exception:
                db_needs_restore = True

        if db_needs_restore and discovered.get("db_file_id"):
            logger.info("🔄 [ColdBoot] Local DB is empty; restoring Trade DB from Telegram Cloud Vault...")
            results["db_restore"] = self.restore_db_from_cloud(file_id=discovered["db_file_id"])

        # Check OHLCV Cache
        local_pkl_count = len(list(HISTORICAL_DATA_DIR.glob("*.pkl")))
        if local_pkl_count < 50 and discovered.get("ohlcv_file_id"):
            logger.info(f"🔄 [ColdBoot] Local OHLCV cache has only {local_pkl_count} files; restoring OHLCV Vault from Telegram Cloud...")
            results["ohlcv_restore"] = self.restore_ohlcv_from_cloud(file_id=discovered["ohlcv_file_id"])

        return results

    # --------------------------------------------------------------------------
    # 4. ASYNC ALERTS & AUTOMATIC BACKGROUND CLOUD SYNC
    # --------------------------------------------------------------------------
    def send_alert_and_sync(self, title: str, message: str, sync_db: bool = True, sync_ohlcv: bool = False):
        """Dispatches a Telegram message and optionally backs up DB/OHLCV in a background thread."""
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

            if sync_db:
                self.backup_db_to_cloud(custom_note=title)
            if sync_ohlcv:
                self.backup_ohlcv_to_cloud(custom_note=title)

        threading.Thread(target=_worker, daemon=True).start()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status of Telegram Cloud Vault (DB + OHLCV Cache)."""
        creds = self.get_credentials()
        manifest = self.get_manifest()
        last_db = manifest.get("last_db_backup") or {}
        last_ohlcv = manifest.get("last_ohlcv_backup") or {}

        pkl_files = list(HISTORICAL_DATA_DIR.glob("*.pkl"))
        local_ohlcv_count = len(pkl_files)

        latest_local_date = "--"
        if pkl_files:
            try:
                sample_df = pd.read_pickle(pkl_files[0])
                if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
                    latest_local_date = str(sample_df.index[-1])[:10]
            except Exception:
                pass

        return {
            "status": "ONLINE" if creds["enabled"] else "DISABLED",
            "enabled": creds["enabled"],
            "token_masked": creds["token_masked"],
            "chat_id": creds["chat_id_masked"],
            "local_ohlcv_stocks": local_ohlcv_count,
            "local_ohlcv_files": local_ohlcv_count,
            "latest_local_ohlcv_date": latest_local_date,
            "latest_bar_date": last_ohlcv.get("latest_bar_date") or latest_local_date,
            "last_db_backup_ist": last_db.get("timestamp", "Synced"),
            "last_db_backup": {
                "timestamp": last_db.get("timestamp", "Never"),
                "filename": last_db.get("filename", "--"),
                "size_kb": last_db.get("size_kb", 0.0),
                "file_id": last_db.get("file_id", "")
            },
            "last_ohlcv_backup": {
                "timestamp": last_ohlcv.get("timestamp", "Never"),
                "stocks_count": last_ohlcv.get("stocks_count", 0),
                "latest_bar_date": last_ohlcv.get("latest_bar_date", "--"),
                "size_kb": last_ohlcv.get("size_kb", 0.0),
                "file_id": last_ohlcv.get("file_id", "")
            },
            "history": manifest.get("history", [])[:10]
        }

    def get_vault_status(self) -> Dict[str, Any]:
        return self.get_status()


# Singleton Instance
telegram_vault = TelegramVaultManager()
