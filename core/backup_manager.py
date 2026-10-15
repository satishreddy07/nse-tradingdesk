"""
Market-On-Close (MOC) Daily Database Backup Manager.
Creates an atomic ACID snapshot of the live SQLite database, verifies PRAGMA integrity,
prunes backups older than the retention period, and optionally triggers offsite Cloud Vault sync.
"""
import os
import shutil
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from core import config

logger = logging.getLogger("core.backup_manager")

# IST timezone helper
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def format_ist(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = now_ist()
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")


class DailyBackupManager:
    """
    Production-grade daily SQLite backup engine for dedicated 24/7 VPS.
    Scheduled once daily at 15:45 IST (MOC after market close).
    """

    def __init__(self, backup_dir: Optional[Path] = None, retention_days: int = 14):
        self.retention_days = retention_days
        if backup_dir:
            self.backup_dir = Path(backup_dir)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Check if production /var/backups path is writable or can be created
            vps_backup_path = Path("/var/backups/nse-tradingdesk/daily")
            try:
                vps_backup_path.mkdir(parents=True, exist_ok=True)
                test_file = vps_backup_path / ".perm_check"
                test_file.touch()
                test_file.unlink()
                self.backup_dir = vps_backup_path
            except Exception:
                # Fallback to application data/backups/daily
                self.backup_dir = config.DATA_DIR / "backups" / "daily"
                self.backup_dir.mkdir(parents=True, exist_ok=True)

    def run_daily_moc_backup(
        self,
        retention_days: Optional[int] = None,
        sync_cloud: bool = True,
        custom_note: str = "Market-On-Close Daily Snapshot"
    ) -> Dict[str, Any]:
        """
        Executes the atomic MOC backup sequence:
        1. Ensures destination directory exists.
        2. Performs online hot backup via sqlite3.Connection.backup.
        3. Executes PRAGMA integrity_check on the generated snapshot.
        4. Auto-prunes snapshots older than retention_days.
        5. Syncs snapshot to Telegram Cloud Vault if configured.
        """
        retention = retention_days if retention_days is not None else self.retention_days
        source_db = Path(config.DB_PATH)

        if not source_db.exists():
            msg = f"Backup failed: Source database {source_db} does not exist."
            logger.error(msg)
            return {"status": "ERROR", "message": msg}

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts_now = now_ist()
        timestamp_str = ts_now.strftime("%Y%m%d_%H%M%S")
        dest_filename = f"swing_trades_moc_{timestamp_str}.db"
        dest_path = self.backup_dir / dest_filename

        logger.info(f"Initiating MOC Daily Backup: {source_db} -> {dest_path}")

        try:
            # 1. Hot online SQLite backup (zero table locks, ACID safe)
            with sqlite3.connect(str(source_db), timeout=60.0) as src_conn:
                with sqlite3.connect(str(dest_path)) as dst_conn:
                    src_conn.backup(dst_conn)

            # 2. Verify integrity of the new backup
            with sqlite3.connect(str(dest_path)) as chk_conn:
                cur = chk_conn.cursor()
                cur.execute("PRAGMA integrity_check;")
                row = cur.fetchone()
                integrity_result = row[0] if row else "unknown"

            if integrity_result.lower() != "ok":
                logger.error(f"Integrity check failed for {dest_path}: {integrity_result}")
                return {
                    "status": "INTEGRITY_FAILED",
                    "backup_path": str(dest_path),
                    "integrity": integrity_result,
                    "message": "PRAGMA integrity_check returned non-ok result."
                }

            backup_size_bytes = dest_path.stat().st_size
            backup_size_mb = round(backup_size_bytes / (1024 * 1024), 2)

            # 3. Prune older backups
            pruned_files = self.prune_old_backups(retention_days=retention)

            # 4. Optional Cloud Vault sync
            cloud_status = "SKIPPED"
            if sync_cloud:
                try:
                    from core.telegram_vault import telegram_vault
                    creds = telegram_vault.get_credentials()
                    if creds.get("enabled"):
                        cloud_res = telegram_vault.backup_master_file_to_cloud(
                            custom_note=f"{custom_note} ({format_ist(ts_now)})",
                            send_notification=False
                        )
                        cloud_status = cloud_res.get("status", "COMPLETED")
                except Exception as e_cloud:
                    logger.warning(f"Cloud vault sync during daily backup encountered warning: {e_cloud}")
                    cloud_status = f"WARNING: {str(e_cloud)}"

            report = {
                "status": "SUCCESS",
                "backup_filename": dest_filename,
                "backup_path": str(dest_path),
                "backup_size_mb": backup_size_mb,
                "integrity_check": integrity_result,
                "retention_days": retention,
                "pruned_count": len(pruned_files),
                "pruned_files": pruned_files,
                "cloud_sync": cloud_status,
                "timestamp_ist": format_ist(ts_now)
            }
            logger.info(
                f"✓ MOC Daily Backup Complete: {dest_filename} ({backup_size_mb} MB) | "
                f"Integrity: {integrity_result} | Pruned {len(pruned_files)} old snapshots | Cloud: {cloud_status}"
            )
            return report

        except Exception as e:
            logger.exception(f"Exception during MOC daily backup: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "timestamp_ist": format_ist(ts_now)
            }

    def prune_old_backups(self, retention_days: int) -> List[str]:
        """
        Scans self.backup_dir for files matching swing_trades_moc_*.db.
        Removes files modified older than retention_days.
        """
        pruned = []
        cutoff_time = datetime.now().timestamp() - (retention_days * 86400)

        try:
            for file_path in self.backup_dir.glob("swing_trades_moc_*.db"):
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    if mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            pruned.append(file_path.name)
                            logger.info(f"Pruned expired backup snapshot: {file_path.name}")
                        except Exception as pe:
                            logger.warning(f"Could not prune {file_path.name}: {pe}")
        except Exception as e:
            logger.warning(f"Error scanning for old backups to prune: {e}")

        return pruned

    def list_backups(self) -> List[Dict[str, Any]]:
        """Lists existing daily backups with size and modified timestamp."""
        results = []
        if not self.backup_dir.exists():
            return results

        for p in sorted(self.backup_dir.glob("swing_trades_moc_*.db"), reverse=True):
            if p.is_file():
                stat = p.stat()
                results.append({
                    "filename": p.name,
                    "path": str(p),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")
                })
        return results


daily_backup_manager = DailyBackupManager()
