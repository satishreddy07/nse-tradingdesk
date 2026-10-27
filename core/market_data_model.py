"""
core/market_data_model.py
=========================
Canonical Normalized Market-Data Model for NSE Intraday Analytics & Historical Storage.
Complies with AGENTS.md & docs/ANTI_FABRICATION_POLICY.md.

Establishes a strict data contract for:
- Quote events (LTP, OHLC, volume, OI, bid/ask)
- 5-Level Market Depth & Microstructure metrics (OBI, microprice, midpoint, displacement)
- Trade execution records
- Full Option-Chain snapshots
- 10-second analytical snapshots & 10-minute consolidated intervals
- Unified data-quality states
"""

import time
import uuid
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from core.data_quality import IST, now_ist, format_ist


class DataQualityStatus(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    RECONNECTING = "RECONNECTING"
    SUBSCRIPTION_MISSING = "SUBSCRIPTION_MISSING"
    CALCULATION_ERROR = "CALCULATION_ERROR"


class MarketEventType(str, Enum):
    RAW_TICK = "RAW_TICK"
    NORMALIZED_QUOTE = "NORMALIZED_QUOTE"
    DEPTH_SNAPSHOT = "DEPTH_SNAPSHOT"
    TRADE_EVENT = "TRADE_EVENT"
    OPTION_CHAIN_SNAPSHOT = "OPTION_CHAIN_SNAPSHOT"
    ANALYTICAL_SNAPSHOT_10S = "ANALYTICAL_SNAPSHOT_10S"
    CONSOLIDATED_SNAPSHOT_10M = "CONSOLIDATED_SNAPSHOT_10M"
    DATA_QUALITY_ALERT = "DATA_QUALITY_ALERT"


class InstrumentType(str, Enum):
    INDEX = "INDEX"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    COMMODITY = "COMMODITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class NormalizedMarketEvent:
    """Canonical representation of an authentic market-data event."""
    event_id: str
    instrument_token: str
    trading_symbol: str
    exchange: str
    segment: str
    instrument_type: InstrumentType
    receive_timestamp_utc: float
    receive_timestamp_ist: str
    event_type: MarketEventType
    quality_status: DataQualityStatus
    schema_version: str = "v1.0"
    
    exchange_timestamp: Optional[str] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None
    source_event_id: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    normalized_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        instrument_token: str,
        trading_symbol: str,
        event_type: MarketEventType,
        normalized_payload: Dict[str, Any],
        exchange: str = "NSE",
        segment: str = "EQUITY",
        instrument_type: InstrumentType = InstrumentType.UNKNOWN,
        quality_status: DataQualityStatus = DataQualityStatus.VALID,
        exchange_timestamp: Optional[str] = None,
        expiry: Optional[str] = None,
        strike: Optional[float] = None,
        option_type: Optional[str] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        source_event_id: Optional[str] = None,
    ) -> "NormalizedMarketEvent":
        now_utc = time.time()
        now_str = format_ist()
        evt_id = f"{instrument_token}_{int(now_utc * 1000)}_{uuid.uuid4().hex[:6]}"
        return cls(
            event_id=evt_id,
            instrument_token=str(instrument_token or ""),
            trading_symbol=str(trading_symbol or ""),
            exchange=exchange,
            segment=segment,
            instrument_type=instrument_type,
            receive_timestamp_utc=now_utc,
            receive_timestamp_ist=now_str,
            event_type=event_type,
            quality_status=quality_status,
            exchange_timestamp=exchange_timestamp,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            source_event_id=source_event_id,
            raw_payload=raw_payload,
            normalized_payload=normalized_payload,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["instrument_type"] = self.instrument_type.value
        d["event_type"] = self.event_type.value
        d["quality_status"] = self.quality_status.value
        return d


@dataclass
class MarketDepthRecord:
    """Normalized 5-level market depth record with strict microstructure metrics."""
    instrument_token: str
    trading_symbol: str
    ltp: Optional[float]
    bids: List[Dict[str, Any]]  # [{"price": float, "quantity": int, "orders": int}, ...]
    asks: List[Dict[str, Any]]
    total_buy_quantity: int
    total_sell_quantity: int
    l1_obi: Optional[float]
    l3_obi: Optional[float]
    l5_obi: Optional[float]
    microprice: Optional[float]
    midpoint: Optional[float]
    displacement: Optional[float]
    spread: Optional[float]
    relative_spread_pct: Optional[float]
    book_status: str
    receive_timestamp_utc: float
    receive_timestamp_ist: str
    exchange_timestamp: Optional[str] = None
    quality_status: DataQualityStatus = DataQualityStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["quality_status"] = self.quality_status.value
        return d


@dataclass
class OptionChainRecord:
    """Normalized full option-chain snapshot record."""
    expiry: str
    spot_price: float
    atm_strike: int
    strikes: List[Dict[str, Any]]
    total_ce_oi: int
    total_pe_oi: int
    pcr: float
    call_wall: Optional[int]
    put_wall: Optional[int]
    max_pain: Optional[int]
    receive_timestamp_utc: float
    receive_timestamp_ist: str
    quality_status: DataQualityStatus = DataQualityStatus.VALID


@dataclass
class AnalyticalSnapshot10s:
    """Consolidated 10-second analytical market state snapshot."""
    session_date: str
    timestamp_utc: float
    timestamp_ist: str
    spot_price: float
    futures_price: Optional[float]
    futures_obi: Optional[float]
    futures_microprice: Optional[float]
    futures_displacement: Optional[float]
    futures_volume_delta: Optional[float]
    futures_cvd: Optional[float]
    atm_strike: Optional[int]
    atm_iv: Optional[float]
    atm_ce_iv: Optional[float]
    atm_pe_iv: Optional[float]
    iv_skew: Optional[float]
    total_net_gex_cr: Optional[float]
    zero_gamma_flip: Optional[float]
    spot_vs_flip_pts: Optional[float]
    gex_regime: Optional[str]
    vwap: Optional[float]
    vwap_distance_pts: Optional[float]
    vwap_distance_pct: Optional[float]
    orb_15m_high: Optional[float]
    orb_15m_low: Optional[float]
    orb_15m_width: Optional[float]
    orb_15m_status: Optional[str]
    orb_30m_high: Optional[float]
    orb_30m_low: Optional[float]
    orb_30m_width: Optional[float]
    orb_30m_status: Optional[str]
    sector_breadth_pct: Optional[float]
    sector_spread_pp: Optional[float]
    weighted_index_drag_pp: Optional[float]
    top_drag_sector: Optional[str]
    top_lift_sector: Optional[str]
    data_quality_status: DataQualityStatus = DataQualityStatus.VALID
    data_quality_flags: List[str] = field(default_factory=list)


@dataclass
class ConsolidatedSnapshot10m:
    """Consolidated 10-minute historical macro interval record."""
    session_date: str
    slot_label: str
    slot_start_utc: float
    slot_end_utc: float
    open_spot: Optional[float]
    high_spot: Optional[float]
    low_spot: Optional[float]
    close_spot: Optional[float]
    spot_change_pts: Optional[float]
    spot_change_pct: Optional[float]
    vwap: Optional[float]
    vwap_distance_pts: Optional[float]
    futures_flow_summary: Dict[str, Any] = field(default_factory=dict)
    obi_stats: Dict[str, Any] = field(default_factory=dict)
    delta_cvd_summary: Dict[str, Any] = field(default_factory=dict)
    option_chain_summary: Dict[str, Any] = field(default_factory=dict)
    gex_summary: Dict[str, Any] = field(default_factory=dict)
    sector_breadth_pct: Optional[float] = None
    sector_spread_pp: Optional[float] = None
    weighted_index_drag_pp: Optional[float] = None
    completeness_score: float = 100.0
    missing_intervals: List[Dict[str, Any]] = field(default_factory=list)
    quality_status: DataQualityStatus = DataQualityStatus.VALID
