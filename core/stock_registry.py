"""
NSE Equities Sector Registry & Benchmark Mappings
"""
from typing import Dict

NSE_SECTOR_MAP = {
    # Banking & Financial Services -> ^NSEBANK
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK", "KOTAKBANK": "BANK", "AXISBANK": "BANK",
    "INDUSINDBK": "BANK", "BANKBARODA": "BANK", "PNB": "BANK", "CANBK": "BANK", "AUBANK": "BANK",
    "FEDERALBNK": "BANK", "IDFCFIRSTB": "BANK", "BANDHANBNK": "BANK", "BAJFINANCE": "BANK",
    "BAJAJFINSV": "BANK", "CHOLAFIN": "BANK", "SHRIRAMFIN": "BANK", "MUTHOOTFIN": "BANK",
    "M&MFIN": "BANK", "SUNDARMFIN": "BANK", "POONAWALLA": "BANK", "MANAPPURAM": "BANK",
    "SBICARD": "BANK", "HDFCLIFE": "BANK", "SBILIFE": "BANK", "ICICIPRULI": "BANK", "ICICIGI": "BANK",
    
    # IT & Technology -> ^CNXIT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT",
    "COFORGE": "IT", "PERSISTENT": "IT", "MPHASIS": "IT", "LTTS": "IT", "TATAELXSI": "IT",
    "KPITTECH": "IT", "OFSS": "IT", "CYIENT": "IT", "SONATSOFTW": "IT", "BSOFT": "IT",
    "ZENSARTECH": "IT", "HAPPSTMNDS": "IT", "MASTEK": "IT", "INTELLECT": "IT", "LATENTVIEW": "IT",
    
    # Auto & Ancillaries -> ^CNXAUTO
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO", "HEROMOTOCO": "AUTO",
    "EICHERMOT": "AUTO", "TVSMOTOR": "AUTO", "BHARATFORG": "AUTO", "ASHOKLEY": "AUTO", "MOTHERSON": "AUTO",
    "BOSCHLTD": "AUTO", "BALKRISIND": "AUTO", "MRF": "AUTO", "APOLLOTYRE": "AUTO", "EXIDEIND": "AUTO",
    "AMARAJABAT": "AUTO", "SONACOMS": "AUTO", "CIEINDIA": "AUTO", "TIINDIA": "AUTO", "ESCORTS": "AUTO",
    
    # Pharma & Healthcare -> ^CNXPHARMA
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA", "LUPIN": "PHARMA",
    "AUROPHARMA": "PHARMA", "IPCALAB": "PHARMA", "LAURUSLABS": "PHARMA", "TORNTPHARM": "PHARMA",
    "ALKEM": "PHARMA", "ABBOTINDIA": "PHARMA", "BIOCON": "PHARMA", "GLENMARK": "PHARMA", "ZYDUSLIFE": "PHARMA",
    "MANKIND": "PHARMA", "APOLLOHOSP": "PHARMA", "MAXHEALTH": "PHARMA", "FORTIS": "PHARMA", "SYNGENE": "PHARMA",
    "NATCOPHARM": "PHARMA", "AJANTPHARM": "PHARMA", "ERIS": "PHARMA", "GRANULES": "PHARMA", "SURAKSHA": "PHARMA",
    "GLAXO": "PHARMA", "SANOFI": "PHARMA", "PFIZER": "PHARMA",
    
    # FMCG & Consumption -> ^CNXFMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG", "TATACONSUM": "FMCG",
    "DABUR": "FMCG", "GODREJCP": "FMCG", "MARICO": "FMCG", "VBL": "FMCG", "COLPAL": "FMCG",
    "PGHH": "FMCG", "RADICO": "FMCG", "UBL": "FMCG", "UNITDSPR": "FMCG", "EMAMILTD": "FMCG",
    "JYOTHYLAB": "FMCG", "BIKAJI": "FMCG", "PATANJALI": "FMCG",
    
    # Metals & Mining -> ^CNXMETAL
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL", "VEDL": "METAL", "JINDALSTEL": "METAL",
    "NMDC": "METAL", "SAIL": "METAL", "NATIONALUM": "METAL", "COALINDIA": "METAL", "HINDZINC": "METAL",
    "APLAPOLLO": "METAL", "JSL": "METAL", "RATNAMANI": "METAL", "WELCORP": "METAL", "HINDCOPPER": "METAL",
    
    # Realty & Infrastructure -> ^CNXREALTY
    "DLF": "REALTY", "GODREJPROP": "REALTY", "LODHA": "REALTY", "OBEROIRLTY": "REALTY", "PHOENIXLTD": "REALTY",
    "BRIGADE": "REALTY", "PRESTIGE": "REALTY", "SOBHA": "REALTY", "SUNTECK": "REALTY", "SIGNATURE": "REALTY",
    "MAHLIFE": "REALTY",
    
    # Energy, Oil & Power -> ^CNXENERGY
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "GAIL": "ENERGY", "ADANIGREEN": "ENERGY", "ADANIPOWER": "ENERGY", "ADANIENSOL": "ENERGY",
    "TATAPOWER": "ENERGY", "NHPC": "ENERGY", "IEX": "ENERGY", "OIL": "ENERGY", "PETRONET": "ENERGY",
    "GUJGASLTD": "ENERGY", "IGL": "ENERGY", "MGL": "ENERGY", "SJVN": "ENERGY", "SUZLON": "ENERGY",
    
    # Capital Goods & Industrials -> ^CRSLDX proxy
    "LT": "CAPGOODS", "SIEMENS": "CAPGOODS", "ABB": "CAPGOODS", "BHEL": "CAPGOODS", "HAL": "CAPGOODS",
    "BEL": "CAPGOODS", "CUMMINSIND": "CAPGOODS", "THERMAX": "CAPGOODS", "AIAENG": "CAPGOODS", "APARINDS": "CAPGOODS",
    "MTARTECH": "CAPGOODS", "KEI": "CAPGOODS", "POLYCAB": "CAPGOODS", "HAVELLS": "CAPGOODS", "VOLTAS": "CAPGOODS",
    "DIXON": "CAPGOODS", "KAYNES": "CAPGOODS", "ASTRAL": "CAPGOODS", "SUPREMEIND": "CAPGOODS",
    
    # Chemicals & Specialty Materials -> ^CRSLDX proxy
    "PIDILITIND": "CHEMICALS", "SRF": "CHEMICALS", "NAVINFLUOR": "CHEMICALS", "DEEPAKNTR": "CHEMICALS",
    "TATACHEM": "CHEMICALS", "FINEORG": "CHEMICALS", "PRIVISCL": "CHEMICALS", "FLUOROCHEM": "CHEMICALS",
    "ATUL": "CHEMICALS", "AARTIIND": "CHEMICALS", "PIIND": "CHEMICALS", "COROMANDEL": "CHEMICALS"
}

SECTOR_BENCHMARK_MAP = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "REALTY": "^CNXREALTY",
    "ENERGY": "^CNXENERGY",
    "CAPGOODS": "^CRSLDX",   # Nifty 500 proxy
    "CHEMICALS": "^CRSLDX",  # Nifty 500 proxy
    "DEFAULT": "^CRSLDX"     # Nifty 500 proxy
}

def clean_symbol(ticker: str) -> str:
    """Removes .NS or .BO suffix and whitespace."""
    if not ticker:
        return ""
    return ticker.replace(".NS", "").replace(".BO", "").strip().upper()

def get_stock_sector(ticker: str) -> str:
    """Returns sector group for given ticker."""
    sym = clean_symbol(ticker)
    return NSE_SECTOR_MAP.get(sym, "DIVERSIFIED")

def format_nse_ticker(ticker: str) -> str:
    """Ensures .NS suffix for Yahoo Finance ticker requests."""
    sym = ticker.strip().upper()
    if sym.startswith("^"):
        return sym
    if not sym.endswith(".NS") and not sym.endswith(".BO"):
        return f"{sym}.NS"
    return sym
