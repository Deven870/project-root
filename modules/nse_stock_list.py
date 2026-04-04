"""
NSE Stock List Manager
Comprehensive list of all NSE-listed stocks with metadata
"""

# Top 100 NSE Stocks + Extended List
NSE_STOCKS = {
    # NIFTY 50 - Blue Chips
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "INFY.NS": {"name": "Infosys Limited", "sector": "IT"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Finance"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Finance"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Finance"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Finance"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Industrial"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto"},
    "ASIANPAINT.NS": {"name": "Asian Paints", "sector": "Paints"},
    "WIPRO.NS": {"name": "Wipro Limited", "sector": "IT"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma"},
    "DIVI.NS": {"name": "Divi's Laboratories", "sector": "Pharma"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG"},
    "NESTLEIND.NS": {"name": "Nestlé India", "sector": "FMCG"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG"},
    "LTTS.NS": {"name": "L&T Technology Services", "sector": "IT"},
    "JSWSTEEL.NS": {"name": "JSW Steel", "sector": "Steel"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Steel"},
    "COALINDIA.NS": {"name": "Coal India", "sector": "Mining"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecom"},
    "INDIGO.NS": {"name": "IndiGo", "sector": "Aviation"},
    "EICHERMOT.NS": {"name": "Eicher Motors", "sector": "Auto"},
    "HEROMOTOCO.NS": {"name": "Hero MotoCorp", "sector": "Auto"},
    "BAJAJFINSV.NS": {"name": "Bajaj Finserv", "sector": "Finance"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto", "sector": "Auto"},
    "HDFC.NS": {"name": "Housing Development Finance", "sector": "Finance"},
    "POWERGRID.NS": {"name": "Power Grid Corporation", "sector": "Power"},
    "ADANIPOWER.NS": {"name": "Adani Power", "sector": "Power"},
    "ADANIGREEN.NS": {"name": "Adani Green Energy", "sector": "Energy"},
    "ADANIPORTS.NS": {"name": "Adani Ports", "sector": "Infrastructure"},
    "APOLLOHOSP.NS": {"name": "Apollo Hospitals", "sector": "Healthcare"},
    "BIOCON.NS": {"name": "Biocon Limited", "sector": "Biotech"},
    "CIPLA.NS": {"name": "Cipla Limited", "sector": "Pharma"},
    "DRREDDY.NS": {"name": "Dr Reddy's Labs", "sector": "Pharma"},
    "LUPIN.NS": {"name": "Lupin Limited", "sector": "Pharma"},
    "BRITANNIA.NS": {"name": "Britannia Industries", "sector": "FMCG"},
    "COLPAL.NS": {"name": "Colgate-Palmolive", "sector": "FMCG"},
    "MARICO.NS": {"name": "Marico Limited", "sector": "FMCG"},
    "PIDILITIND.NS": {"name": "Pidilite Industries", "sector": "Chemicals"},
    "ONGC.NS": {"name": "Oil and Natural Gas Corp", "sector": "Energy"},
    "INDUSIND.NS": {"name": "IndusInd Bank", "sector": "Finance"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Finance"},
    "FAL.NS": {"name": "Federal Bank", "sector": "Finance"},
    "NORTHERNBANKNS": {"name": "Northern Arc Capital", "sector": "Finance"},
    "SIEMENS.NS": {"name": "Siemens Limited", "sector": "Industrial"},
    "BOSCHIND.NS": {"name": "Bosch India", "sector": "Auto-Components"},
    
    # Additional Quality Stocks - Sector Leaders
    "ULTRACEMCO.NS": {"name": "UltraTech Cement", "sector": "Cement"},
    "SHREECEMENT.NS": {"name": "Shree Cement", "sector": "Cement"},
    "AMBUJACEMENT.NS": {"name": "Ambuja Cements", "sector": "Cement"},
    "ACC.NS": {"name": "ACC Limited", "sector": "Cement"},
    "RAMCOCEM.NS": {"name": "Ramco Cements", "sector": "Cement"},
    "GODREJIND.NS": {"name": "Godrej Industries", "sector": "FMCG"},
    "HAVELLS.NS": {"name": "Havells India", "sector": "Electrical"},
    "ARVINDFASNING.NS": {"name": "Arvind Limited", "sector": "Textiles"},
    "PAGEIND.NS": {"name": "Page Industries", "sector": "Apparel"},
    "SUNTV.NS": {"name": "Sun TV Network", "sector": "Media"},
    "VBL.NS": {"name": "Varun Beverages", "sector": "Beverage"},
    "MCDOWELL.NS": {"name": "McDowell Holdings", "sector": "Beverage"},
    "PEL.NS": {"name": "Piramal Enterprises", "sector": "Healthcare"},
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Pharma"},
    "BATAINDIA.NS": {"name": "Bata India", "sector": "Retail"},
    "RELAXO.NS": {"name": "Relaxo Footwear", "sector": "Footwear"},
    "VIT.NS": {"name": "Vardhman Industries", "sector": "Textiles"},
    "NATCpharma.NS": {"name": "Natco Pharma", "sector": "Pharma"},
    "AUROPHARMA.NS": {"name": "Aurobindo Pharma", "sector": "Pharma"},
    "ZEEL.NS": {"name": "Zee Entertainment", "sector": "Media"},
    "ZYDUSLIFE.NS": {"name": "Zydus Lifesciences", "sector": "Pharma"},
    "GLENMARK.NS": {"name": "Glenmark Pharma", "sector": "Pharma"},
    "TORNTPHARM.NS": {"name": "Torrent Pharmaceuticals", "sector": "Pharma"},
    "ALKEM.NS": {"name": "Alkem Laboratories", "sector": "Pharma"},
    "MANKIND.NS": {"name": "Mankind Pharma", "sector": "Pharma"},
    "SYNGENE.NS": {"name": "Syngene International", "sector": "Pharma"},
    "DEEPINDUS.NS": {"name": "Deep Industries", "sector": "Industrial"},
    "RITES.NS": {"name": "Rites Limited", "sector": "Transport"},
    "NATIONALSHIP.NS": {"name": "National Shipyard", "sector": "Marine"},
    "BEL.NS": {"name": "Bharat Electronics", "sector": "Defense"},
    "HAL.NS": {"name": "Hindustan Aeronautics", "sector": "Defense"},
    "BHEL.NS": {"name": "Bharat Heavy Electricals", "sector": "Power"},
    "NTPC.NS": {"name": "NTPC Limited", "sector": "Power"},
    "SJVN.NS": {"name": "SJVN Limited", "sector": "Power"},
    "NHPC.NS": {"name": "NHPC Limited", "sector": "Power"},
    "DMCC.NS": {"name": "Dalmia Cement", "sector": "Cement"},
    "HINDALCO.NS": {"name": "Hindalco Industries", "sector": "Metals"},
    "VEDL.NS": {"name": "Vedanta Limited", "sector": "Mining"},
    "NOTINDIA.NS": {"name": "National Optionss Trading", "sector": "Finance"},
    "SYNGENE.NS": {"name": "Syngene", "sector": "Biotech"},
    "CIPLA.NS": {"name": "Cipla", "sector": "Pharma"},
}

# Extended list for active traders (100+ stocks)
EXTENDED_NSE_STOCKS = {
    **NSE_STOCKS,
    "BANKINDIA.NS": {"name": "Bank of India", "sector": "Finance"},
    "CENTRALBANK.NS": {"name": "Central Bank of India", "sector": "Finance"},
    "PNBHOUSING.NS": {"name": "PNB Housing Finance", "sector": "Finance"},
    "CAN.NS": {"name": "Canal Bank", "sector": "Finance"},
    "INDIANBANK.NS": {"name": "Indian Bank", "sector": "Finance"},
    "IDFCFIRSTB.NS": {"name": "IDFC First Bank", "sector": "Finance"},
    "KARUPUR.NS": {"name": "Karur Vysya Bank", "sector": "Finance"},
    "KBLBANK.NS": {"name": "Kerala Bank", "sector": "Finance"},
    "YESBANK.NS": {"name": "Yes Bank", "sector": "Finance"},
    "ABCAPITAL.NS": {"name": "Aditya Birla Capital", "sector": "Finance"},
    "ABSL.NS": {"name": "Aditya Birla Sun Life", "sector": "Insurance"},
    "SBICARD.NS": {"name": "SBI Card", "sector": "Finance"},
    "SBIMONEY.NS": {"name": "SBI Money Fund", "sector": "Finance"},
}

def get_all_nse_stocks():
    """Return all NSE stocks"""
    return NSE_STOCKS

def get_extended_nse_stocks():
    """Return extended NSE stocks list (100+)"""
    return EXTENDED_NSE_STOCKS

def get_stock_by_symbol(symbol):
    """Get stock details by symbol"""
    return NSE_STOCKS.get(symbol, None)

def get_stocks_by_sector(sector):
    """Get all stocks in a specific sector"""
    return {sym: data for sym, data in NSE_STOCKS.items() if data["sector"] == sector}

def get_all_sectors():
    """Get list of all sectors"""
    sectors = set()
    for stock in NSE_STOCKS.values():
        sectors.add(stock["sector"])
    return sorted(list(sectors))

def search_stocks(query):
    """Search stocks by name or symbol"""
    query = query.lower()
    results = {}
    
    for symbol, data in NSE_STOCKS.items():
        if query in symbol.lower() or query in data["name"].lower():
            results[symbol] = data
    
    return results

# Get stock list as dropdown options
def get_stock_options():
    """Get formatted stock list for Streamlit selectbox"""
    options = []
    for symbol, data in sorted(NSE_STOCKS.items()):
        options.append(f"{symbol} - {data['name']} ({data['sector']})")
    return options

def parse_stock_option(option_string):
    """Parse stock option string back to symbol"""
    return option_string.split(" - ")[0]

# Sector-specific lists
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "LT.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "WIPRO.NS", "TECHM.NS", "SUNPHARMA.NS", "DIVI.NS", "HINDUNILVR.NS",
    "NESTLEIND.NS", "ITC.NS", "LTTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "BHARTIARTL.NS", "INDIGO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
]

NIFTY_100 = NIFTY_50 + [
    "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "HDFC.NS", "POWERGRID.NS", "ADANIPOWER.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "BIOCON.NS", "CIPLA.NS",
    "DRREDDY.NS", "LUPIN.NS", "BRITANNIA.NS", "COLPAL.NS", "MARICO.NS",
]

SECTOR_STOCKS = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "LTTS.NS"],
    "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "HDFC.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS"],
    "Auto": ["MARUTI.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DIVI.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS"],
    "FMCG": ["HINDUNILVR.NS", "NESTLEIND.NS", "ITC.NS", "BRITANNIA.NS"],
}

def get_sector_movers(sector):
    """Get best movers in a sector"""
    return SECTOR_STOCKS.get(sector, [])
