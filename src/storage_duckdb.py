import duckdb
import pandas as pd
from src.config import DB_PATH

def get_connection():
    return duckdb.connect(str(DB_PATH))

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT,
            provider TEXT,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY(symbol, provider, date)
        );
    """)
    conn.close()

def upsert_ohlcv(df: pd.DataFrame):
    if df.empty:
        return
    
    conn = get_connection()
    # Ensure columns match expected schema order/names or select explicitly
    # Expected: symbol, provider, date, open, high, low, close, volume
    
    # We'll rely on the dataframe having these columns.
    # Convert date to date object if it's datetime, for DuckDB DATE type compatibility
    # although DuckDB handles datetime -> date cast usually.
    
    conn.register('df_source', df)
    conn.execute("""
        INSERT INTO ohlcv 
        SELECT symbol, provider, date, open, high, low, close, volume 
        FROM df_source
        ON CONFLICT (symbol, provider, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """)
    conn.close()

def load_ohlcv(symbol: str, provider: str, start=None, end=None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM ohlcv WHERE symbol = ? AND provider = ?"
    params = [symbol, provider]
    
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
        
    query += " ORDER BY date ASC"
    
    df = conn.execute(query, params).df()
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def latest_date(symbol: str, provider: str):
    conn = get_connection()
    res = conn.execute(
        "SELECT MAX(date) FROM ohlcv WHERE symbol = ? AND provider = ?", 
        [symbol, provider]
    ).fetchone()
    conn.close()
    return res[0] if res else None

# Initialize DB on import to ensure table exists
init_db()
