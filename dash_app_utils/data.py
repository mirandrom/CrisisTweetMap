import pandas as pd
import sqlite3

def get_data(db : str, table: str, num_hours: int = 0, num_minutes: int = 30):
    """Extracts tweets from the last num_hours:num_minutes

    """
    con = sqlite3.connect(db)
    df_plot = pd.read_sql_query(f"SELECT * FROM {table} WHERE datetime(created_at) >= datetime('now', -{num_hours} hours', -{num_minutes} minutes')", con)
    return df_plot