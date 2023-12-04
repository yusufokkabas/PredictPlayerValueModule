import pyodbc

def create_conn():
    server = 'idsscout.database.windows.net' 
    database = 'idsscout' 
    username = 'idssdb@idsscout'
    password = '.99#wA&47R' 
    driver= '{ODBC Driver 17 for SQL Server}'
    conn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    return conn

# Establish connection
connection = create_conn()