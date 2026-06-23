import pandas as pd

sheets = ['Stores', 'Promotions', 'Weather', 'Inventory', 'CompetitionIndex', 'CustomerFeedback']
try:
    xl = pd.ExcelFile('data/DILO.xlsx')
    for sheet in sheets:
        if sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            print(f"\nColumns in {sheet}:")
            print(df.columns.tolist())
        else:
            print(f"\nSheet {sheet} not found")
except Exception as e:
    print(e)
