import pandas as pd

csv_files = [
    "/content/ResaleFlatPricesBasedonApprovalDate19901999.csv",
    "/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv",
    "/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv",
    "/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv",
    "/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv"
]

df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)


df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

if 'remaining_lease' in df.columns:
    df.drop(columns=['remaining_lease'], inplace=True)
df['month'] = pd.to_datetime(df['month'], errors='coerce')
df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
df['floor_area_sqm'] = pd.to_numeric(df.get('floor_area_sqm'), errors='coerce')
df['lease_commence_date'] = pd.to_numeric(df.get('lease_commence_date'), errors='coerce')

df.dropna(subset=['month', 'resale_price', 'town', 'flat_type'], inplace=True)

categorical_cols = ['town', 'flat_type', 'flat_model', 'block', 'street_name']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

df.sort_values(by='month', inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv("/content/cleaned_hdb_resale_data.csv", index=False)
df.to_parquet("/content/cleaned_hdb_resale_data.parquet")
