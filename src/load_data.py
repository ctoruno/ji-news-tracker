import pandas as pd

eu_member_states = [
    "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland",
    "France","Germany","Greece", "Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg",
    "Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"
]


def read_data(country):
    """
    Load data from a parquet file based on the specified country.

    Args:
        country (str): The name of the country to load data for.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """

    print(f"Loading data for {country}...")

    target_columns = [
        "id", "link", "domain_url", "published_date", "title_trans",
        "description_trans", "content_trans", "summary", "impact_score"
    ]

    data_path = f"https://github.com/ctoruno/EU-copilot/raw/refs/heads/main/data/{country}_master.parquet.gzip"
    df = pd.read_parquet(data_path)
    filtered_df = (
        df.copy()
        .loc[df["associated_pillar"].isin(["Pillar 1", "Pillar 7", "Pillar 8"]), target_columns]
        .drop_duplicates(subset=["id"])
        .assign(
            country = country
        )
    )
    filtered_df["published_date"] = pd.to_datetime(filtered_df["published_date"])
    
    return filtered_df


def gather_data():
    """
    Gather data for all EU member states.

    Returns:
        dict: A dictionary where keys are country names and values are DataFrames.
    """

    gathered_data = [read_data(member) for member in eu_member_states]
    master_data = pd.concat(gathered_data, ignore_index=True)
    
    return master_data


if __name__ == "__main__":
    master = gather_data()
    print(master.head())