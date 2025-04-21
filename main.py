"""
Main script for running the binary classification project pipeline.
"""

import os
import argparse
import logging
import time
import pandas as pd
from dotenv import load_dotenv

from src.load_data import gather_data
from src.classifier import process_data
from src.topic_model import newsData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def main(args):
    """
    Main function to run the entire pipeline or specific steps.
    """

    start_time = time.time()

    if args.data:
        logger.info("Gathering country data...")
        master_data = gather_data()
        master_data.to_parquet("data/master_data_p1.parquet.gzip", compression="gzip")
    
    if args.cl:
        logger.info("Loading master data...")
        master_data = (
            pd.read_parquet("data/master_data_p1.parquet.gzip")
            .sample(frac=1, random_state=1910)
            .reset_index(drop=True)
        )
        try:
            previous_data = pd.read_parquet("data/proccessed_data_p1.parquet.gzip")
            proccessed_articles = previous_data.id.to_list()
        except FileNotFoundError:
            previous_data = None
            proccessed_articles = []
        
        proccessed_data = process_data(
            df = master_data,
            api_key = os.getenv("OPENAI_API_KEY"),
            excluded_articles = proccessed_articles
        )

        if previous_data is not None:
            updated_processed_data = pd.concat([previous_data, proccessed_data], ignore_index=True)
            updated_processed_data.to_parquet("data/proccessed_data_p1.parquet.gzip", compression="gzip")
        else:
            proccessed_data.to_parquet("data/proccessed_data_p1.parquet.gzip", compression="gzip")

    if args.lda:
        logger.info("Loading and processing data...")
        proccessed_data = pd.read_parquet("data/proccessed_data_p1.parquet.gzip")
        ji_related_data = proccessed_data[proccessed_data["status"] == "TRUE"]
        proccessed_data = newsData(
            data = ji_related_data,
            target_col = "content_trans"
        )
        logger.info("Fitting a LDA model...")
        proccessed_data.train_lda(ntopics = 5)
        logger.info("LDA results saved in /data/lda.html")

    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--data", action="store_true", help="Run data loading")
    parser.add_argument("--cl", action="store_true", help="Run data classification")
    parser.add_argument("--lda", action="store_true", help="Train an LDA model and save the visualization")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run ONLY the data pipeline
    if not any(vars(args).values()):
        args.data = True
    
    main(args)