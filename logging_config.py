#!/usr/bin/env python3

import logging

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        level (int): Logging level (default is logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler('app.log'),  # Optional: Output to a file
        ]
    )

