"""
Dual Logger & Output Stream Manager
====================================
Single Responsibility Principle (SRP):
Dedicated exclusively to stream multiplexing (Tee-style logging) between standard terminal
and persistent log files.
"""

import os
import sys
from typing import TextIO, Optional


class DualLogger:
    """
    Tee-style logger that writes output streams concurrently to standard terminal (stdout)
    and a persistent log file (e.g., checkpoints/search_execution.log or report/pipeline.log).
    """
    def __init__(self, filepath: str):
        self.terminal: TextIO = sys.stdout
        self.filepath: str = filepath
        
        log_dir = os.path.dirname(filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        self.log_file: Optional[TextIO] = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self) -> None:
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
