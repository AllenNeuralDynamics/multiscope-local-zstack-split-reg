import logging
from pathlib import Path
import argparse
import os
from time import time
from .zstack_processing import process_zstack_paths

logger = logging.getLogger(__name__)

def main():
    start_time = time()
    parser = argparse.ArgumentParser(description='Process zstack files in parallel')
    parser.add_argument('input_dir', type=str, help='Directory containing zstack files')
    parser.add_argument('output_dir', type=str, help='Directory to save processed files')
    parser.add_argument('--pattern', type=str, default='*_local_z_stack*.tif*', 
                       help='File pattern to match (default: *_local_z_stack*.tif*)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    zstack_paths = list(input_dir.glob(args.pattern))
    
    if not zstack_paths:
        print(f"No files matching pattern '{args.pattern}' found in {input_dir}")
        return
    
    # Process files
    process_zstack_paths(zstack_paths, output_dir)
    
    print("Processing complete!")
    print(f"Total time taken: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()