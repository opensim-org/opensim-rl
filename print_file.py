#!/usr/bin/env python3

import argparse
import os
import sys

def create_blank_file(directory, filename="blank_file.txt"):
    """Create a blank file in the specified directory."""
    try:
        # Ensure the directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return False

        # Create the full file path
        file_path = os.path.join(directory, filename)

        # Create the blank file
        with open(file_path, 'w') as f:
            pass

        print(f"Created blank file: {file_path}")
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create a blank file in a specified directory.")
    parser.add_argument("directory", help="Directory where the blank file will be created")
    parser.add_argument("--filename", "-f", default="blank_file.txt",
                        help="Name of the blank file (default: blank_file.txt)")

    args = parser.parse_args()

    success = create_blank_file(args.directory, args.filename)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()