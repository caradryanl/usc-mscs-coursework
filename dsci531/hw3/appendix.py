import os
import argparse

def convert_to_txt(directories):
    """
    Convert all files in the given directories to .txt by changing their extension.
    No safety checks, no content analysis.
    """
    total_converted = 0
    
    for directory in directories:
        print(f"Processing directory: {directory}")
        
        # Walk through all files in the directory
        for root, _, files in os.walk(directory):
            for file in files:
                # Skip files that are already .txt
                if file.endswith('.txt'):
                    continue
                    
                file_path = os.path.join(root, file)
                # Get the file path without extension
                file_base = os.path.splitext(file_path)[0]
                # Create new path with .txt extension
                new_path = f"{file_base}.txt"
                
                # Rename the file
                os.rename(file_path, new_path)
                print(f"Converted: {file_path} → {new_path}")
                total_converted += 1
    
    print(f"\nTotal files converted to .txt: {total_converted}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert all files in directories to .txt extension')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    
    args = parser.parse_args()
    
    # Process directories
    convert_to_txt(args.directories)

if __name__ == "__main__":
    main()
    # python appendix.py directories data_processed/Template_2/ data_processed/Template_3/ data_processed/Template_4/ data_processed/Template_5/ data_processed/Template_6/ data_processed/Template_7/ data_processed/Template_8/ data_processed/Template_9/ 
