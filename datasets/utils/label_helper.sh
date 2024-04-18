#!/bin/bash

# Function to get a random file from the current directory
get_random_file() {
    find . -type f -maxdepth 1 -print0 | shuf -z -n1
}

# Function to extract number from filename
extract_number() {
    local filename=$1
    # Extract the number based on the expected filename format: img_0033_0006_{here}_1.png
    [[ $filename =~ img_[0-9]+_[0-9]+_([0-1])_1\.png ]]
    echo "${BASH_REMATCH[1]}"
}

# Function to move a percentage of files from one directory to another
move_percentage() {
    local src_dir=$1
    local dest_dir=$2
    local percentage=$3

    # Calculate the total number of files
    total_files=$(find "$src_dir" -type f | wc -l)
    # Calculate the number of files to move
    files_to_move=$((total_files * percentage / 100))

    # Move the calculated number of files
    find "$src_dir" -type f | shuf | head -n $files_to_move | xargs -I{} mv {} "$dest_dir"
}

# Create directories if they do not exist
mkdir -p RIGHT WRONG

# Main loop
while true; do
    # Get a random file
    file=$(get_random_file)
    
    # Check if the file variable is empty
    if [[ -z "$file" ]]; then
        echo "No more files in the directory."
        break
    fi
    
    # Extract number
    number=$(extract_number "$(basename "$file")")
    echo "Extracted Number: $number"
    
    # Display the file using imgcat
    imgcat "$file"
    
    # Ask the user if the input matches the number
    read -p "Does this photo have pneumonia? (yes=1 / no=0): " user_input
    
    # Move the file to the correct directory based on user input
    if [[ "$user_input" == "$number" ]]; then
        echo "Moving to RIGHT directory."
        mv "$file" RIGHT/
    else
        echo "Moving to WRONG directory."
        mv "$file" WRONG/
    fi
    
    # Check if there are any files left
    if [ $(find . -maxdepth 1 -type f | grep -vcE 'RIGHT|WRONG') -eq 0 ]; then
        echo "No more files to process in the main directory."
        break
    fi
done


# Ensure the train directories exist
mkdir -p "../maindata/train/RIGHT" "../maindata/train/WRONG"
# Ensure the test directories exist
mkdir -p "../maindata/test/RIGHT" "../maindata/test/WRONG"

cp ./WRONG/* ../maindata/train/WRONG/
cp ./RIGHT/* ../maindata/train/RIGHT/

# Move 30% of the files from train to test directories
move_percentage "../maindata/train/RIGHT" "../maindata/test/RIGHT" 30
move_percentage "../maindata/train/WRONG" "../maindata/test/WRONG" 30

echo "Moved 30% of files from train to test directories."
