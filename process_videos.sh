#!/bin/bash

# Loop over each folder matching the pattern HxAyBz
for folder in H*A*B*; do
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"

        # Define the input and output paths
        input_file="$folder/data.avi"
        output_folder="${folder}_ANALYSED"

        # Create the output folder if it doesn't exist
        mkdir -p "$output_folder"

        # Run the Python script with the specified arguments
        python3 main.py "$input_file" \
            --save_plot "$output_folder/plot.png" \
            --save_data "$output_folder/tracking_data.csv"

        echo "Finished processing $folder. Results saved in $output_folder."
    else
        echo "Skipping $folder as it is not a directory."
    fi
done
