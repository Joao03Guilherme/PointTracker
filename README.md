# PointTracker

## Requirements
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the program, run the following command:
```bash
python3 main.py <video_file> --save_plot <plot_file> --save_data <data_file>
```

The program will read the video file and track the points in the video. The results will be displayed in a new window. The results will be saved to the specified files if the `--save_plot` and `--save_data` arguments are provided, otherwise the results will not be saved.

An image of the first frame will appear and you will be asked for the following inputs:
1. Click on the image to select the origin point (red).
2. Click to add two scale points (green).
3. Click and drag the scale points to adjust their positions.
4. Press 'Enter' to proceed after you have selected and adjusted all points.

The program will then track the points in the video and display the results in a new window. 
