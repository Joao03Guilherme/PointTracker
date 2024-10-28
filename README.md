# PointTracker
This code was used for the Advaced Physics Laboratory II curricular unit 2024/2025

## Requirements
Clone the repository using:
```bash
git clone https://github.com/Joao03Guilherme/PointTracker.git
```
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the program, run the following command:
```bash
python3 main.py <video_file> --save_plot <plot_file> --save_data <data_file>
```

The program will read the video file and track the points in the video. The results will be displayed in a new window. The results will be saved to the specified files if the `--save_plot` and `--save_data` arguments are provided, otherwise the results will not be saved and only shown.

An image of the first frame will appear and you will be asked for the following inputs:
1. Click on the image to select the origin point (red).
2. Click to add two scale points (green).
3. Click and drag the scale points to adjust their positions.
4. Press 'Enter' to proceed after you have selected and adjusted all points.
5. In the terminal, you will be prompted to insert the real distance between the two selected points for scale (in cm).

The program will then track the points in the video and display the results in a new window. If the `--save_data` option was used, a csv file will also be saved with all the tracking data. If the `--save_plot`option was used, a png file of the plot will be saved.
