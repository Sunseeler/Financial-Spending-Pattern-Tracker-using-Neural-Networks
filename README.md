# Spending Analysis Neural Network

A Python-based machine learning tool that analyzes spending patterns from ODS spreadsheets using neural networks. The system processes historical spending data and creates predictions using TensorFlow/Keras.

## Features

- ODS file format support
- Data preprocessing and restructuring
- Neural network-based analysis
- Spending pattern prediction
- Automated data normalization
- Date-based analysis

## Prerequisites

- Python 3.7+
- Required packages:
```bash
pip install numpy pandas ezodf sklearn tensorflow
```

## Data Format Requirements

Input file (`spending.ods`) should contain:
- Date columns (format: MM/DD/YY)
- Amount columns (format: $XXX.XX)
- Groups of 3 columns (Date, Amount, Empty)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place your `spending.ods` file in the project directory

## Neural Network Architecture

- Input Layer: 2 neurons (day of year, amount)
- Hidden Layers:
  - Dense layer (64 neurons, ReLU)
  - Dense layer (32 neurons, ReLU)
  - Dense layer (16 neurons, ReLU)
- Output Layer: 1 neuron (prediction)

## Data Processing

1. File Reading
   - Reads ODS file
   - Extracts date and amount columns
   - Handles missing data

2. Data Transformation
   - Converts dates to day of year
   - Normalizes amount values
   - Removes invalid entries

3. Data Preparation
   - Feature scaling
   - Train/test split (80/20)
   - Validation set creation

## Model Training

- Optimizer: Adam
- Loss function: Mean Squared Error
- Epochs: 100
- Batch size: 32
- Validation split: 20%

## Error Handling

- ODS file reading errors
- Date parsing failures
- Amount conversion issues
- Missing data handling

## Output

- Training progress information
- Model evaluation metrics
- Prediction results
- Error logs

## Usage

Run the script:
```bash
python spending_analysis.py
```

## Limitations

- Requires specific ODS format
- Sensitive to data quality
- Requires sufficient historical data
- Memory usage scales with dataset size

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]

## Acknowledgments

- TensorFlow team
- scikit-learn contributors
- ezodf developers
