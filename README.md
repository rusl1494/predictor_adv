## Description
Python-based crypto price analysis and prediction tool.

The project collects historical market data, prepares features, trains a model, and generates price predictions and reports.  
Designed as an automated analysis tool with scheduled execution.

## Features
- Historical data processing
- Feature preparation and scaling
- Model training and prediction
- Daily prediction reports
- CSV and JSON outputs
- Suitable for automation (cron jobs)

## Project Structure
- `prepare_data.py` – data preparation and feature engineering  
- `train_lstm_model.py` – model training  
- `predictor_adv.py` – price prediction and reporting  
- `update_data.py` – data updates  
- Output files: CSV, JSON, logs

## Tech Stack
Python, Pandas, NumPy, TensorFlow/Keras, SQLite

## Notes
This project is provided for demonstration and portfolio purposes.
