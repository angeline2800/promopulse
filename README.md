# PromoPulse: NLP-Driven Sales Nowcasting Dashboard

PromoPulse is a lightweight Streamlit dashboard developed to present the results of an NLP-driven short-horizon sales nowcasting project. The dashboard focuses on comparing forecasting performance between **HistoryOnly** and **History+NLP** settings, while also providing a simple NLP signal exploration view.

## Features

The dashboard contains three main views:

### 1. Forecast Summary
- Shows the best-performing model and feature setting
- Displays overall forecasting results
- Displays forecasting results by horizon
- Includes an sMAPE comparison chart for the selected model

### 2. History vs NLP Analysis
- Compares **HistoryOnly** and **History+NLP** forecasting performance
- Shows aggregate comparison results using sMAPE
- Includes a store-level forecast demo using **RandomForest**
- Visualises **Actual Sales**, **HistoryOnly Prediction**, and **History+NLP Prediction**

### 3. NLP Signal Explorer
- Allows the user to enter a sample review
- Displays cleaned text
- Shows a simple sentiment result
- Detects promo-related and complaint-related cues

## Project Purpose

This dashboard was developed as part of the project:

**PromoPulse: NLP-Driven Sales Nowcasting**

The aim of the project is to examine whether customer-voice signals extracted from BM/EN code-switched reviews can provide useful auxiliary information for short-horizon sales forecasting.

## Files Required

Place the following files in the same folder as `app.py`:

- `summary_overall.csv`
- `summary_by_horizon.csv`
- `compare_pivot.csv`
- `forecast_detailed_predictions_with_nlp.csv`

## Installation

Make sure Python is installed, then install the required packages:

```bash
pip install streamlit pandas matplotlib

How to Run

Open a terminal in the project folder and run:
streamlit run app.py
