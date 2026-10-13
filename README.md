# Interactive VGG Research Dashboard

A Streamlit dashboard for comparing VGG-11, VGG-13 and VGG-16 on CIFAR-10, with and without Batch Normalization.

## Features
- Interactive architecture and BatchNorm filters
- Results table and train/validation/test accuracy comparison
- Model parameter-count vs. accuracy plot
- Optional training-history line chart
- Download filtered results as CSV

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses the recorded final five-epoch results by default. You can upload your own result CSV and `final_training_history.csv` in the sidebar.

**Important:** Filters change the displayed analysis, not the trained model. The dashboard does not retrain models or classify new images. The current results are from one five-epoch run per configuration and should not be treated as universal evidence that BatchNorm is worse.
