# UPI Fraud Detection Streamlit App

This project is a Streamlit application designed for detecting fraudulent UPI transactions using machine learning models. The application allows users to explore transaction data, visualize insights, and make predictions based on trained models.

## Project Structure

```
upi_fraud_streamlit_app
├── src
│   ├── app.py               # Main entry point for the Streamlit application
│   ├── utils.py             # Utility functions for data loading, preprocessing, and model training
│   ├── models
│   │   ├── __init__.py      # Marks the models directory as a package
│   │   └── train.py         # Responsible for training machine learning models
│   ├── pages
│   │   ├── home.py          # Home page of the Streamlit app
│   │   └── explore.py       # Exploratory data analysis page
│   └── components
│       └── __init__.py      # Marks the components directory as a package
├── data
│   └── upi_transactions_2025.csv  # Dataset containing transaction data
├── requirements.txt          # Python dependencies required for the project
├── .gitignore                # Files and directories to be ignored by Git
├── Procfile                  # Deployment configuration for platforms like Heroku
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd upi_fraud_streamlit_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application locally:
   ```
   streamlit run src/app.py
   ```

## Deployment Instructions

To deploy the Streamlit app, follow these steps:

1. Ensure all dependencies are listed in `requirements.txt`.
2. In `Procfile`, add the line:
   ```
   web: streamlit run src/app.py
   ```
3. Push your code to a Git repository.
4. Deploy to a platform that supports Streamlit, such as Heroku or Streamlit Sharing, following their specific deployment instructions.

## Usage

Once the application is running, you can navigate through the home page to view summaries and visualizations of the data. The explore page allows for interactive data analysis, and you can make predictions based on the trained models.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.