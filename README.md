# DiabetesGuard Pro 🏥

A modern web application for diabetes risk prediction and health analytics, built with Streamlit and scikit-learn.

## Features 🌟

- **Risk Prediction**: Get personalized diabetes risk assessment
- **Health Analytics**: Visualize health metrics and trends
- **Smart Recommendations**: Receive tailored health recommendations
- **Interactive Dashboard**: Explore data through dynamic visualizations

## Tech Stack 💻

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: scikit-learn
- **Data Analysis**: pandas, numpy
- **Visualization**: plotly

## Project Structure 📁

```
diabetes-guard-pro/
├── app/                    # Application components
│   ├── components/         # UI components
│   │   ├── analytics.py   # Analytics dashboard
│   │   ├── home.py        # Home page
│   │   └── predict.py     # Prediction interface
│   └── utils/             # Utility functions
│       ├── data_loader.py # Data loading utilities
│       └── model_handler.py# Model management
├── data/                  # Dataset directory
├── models/               # Trained model files
├── notebooks/           # Development notebooks
├── plots/               # Generated visualizations
├── diabetes_app.py      # Main application
├── verify_and_train.py  # Model training script
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-guard-pro.git
cd diabetes-guard-pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python verify_and_train.py
```

4. Run the application:
```bash
streamlit run diabetes_app.py
```

## Usage 📱

1. **Home Page**: Overview of the application and its features
2. **Predict**: Enter your health metrics to get a diabetes risk assessment
3. **Analytics**: Explore health data trends and visualizations

## Model Details 🤖

- **Algorithm**: Random Forest Classifier
- **Features**: Age, Gender, BMI, Blood Pressure, Glucose Level, Exercise Hours, Smoking Status, Alcohol Consumption, Stress Level
- **Performance**: 
  - Training Accuracy: 73.18%
  - Testing Accuracy: 50.35%

## Contributing 🤝

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Author ✨

Created by Fahad

## Acknowledgments 🙏

- Streamlit for the amazing framework
- scikit-learn for machine learning tools
- The open-source community for inspiration
