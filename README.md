# Footbal Games predcition

## Objective
The objective of analyzing this dataset is to identify patterns that could be useful in pricing bets on goals in football games. The long-term goal is to determine the probability of there being a certain number of goals in a game given specific conditions.

## Analytical Approaches
- **Exploratory Data Analysis (EDA)**: Conduct thorough exploration of the dataset to understand distributions, trends, and relationships between variables.
- **Statistical Analysis**: Utilize statistical methods to calculate probabilities and assess correlations between different factors and goal outcomes.
- **Data Visualization**: Create human-readable graphs, charts, and tables to visually represent key findings and insights.
- **Machine Learning**: Implement predictive models to forecast goal probabilities based on various game conditions.
- **Documentation**: Prepare a comprehensive report documenting your findings, methodologies, and recommendations.

## Getting Started

To get started with the project using this template repository, follow these steps:

2. **Clone the Repository**: Clone the created repository to your local machine using Git.

   ```bash
   git clone https://github.com/rashidis/game-goals-regression.git
3. **Navigate to the Project Directory**: Enter the project directory in your terminal or command prompt.
4. **Install Dependencies**: Create the conda environment with dependencies installed:

   ```bash
   conda env create -f environment.yml
5. **Activate the conda environment**:

   ```bash
   conda activate goals-prediction-env
6. **Run code**
   ```bash
   python src/transformation.py
   python src/train_and_evaluation.py
7. **Configurable parameters**
   parameters can be configured in data_config.json and model_config.json files

## Directory Structure

The repository is organized as follows:

- **`data/`:** Contains the dataset used for training and evaluation, as well as the transformed data files.
- **`models/`:** Contains the trained models.
- **`notebooks/`:** Jupyter notebooks detailing the data exploration, preprocessing, and model training processes.
- **`src/`:** Python scripts for modularized code, including data preprocessing, feature engineering, and model training.
- **`results/`:** Stores the results of the predictive models.
- **`README.md`:** Project overview and usage
- **`LICENSE`:** License file

## License

This project is licensed under the [MIT License](License). Feel free to use, modify, and distribute this template for your AI/ML projects.