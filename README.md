# Periodontitis Prediction using Generative AI and Machine Learning

This repository contains the supplementary data and code for the research paper:

> **Harnessing Generative Artificial Intelligence for Periodontitis Prediction: A Machine Learning Approach Integrating Systemic Health Indicators for Precision Oral Health in Resource-Limited Settings**
>
> Argajit Sarkar, Epshita Ghosh, Surajit Bhattacharjee
>
> Department of Molecular Biology and Bioinformatics, Tripura University (A Central University), Agartala 799022, Tripura, India

## Overview

This research explores the novel application of Generative Artificial Intelligence (GenAI), specifically ChatGPT-4o, to enhance periodontitis prediction by linking systemic health indicators to oral disease progression. The study demonstrates that GenAI-built machine learning models can effectively identify relationships between systemic health factors and periodontal disease.

## Key Findings

- Logistic regression achieved 72% accuracy (95% CI: 62-81%)
- Support Vector Machine demonstrated significantly superior recall (89%, 95% CI: 81-94%)
- Feature importance analysis identified age (0.233) and blood sugar (0.209) as the strongest predictors
- Significant associations were found between systolic and diastolic blood pressure (r=0.54, p<0.001)

## Repository Contents

- `scripts/`: Contains the full analysis script and utility functions
  - `periodontitis-code.py`: Complete analysis pipeline with data preprocessing, EDA, and modeling
  - `wilson_score_intervals.py`: Script for calculating confidence intervals for model performance metrics
- `requirements.txt`: Required Python packages to run the code

## Usage

### Installation

```bash
git clone https://github.com/username/periodontitis-genai-prediction.git
cd periodontitis-genai-prediction
pip install -r requirements.txt
```

### Running the Analysis

The main analysis can be executed using:

```bash
python scripts/periodontitis-code.py
```

Note: This requires the input dataset file `Dental_Data_18.02.25.xlsx` which is not included in this repository due to privacy considerations. Please contact the authors for access to the dataset.

## Ethics Statement

This study was conducted under the ethical standards outlined in the Declaration of Helsinki for medical research involving human subjects. Ethics approval was granted by the District Hospital, Unakoti, Kailashahar, Tripura, India, on March 06, 2025 (No.F.8(61-Estt. (G)/MS/UDH/KLS/23/559). All patient data has been comprehensively de-identified.

## Citation

If you use this code or methodology in your research, please cite our paper:

```
@article{sarkar2025harnessing,
  title={Harnessing Generative Artificial Intelligence for Periodontitis Prediction: A Machine Learning Approach Integrating Systemic Health Indicators for Precision Oral Health in Resource-Limited Settings},
  author={Sarkar, Argajit and Ghosh, Epshita and Bhattacharjee, Surajit},
  journal={[Journal Name]},
  year={2025},
  volume={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or further information, please contact:
- Argajit Sarkar: argajit05@gmail.com
