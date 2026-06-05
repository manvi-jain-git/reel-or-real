# Reel or Real 

An AI-powered web application that analyzes social media reels and estimates whether the content is AI-generated or human-created.
<img width="2558" height="1153" alt="image" src="https://github.com/user-attachments/assets/f1f30c71-421a-456c-b4c9-74bd3db60613" />



## Overview

The rapid growth of generative AI has made it increasingly difficult to distinguish between authentic and AI-generated content. **Reel or Real** is an experimental web application designed to analyze video reels and provide an estimate of their likelihood of being AI-generated.

The project combines multiple analysis techniques, including visual heuristics, facial analysis, and classification methods, to evaluate uploaded video content and generate an overall authenticity score.
<img width="2504" height="1391" alt="image" src="https://github.com/user-attachments/assets/d17bac2e-3e44-4734-9e61-60fb806332d9" />

## Features

* Upload and analyze video reels
* AI-generated vs human-created probability scoring
* Facial feature analysis
* Visual consistency and heuristic checks
* Modular analysis pipeline
* Simple and user-friendly web interface
* Real-time result generation
  <img width="2504" height="1337" alt="image" src="https://github.com/user-attachments/assets/bf36e464-18d7-485e-95dd-088745b8d3ca" />


## Tech Stack

### Backend

* Python
* Flask

### Frontend

* HTML
* CSS
* JavaScript

### Video Analysis

* Computer Vision Techniques
* Feature Extraction
* Heuristic Scoring
* Classification Models

## Project Structure

```text
reel-or-real/
│
├── analyzer/
│   ├── clip_classifier.py
│   ├── downloader.py
│   ├── extractor.py
│   ├── face_analysis.py
│   ├── heuristics.py
│   └── scorer.py
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

## How It Works

1. User submits a reel or video.
2. The application extracts relevant visual information.
3. Multiple analysis modules evaluate the content.
4. Individual scores are combined into a final authenticity assessment.
5. Results are displayed through the web interface.

## Installation

Clone the repository:

```bash
git clone https://github.com/manvi-jain-git/reel-or-real.git
cd reel-or-real
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

## Future Improvements

* Deep learning-based video classification
* Frame-by-frame temporal analysis
* Audio authenticity detection
* Confidence visualization dashboard
* Support for multiple social media platforms
* Improved model accuracy and explainability

## Learning Outcomes

Through this project, I gained practical experience in:

* Python application development
* Flask web development
* Computer vision workflows
* Video processing pipelines
* Modular software design
* Git and GitHub version control

## Disclaimer

This project is intended for educational and research purposes. AI-generated content detection remains an evolving challenge, and results should be interpreted as estimates rather than definitive conclusions.

## Author

**Manvi Jain**

B.Tech CSE (AI & ML)
Manav Rachna University

If you found this project interesting, feel free to star the repository and share feedback.
