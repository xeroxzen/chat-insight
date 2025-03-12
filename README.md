<div align="center">
  <img src="https://img.shields.io/badge/ChatInsight-WhatsApp_Analytics_Platform-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" alt="ChatInsight" width="600">
  <br>
  <br>
  <p>
    <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status">
    <img src="https://img.shields.io/badge/Version-1.0.0-blue?style=flat-square" alt="Version">
    <img src="https://img.shields.io/badge/License-Apache_2.0-orange?style=flat-square" alt="License">
  </p>
</div>

**ChatInsight** is a web application designed to analyze WhatsApp chat data and provide detailed insights through visualizations. This tool allows users to upload their WhatsApp chat export files and instantly receive analysis results, including message counts, active days and times, commonly used words, and more.

## Features

- **Simple File Upload**: Users can upload their WhatsApp chat export zip files directly to the app.
- **Automated Processing**: The app automatically extracts, parses, and converts the chat data into a CSV file.
- **Comprehensive Analysis**: Detailed analysis of individual chats, including:
  - Message counts per participant
  - Most active days
  - Most active times of day
  - Most commonly used words
  - Sentiment analysis
  - Emoji usage patterns
  - Response time analysis
  - Conversation activity trends
- **Visualizations**: Generate visual insights such as bar plots, line plots, and word clouds.
- **User-Friendly Interface**: Easy-to-navigate web interface powered by FastAPI and Jinja2 templates.
- **Security Features**: Includes session management, rate limiting, and security headers.
- **Kubernetes Support**: Ready for deployment in Kubernetes environments.

## Tech Stack

<div align="center">

### Core Technologies

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.30.1-4EAA25?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)
[![Jinja2](https://img.shields.io/badge/Jinja2-3.1.4-B41717?style=for-the-badge&logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)

### Data Processing & Analysis

[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.9.1-154f5b?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

### Visualization

[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.0-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![WordCloud](https://img.shields.io/badge/WordCloud-1.9.3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/amueller/word_cloud)

### Deployment & Infrastructure

[![Docker](https://img.shields.io/badge/Docker-24.0.7-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.29-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)

</div>

## Screenshots & Demo

<div align="center">
  <p><strong>📊 Interactive Dashboard</strong></p>
  <img src="static/images/screenshots/dashboard.png" alt="Dashboard Screenshot" width="80%">
  <p><em>Dashboard showing message statistics and user activity</em></p>
  
  <br>
  
  <p><strong>📈 Analysis Results</strong></p>
  <img src="static/images/screenshots/analytics.png" alt="Analysis Results" width="80%">
  <p><em>Detailed analytics showing conversation patterns</em></p>
  
  <br>
  
  <p>
    <a href="https://chatinsights.app">
      <img src="https://img.shields.io/badge/Live_Demo-View_Application-blue?style=for-the-badge" alt="Chat Insight">
    </a>
  </p>
</div>

## Installation & Setup

### Option 1: Using Docker (Recommended)

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system

#### Quick Start

```bash
git clone git@github.com:xeroxzen/chat-insight.git
cd chat-insight

# Build and run the Docker container
docker build -t chat-insight .
docker run -p 8000:8000 chat-insight
```

The application will be available at `http://localhost:8000`

### Option 2: Local Installation (Alternative)

If you prefer not to use Docker, you can still run the application locally:

```bash
# Clone the repository
git clone git@github.com:xeroxzen/chat-insight.git
cd chat-insight

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Option 3: Kubernetes Deployment

For production environments, you can deploy using Kubernetes:

```bash
kubectl apply -f k8s/chatinsights.yaml
```

## Project Structure

```plaintext
ChatInsight/
├── __init__.py
├── app.py                # Main FastAPI application
├── parser.py             # Chat parsing utilities
├── file_handler.py       # File upload handling
├── analyzer.py           # Chat analysis functions
├── session_manager.py    # Session management
├── rate_limiter.py       # API rate limiting
├── templates/            # Jinja2 templates
├── static/               # CSS, JS, and other static files
├── uploads/              # Temporary storage for uploads
├── tests/                # Test cases
├── k8s/                  # Kubernetes configuration
│   └── chatinsights.yaml # K8s deployment manifest
├── .github/              # GitHub workflows and templates
├── Dockerfile            # Container configuration
├── requirements.txt      # Project dependencies
├── pytest.ini           # PyTest configuration
├── conftest.py          # PyTest fixtures
├── CODE_OF_CONDUCT.md   # Community guidelines
├── CONTRIBUTORS.md      # Contributor information
└── README.md            # Project documentation
```

### Key Components

- `app.py`: Entry point for the FastAPI application, defines routes and middleware.
- `parser.py`: Contains utility functions for parsing WhatsApp chat exports.
- `file_handler.py`: Handles file uploads, validation, and storage.
- `analyzer.py`: Contains the analysis functions for processing chat data.
- `session_manager.py`: Manages user sessions and security.
- `rate_limiter.py`: Implements rate limiting for API endpoints.
- `Dockerfile`: Defines the container configuration:
  - Uses Python 3.11 slim image
  - Sets up the working directory at `/app`
  - Installs project dependencies
  - Exposes port 8000
  - Configures the FastAPI application to run with uvicorn
- `templates/`: Contains Jinja2 templates for rendering HTML pages.
- `static/`: Contains static files such as CSS and JavaScript.
- `k8s/`: Contains Kubernetes deployment configurations.
- `tests/`: Contains test cases for the application.
- `requirements.txt`: Lists the dependencies required for the project.

## Contributing

Contributions are welcome! Please see [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed guidelines on how to contribute to this project. All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Adding Screenshots

When adding new screenshots to the documentation:

1. Save your screenshots in the `static/images/screenshots/` directory
2. Use descriptive filenames (e.g., `feature-name.png`)
3. Optimize images for web (compress if necessary)
4. Update the README.md in the screenshots directory
5. Reference the images in the main README.md using the correct path

```markdown
![Description](static/images/screenshots/filename.png)
```

## License

This project is licensed under the Apache License 2.0. See the [Apache License 2.0](Apache%20License%202.0) file for more details.

## Contact

<div align="center">

### 👨‍💻 Andile Mbele

[![GitHub](https://img.shields.io/badge/GitHub-xeroxzen-181717?style=for-the-badge&logo=github)](https://github.com/xeroxzen)
[![Email](https://img.shields.io/badge/Email-andilembele020%40gmail.com-D14836?style=for-the-badge&logo=gmail)](mailto:andilembele020@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Andile_Mbele-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/andile-jaden-mbele/)
[![Twitter](https://img.shields.io/badge/Twitter-@xeroxzen-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/xeroxzen)

</div>

---

<div align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love">
  <img src="https://img.shields.io/badge/WhatsApp-Analysis-25D366?logo=whatsapp" alt="WhatsApp Analysis">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python" alt="Python 3.11">
</div>

---
