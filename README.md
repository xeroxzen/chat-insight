Sure! Here's a comprehensive documentation for the ChatInsight project that you can include in your GitHub repository:

---

# ChatInsight

**ChatInsight** is a web application designed to analyze WhatsApp chat data and provide detailed insights through visualizations. This tool allows users to upload their WhatsApp chat export files and instantly receive analysis results, including message counts, active days and times, commonly used words, and more.

## Features

- **Simple File Upload**: Users can upload their WhatsApp chat export zip files directly to the app.
- **Automated Processing**: The app automatically extracts, parses, and converts the chat data into a CSV file.
- **Comprehensive Analysis**: Detailed analysis of individual chats, including:
  - Message counts per participant
  - Most active days
  - Most active times of day
  - Most commonly used words
- **Visualizations**: Generate visual insights such as bar plots, line plots, and word clouds.
- **User-Friendly Interface**: Easy-to-navigate web interface powered by FastAPI and Jinja2 templates.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository

```bash
git clone git@github.com:xeroxzen/chat-insight.git
cd chat-insight
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
uvicorn app:app --reload
```

### Accessing the Web Interface

- Open your web browser and go to `http://127.0.0.1:8000/`.
- Upload your WhatsApp chat export zip file.
- View the detailed analysis results and visualizations.

## Project Structure

```plaintext
ChatInsight/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── routes.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── results.html
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── utils.py
│   └── analysis.py
├── tests/
│   ├── test_app.py
│   └── test_analysis.py
├── requirements.txt
└── README.md
```

### Key Files and Directories

- `app/__init__.py`: Initializes the FastAPI application.
- `app/app.py`: Entry point for the FastAPI application.
- `app/models.py`: Defines data models for the application.
- `app/routes.py`: Defines the routes for file upload and analysis results.
- `app/utils.py`: Contains utility functions for file handling and parsing.
- `app/analysis.py`: Contains the analysis functions for processing chat data.
- `app/templates/`: Contains Jinja2 templates for rendering HTML pages.
- `app/static/`: Contains static files such as CSS and JavaScript.
- `tests/test_app.py`: Contains test cases for the application.
- `tests/test_analysis.py`: Contains test cases for the analysis functions.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Documentation for the project.

## Development

### Adding New Features

1. **Fork the Repository**: Create a fork of the repository on GitHub.
2. **Create a Branch**: Create a new branch for your feature.
   ```bash
   git checkout -b feature-name
   ```
3. **Implement the Feature**: Add your code and tests for the new feature.
4. **Commit and Push**: Commit your changes and push them to your fork.
   ```bash
   git commit -m "Add feature-name"
   git push origin feature-name
   ```
5. **Create a Pull Request**: Open a pull request on GitHub.

### Running Tests

```bash
pytest tests/
```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Make sure to follow the contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or support, please contact:
- **Name**: Andile
- **Email**: andilembele020@gmail.com
- **GitHub**: [xeroxzen](https://github.com/xeroxzen)

---