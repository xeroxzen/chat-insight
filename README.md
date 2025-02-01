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

## Project Structure

```plaintext
ChatInsight/
├── __init__.py
├── app.py
├── parser.py
├── file_handler.py
├── analyzer.py
├── templates/
├── static/
├── tests/
├── Dockerfile          
├── requirements.txt
└── README.md
```

### Key Components

- `Dockerfile`: Defines the container configuration:
  - Uses Python 3.12 slim image
  - Sets up the working directory at `/app`
  - Installs project dependencies
  - Exposes port 8000 
  - Configures the FastAPI application to run with uvicorn
- `app/__init__.py`: Initializes the FastAPI application.
- `app/app.py`: Entry point for the FastAPI application.
- `app/routes.py`: Defines the routes for file upload and analysis results.
- `app/parser.py`: Contains utility functions for file handling and parsing.
- `app/file_handler.py`: Contains utility function for handling file uploads.
- `app/analyzer.py`: Contains the analysis functions for processing chat data.
- `app/templates/`: Contains Jinja2 templates for rendering HTML pages.
- `app/static/`: Contains static files such as CSS and JavaScript.
- `tests/test_app.py`: Contains test cases for the application.
- `tests/test_analyzer.py`: Contains test cases for the analysis functions.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Documentation for the project.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Make sure to follow the contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or support, please contact:
- **Name**: Andile Mbele
- **Email**: andilembele020@gmail.com
- **GitHub**: [xeroxzen](https://github.com/xeroxzen)

---