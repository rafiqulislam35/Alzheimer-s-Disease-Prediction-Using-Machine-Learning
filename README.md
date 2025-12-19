![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

# Alzheimer’s Disease Prediction Using Machine Learning

A machine learning–based web application for early detection of Alzheimer’s disease using patient clinical data and cognitive test results.

The system provides:
- Alzheimer’s risk prediction using a trained ML model
- Secure user authentication (Admin & User roles)
- Patient management and prediction history
- An AI-powered chatbot for Alzheimer’s-related queries


## Technologies Used
- **Backend:** FastAPI
- **Machine Learning:** Scikit-learn (Random Forest)
- **Database:** MongoDB
- **Frontend:** HTML, CSS, Jinja2
- **Authentication:** JWT, Passlib
- **Chatbot & Retrieval:** LangChain, FAISS, Ollama
- **Testing:** PyTest


## How to Run the Project

### 1- Install dependencies

pip install -r requirements.txt


##  Unit Testing

The login authentication module was unit tested using PyTest to verify correctness, reliability, and role-based access control.

### Covered Test Scenarios
- Login page loading
- Invalid credential handling
- Disabled user access restriction
- Successful login for normal users
- Role-based redirection (Admin vs User)
- Authentication cookie generation

### Testing Approach
- Unit tests are located in the `tests/` directory
- External dependencies such as the machine learning model, FAISS vector store, and LLM-based chatbot components were mocked to isolate authentication logic
- FastAPI’s `TestClient` was used to simulate HTTP requests

### Run Tests
python -m pytest tests/test_login.py


## 2️ Start the server
python -m uvicorn app:app --reload
