import requests
import json  # Included for potential data handling, although not strictly needed for requests.post

# Base URL for the local FastAPI server
HOST = "http://127.0.0.1:8000"


def test_api():
    # Data to be sent in the POST request
    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    print("--- 1. Testing GET / ---")
    # TODO: send a GET using the URL http://127.0.0.1:8000
    r = requests.get(HOST)  # Sent GET request
    # TODO: print the status code
    print(f"Status Code: {r.status_code}")
    # TODO: print the welcome message
    print(f"Welcome Message: {r.json()}")

    print("\n--- 2. Testing POST /data/ ---")
    # TODO: send a POST using the data above
    r = requests.post(f"{HOST}/data/", json=data)  # Sent POST request
    # TODO: print the status code
    print(f"Status Code: {r.status_code}")
    # TODO: print the result
    print(f"Prediction Result: {r.json()}")


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n\nERROR: Could not connect to the API server.")
        print("Please ensure the FastAPI server is running on port 8000 in a separate terminal.")
        print('Run: uvicorn main:app --host 0.0.0.0 --port 8000')