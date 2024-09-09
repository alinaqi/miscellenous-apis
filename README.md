
```markdown
# Miscellenous APIs

These are different apis, I've been experimenting with. README is outdated.. just look at the code :)

## Features

- Upload an image of food, and the API will identify the food and estimate the calories.
- Supports Cross-Origin Resource Sharing (CORS) to allow integration with frontend applications.
- Uses the OpenAI API to analyze images and return results in JSON format.

## Requirements

- Python 3.8+
- FastAPI
- Pillow
- Requests
- OpenAI API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/alinaqi/backend-recipie.git
   cd backend-recipie
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:

   Create a `.env` file in the root directory of the project and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Run the FastAPI application:

   ```bash
   uvicorn main:app --reload
   ```

2. Use a tool like `curl` or Postman, or integrate it into your frontend to upload images and receive calorie estimates.

   Example using `curl`:

   ```bash
   curl -X POST "http://127.0.0.1:8000/upload/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_image"
   ```

## API Endpoints

### POST /upload/

- **Description**: Upload an image to be processed by the API.
- **Parameters**:
  - `file`: The image file to be uploaded.
- **Response**:
  - JSON object containing the identified food and estimated calories.
  
  Example Response:

  ```json
  {
    "food": "pizza",
    "calories": "285"
  }
  ```

## Notes

- The application uses the OpenAI API to process images, which may have usage limits based on your API key's plan.
- Ensure that your `.env` file is not tracked by Git to keep your API key secure. Add `.env` to your `.gitignore` file if not already included.

## License

This project is licensed under the Do-whatever-you-want License. :)
```
