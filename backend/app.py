from flask import Flask, request, jsonify
from generator import generate_synthetic_data
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    topic = data.get("topic")
    fields = int(data.get("fields"))
    records = int(data.get("records"))
    file_format = data.get("format", "excel")

    try:
        filename = generate_synthetic_data(topic, fields, records, file_format)
        return jsonify({"message": f"Data generated and saved to {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
