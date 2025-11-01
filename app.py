from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the front page."""
    return render_template('index.html')

@app.route('/api/echo', methods=['POST'])
def echo():
    """Simple demo API route."""
    data = request.get_json()
    text = data.get('text', '')
    return jsonify({"reply": f"You said: {text}"})

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """Example geocoding endpoint (uses Nominatim)."""
    query = request.get_json().get("query")
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "durhack2025"})
    if not r.ok or not r.json():
        return jsonify({"error": "Location not found"}), 404
    res = r.json()[0]
    return jsonify({
        "query": query,
        "lat": res["lat"],
        "lon": res["lon"],
        "display_name": res["display_name"]
    })

if __name__ == '__main__':
    app.run(debug=True)
