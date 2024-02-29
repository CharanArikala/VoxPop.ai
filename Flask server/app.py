from flask import Flask
from flask_cors import CORS
from google_playstore.google_playstore import google_playstore_blueprint
from app_store.app_store import app_store_blueprint

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(google_playstore_blueprint, url_prefix='/google-playstore')
app.register_blueprint(app_store_blueprint, url_prefix='/app-store')

@app.route('/')
def hello_world():
    return 'Kick Started Server'