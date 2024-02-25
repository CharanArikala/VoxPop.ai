from flask import Blueprint
from flask import request
from flask import jsonify
from app_store_scraper import AppStore

app_store_blueprint = Blueprint('app_store_blueprint', __name__)

@app_store_blueprint.route('/test', methods=['GET'])
def test():

    return jsonify(test="Pass"), 200

@app_store_blueprint.route('/get-reviews', methods=['POST'])
def GetReviewsAppStore():

    data = request.json

    app_store = AppStore(country=data.get('country'), app_name=data.get('app_name'), app_id=data.get('app_id'))
    review = []

    app_store.review(how_many=data.get('count'))

    reviews = app_store.reviews
    
    for r_id in reviews:
        r = {'name':None, 'title':None, 'review':None}
        r['name'] = r_id['userName']
        r['title'] = r_id['title']
        r['review'] = r_id['review']
        review.append(r)

    return jsonify(reviews=review), 200

