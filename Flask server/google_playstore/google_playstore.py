from flask import Blueprint
from flask import request
from flask import jsonify
from google_play_scraper import Sort, reviews

google_playstore_blueprint = Blueprint('google_playstore_blueprint', __name__)

@google_playstore_blueprint.route('/test', methods=['GET'])
def Test():

    return jsonify(test="Pass"), 200


@google_playstore_blueprint.route('/get-reviews', methods=['POST'])
def GetReviewsGPlay():

    data = request.json

    review = []
    result, _ = reviews(
        data.get('app_id'),
        lang=data.get('language'), 
        country=data.get('country'), 
        sort=Sort.NEWEST, 
        count=data.get('count'),
        filter_score_with=3
    )

    for r_id in result:
        r = {'name':None, 'review':None}
        r['name'] = r_id['userName']
        r['review'] = r_id['content']
        review.append(r)

    return jsonify(reviews=review), 200



