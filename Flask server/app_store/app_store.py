from flask import Blueprint
from flask import request
from flask import jsonify
from app_store_scraper import AppStore
from tags import AssignTags

app_store_blueprint = Blueprint('app_store_blueprint', __name__)

assign_tag = AssignTags()

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
    
    i = 0
    for r_id in reviews:
        if i < data.get('count'):
            r = {'name':None, 'title':None, 'review':None, 'tags':None}
            r['name'] = r_id['userName']
            r['title'] = r_id['title']
            r['review'] = r_id['review']
            tags = assign_tag.AppStoreReview(question=r_id['review'])
            r['tags'] = tags
            review.append(r)
            i+=1
        else:
            break
        


    return jsonify(reviews=review), 200

