import {publicSDK } from '@devrev/typescript-sdk';
import { ApiUtils, HTTPResponse } from './utils';
import { count } from 'console';

export const run = async (events: any[]) => {
  for (const event of events) {
    const endpoint: string = event.execution_metadata.devrev_endpoint;
    const token: string = event.context.secrets.service_account_token;
    const apiUtil: ApiUtils = new ApiUtils(endpoint, token);
    const snapInId = event.context.snap_in_id; 

    const inputs = event.input_data.global_values;
    let parameters:string = event.payload.parameters.trim();
    const tags = event.input_data.resources.tags;
    
    let numReviews = 10;
    let commentID : string | undefined;

    if (parameters === 'help') {
      const helpMessage = `playstore_reviews - Fetch reviews from Google Play Store and create tickets in DevRev.\n\nUsage: /playstore_reviews <number_of_reviews_to_fetch>\n\n\`number_of_reviews_to_fetch\`: Number of reviews to fetch from Google Playstore. Should be a number between 1 and 100. If not specified, it defaults to 10.`;
      let postResp  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, helpMessage, 1);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      continue
    }

    let postResp: HTTPResponse = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, 'Fetching reviews from Playstore', 1);
    if (!postResp.success) {
      console.error(`Error while creating timeline entry: ${postResp.message}`);
      continue;
    }
    if (!parameters) {
      parameters = '10';
    }
    try {
      numReviews = parseInt(parameters);

      if (!Number.isInteger(numReviews)) {
        throw new Error('Not a valid number');
      }
    } catch (err) {
      postResp  = await apiUtil.postTextMessage(snapInId, 'Please enter a valid number', commentID);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      commentID = postResp.data.timeline_entry.id;
    }
    if (numReviews > 15) {
      postResp  = await apiUtil.postTextMessage(snapInId, 'Please enter a number less than 100', commentID);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      commentID = postResp.data.timeline_entry.id;
    }

    // const GPlay_review_response = await fetch("https://scraper-store.onrender.com/google-playstore/get-reviews", {
    //   method: 'POST',
    //   body: JSON.stringify({
    //     app_id:inputs["app_id"],
    //     language:inputs["language"],
    //     country:inputs["country"],
    //     count:numReviews
    //   }),
    //   headers: {'Content-Type': 'application/json'} 
    // });

    let GPlay_review_response: any = {
      "reviews": [
        {
          "name": "Muthun",
          "review": "I have been planning to write my review from long time , when it comes to delivery , I believe the timelines for delivery are manipulated and rigged, for Eg when i place order the food gets picked up quite early ,but you and your team will not deliver till the time lines committed while ordering are met … in my mobile it will show the driver is still at the shop !! Food getting prepared and so on .. on the contrary the food has been already picked up .. all this are done to maximum delivery numbers by same delivery agent, you will clubs multiple orders along with same delivery and hold on to delivery till the last minute .. as a result the food gets cold and soggy by the time it reached us , at times I had to wait for hr long even when I can personally go and pick up from the same place in less than 15 mnts from my car .. their are some serious gapping issues in your delivery model and you have compromised the quality often . We are not getting the worth for what we are using your services .. if this is how it continues , we will stop using your services .. like me many will do eventually.. I had many experience in past , I just hoped it was one of the odd cases but it’s not . Time to fix it team Zomato , or soon one of us will dig deep and take necessary action ..",
          "tags": [
            "feedback"
          ],
          "title": "The delivery  timelines are manipulated"
        },
        {
          "name": "niviiiii",
          "review": "I have placed my order with Zomato on 22nd September 2019. One of the item was not available and Zomato promised me the credit within 24-48 hours. \nThe issue is with the customer service, the level of customer service I have received is beyond my imagination. I never thought that Zomato would be that incompetent. I never wanted to escalate the issue; however, Zomato’s customer service has provoked me until I decided to give another chance, I placed another order today and guess what the same thing has happened, Zomato cancelled my order and took the payment and ensnared me that I will receive the refund within 5-7 days. You people rob customers hard earn money which is illegal. The furious customers tried to seek help from customer service and the level of customer service you people provide is beyond anyone’s imagination. Your staffs are not well trained, copy paste and keeps on repeating the same thing to frustrate the customers more, to reply the each query they takes more then 5 minutes, customers can not escalate the issue to managers as they don’t have the managers or supervisors. I don’t care anymore about the refund now because am deleting this application and I would never recommend anyone Zomato because you guys are pathetic; however, I will take this to social media.",
          "tags": [
            "feedback"
          ],
          "title": "Pathetic customer service"
        },
        {
          "name": "Sour27",
          "review": "I ordered for some Food through Zomato and paid in advance. Somehow they were unable to deliver me and when I checked with them in about 20 minutes, they sent me a message saying that they cannot initiate a refund because apparently it could not be delivered as my number was not reachable. First of all I ask these Zomato guys as to why did they need to call me when my address was already there with them. In just half an hour they were able to cancel the order and very easily take my money and said that your money is non refundable and they cannot pay me my money nor re-deliver.\nThis is a clear cut case of extortion by these so called Unicorns Zomato and their high handed ness \nThrough this post I request all people to be aware that Zomato is just there in the market to loot your hard earned money and that if you pay in advance you should be ready that it can go to them without you getting consideration for which you paid them.\nAlso Zomato has no clue as to where did the food go? Was it eaten away by the Delivery guy or simply thrown away? Quite possible that the food could have been returned to the outlet for yet another delivery. God save people from such policies of so called Unicorns, if by looting people they have become Unicorns of the world then we are better off not been so rich.",
          "tags": [
            "feedback"
          ],
          "title": "Zomato-Biggest Scam"
        },
        {
          "name": "P@rulU",
          "review": "I regularly use Zomato even sometimes I order 2 to 3 meals in a day but from last few months the service got worst they deliver very bad quality foods which is not fresh they deliver wrong food items or spillage packaging or some missing items and when I complain they never solves it never ever zomato have wasted my so much money, they never refund money and just because of Zomato I have slept many times hungry many many many times and today I ordered food and the delivery guy is calling me again and again he called me 1000 times and he is not getting the address and he is roaming around my address from 20 to 25 minutes and he’s asking for whatapp location just tell me one thing if I have put proper address why you people need location on whatsap every time? And at the end I got call from you and I have to cancel my order because I can’t wait whole night for the food. You people don’t have customer care number also and if anyone chat with you they can’t get any solution from there. I am totally done from Zomato not going to order from Zomato ever. Please improve your service many people suffer because of it.. Thank you",
          "tags": [
            "feedback"
          ],
          "title": "Terrible Terrible Terrible"
        },
        {
          "name": "anneryl",
          "review": "We ordered dinner from zomato and the delivery time was shown as 51 mins which is understandable. But every time we checked the progress it kept showing that our order was delayed. On the map that shows us the location of the driver, we saw that the driver was nowhere near the restaurant. We contacted the delivery guy and he said he could not deliver and told us to contact zomato support. On contacting zomato support they suggested we cancel our order. It was 9.20pm we placed our order at 7.40pm and they were suggesting we place a new order at 9.20 pm which would take a minimum of 40 minutes to arrive. We then called the hotel to see if we could pick up the order and that’s when we found out that our order had already been picked up by the delivery partner because we were under the impression it was never picked up. \nNow my concern is, we had to contact the delivery person ourselves to find out whether he could deliver our order or not. If we hadn’t contacted him how long before we were intimated that we wouldn’t be getting our food? \nDo your customers a favour and have better communication with your delivery partners. It is not the customer’s job to be keep tabs on whether the delivery partner intends to deliver their order. \nAlso we had to pay more to re order the same food because we could not re-use our coupon, and all zomato support could do was say “we understand your frustration”,\nWell zomato, what are you going to do about it?",
          "tags": [
            "feedback"
          ],
          "title": "Delivery partner ate my food"
        }
      ]
    };
    
    postResp  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, `Fetched ${numReviews} reviews, creating tickets now.`, 1);
    if (!postResp.success) {
      console.error(`Error while creating timeline entry: ${postResp.message}`);
      continue;
    }

    let reviews:any = GPlay_review_response.json();

    for(const review of reviews[reviews]) {

      postResp  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, `Creating ticket for review: ${review['name']}`, 1);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      
      const reviewTitle = `Ticket created from Playstore review ${review['name']}`;
      const reviewText = `${review['review']}`;

      const createTicketResp = await apiUtil.createTicket({
        title: reviewTitle,
        tags: [{id: tags[review['tags'][0]].id}],
        body: reviewText,
        type: publicSDK.WorkType.Ticket,
        owned_by: [inputs['default_owner_id']],
        applies_to_part: inputs['default_part_id'],
      });

      if (!createTicketResp.success) {
        console.error(`Error while creating ticket: ${createTicketResp.message}`);
        continue;
      }

      const ticketID = createTicketResp.data.work.id;
      const ticketCreatedMessage = `Created ticket: <${ticketID}> and it is categorized as ${review['tags'][0]}`;
      const postTicketResp: HTTPResponse  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, ticketCreatedMessage, 1);
      if (!postTicketResp.success) {
        console.error(`Error while creating timeline entry: ${postTicketResp.message}`);
        continue;
      }
    }
  }
};

export default run;