import {publicSDK } from '@devrev/typescript-sdk';
import { ApiUtils, HTTPResponse } from './utils';
import { count } from 'console';
import {LLMUtils} from './llm_utils';

export const run = async (events: any[]) => {
  for (const event of events) {
    const endpoint: string = event.execution_metadata.devrev_endpoint;
    const token: string = event.context.secrets.service_account_token;
    const apiUtil: ApiUtils = new ApiUtils(endpoint, token);
    const snapInId = event.context.snap_in_id; 
    const fireWorksApiKey: string = event.input_data.keyrings.fireworks_api_key;

    // const devrevPAT = event.context.secrets.service_account_token;
    // const baseURL = event.execution_metadata.devrev_endpoint;
    const inputs = event.input_data.global_values;
    let parameters:string = event.payload.parameters.trim();
    const tags = event.input_data.resources.tags;
    
    const llmUtil: LLMUtils = new LLMUtils(fireWorksApiKey, `accounts/fireworks/models/${inputs['llm_model_to_use']}`, 200);

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
      // Default to 10 reviews.
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
    if (numReviews > 100) {
      postResp  = await apiUtil.postTextMessage(snapInId, 'Please enter a number less than 100', commentID);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      commentID = postResp.data.timeline_entry.id;
    }

    const GPlay_review_response = await fetch("https://scraper-store.onrender.com/app-store/get-reviews", {
      method: 'POST',
      body: JSON.stringify({
        app_id:inputs["app_id"],
        language:inputs["language"],
        country:inputs["country"],
        count:numReviews
      }),
      headers: {'Content-Type': 'application/json'} 
    });

    
    postResp  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, `Fetched ${numReviews} reviews, creating tickets now.`, 1);
    if (!postResp.success) {
      console.error(`Error while creating timeline entry: ${postResp.message}`);
      continue;
    }

    // commentID = postResp.data.timeline_entry.id;

    let reviews:any = GPlay_review_response;

    for(const review of reviews) {
      postResp  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, `Creating ticket for review: ${review['name']}`, 1);
      if (!postResp.success) {
        console.error(`Error while creating timeline entry: ${postResp.message}`);
        continue;
      }
      const reviewText = `Ticket created from Playstore review ${review['name']}\n\n${review['content']}`;
      const reviewTitle = review['name'] || `Ticket created from Playstore review ${review['name']}`;
      const systemPrompt = `You are an expert at labelling a given Google Play Store Review as bug, feature_request, question or feedback. You are given a review provided by a user for the app ${inputs['app_id']}. You have to label the review as bug, feature_request, question or feedback. The output should be a JSON with fields "category" and "reason". The "category" field should be one of "bug", "feature_request", "question" or "feedback". The "reason" field should be a string explaining the reason for the category. \n\nReview: {review}\n\nOutput:`;
      const humanPrompt = ``;

      let llmResponse = {};
      try {
        llmResponse = await llmUtil.chatCompletion(systemPrompt, humanPrompt, {review: (reviewTitle ? reviewTitle + '\n' + reviewText: reviewText)})
      } catch (err) {
        console.error(`Error while calling LLM: ${err}`);
      }
      let tagsToApply = [];
      let inferredCategory = 'failed_to_infer_category';
      if ('category' in llmResponse) {
        inferredCategory = llmResponse['category'] as string;
        if (!(inferredCategory in tags)) {
          inferredCategory = 'failed_to_infer_category';
        }
      }
      const createTicketResp = await apiUtil.createTicket({
        title: reviewTitle,
        tags: [{id: tags[inferredCategory].id}],
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
      const ticketCreatedMessage = inferredCategory != 'failed_to_infer_category' ? `Created ticket: <${ticketID}> and it is categorized as ${inferredCategory}` : `Created ticket: <${ticketID}> and it failed to be categorized`;
      const postTicketResp: HTTPResponse  = await apiUtil.postTextMessageWithVisibilityTimeout(snapInId, ticketCreatedMessage, 1);
      if (!postTicketResp.success) {
        console.error(`Error while creating timeline entry: ${postTicketResp.message}`);
        continue;
      }
    }
  }
};

export default run;