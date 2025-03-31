from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

async def get_vader_sentiment(feedback: str) -> dict:
    """
    Analyze sentiment of feedback using VADER sentiment analyzer.
    Returns a dictionary containing:
    - compound: Normalized compound score (-1 to 1)
    - sentiment_label: POSITIVE, NEGATIVE, or NEUTRAL based on compound score
    - scores: Dictionary of pos, neg, neu scores
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(feedback)
        
        # Determine sentiment label based on compound score
        if scores['compound'] >= 0.05:
            sentiment_label = "POSITIVE"
        elif scores['compound'] <= -0.05:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
            
        return {
            "compound": scores['compound'],
            "sentiment_label": sentiment_label,
            "scores": {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu']
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing sentiment analysis: {str(e)}"
        )
