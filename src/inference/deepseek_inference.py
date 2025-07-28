from openai import OpenAI
from loguru import logger

def test_deepseek_api(api_key: str):
    """Test basic functionality of DeepSeek API."""
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # alpaca_5k; 
    # Test different prompts
    test_cases = [
        {
            "description": "Basic conversation",
            "messages": [
                {"role": "user", "content": "What is Python?"}
            ]
        },
        {
            "description": "Paper abbreviation extraction",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that identifies paper names and abbreviations."},
                {"role": "user", "content": "BERT and GPT are popular language models."}
            ]
        },
        {
            "description": "Code generation",
            "messages": [
                {"role": "system", "content": "You are a Python programming expert."},
                {"role": "user", "content": "Write a simple function to calculate Fibonacci numbers."}
            ]
        }
    ]
    
    # Run tests
    for test in test_cases:
        logger.info(f"\nTesting: {test['description']}")
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=test["messages"],
                temperature=0.7
            )
            
            logger.info("Response received:")
            logger.info(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Replace with your API key
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    logger.info("Starting DeepSeek API tests")
    test_deepseek_api(API_KEY) 