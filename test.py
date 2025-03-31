from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from pydantic import BaseModel



class CityLocation(BaseModel):
    emotions: str


model = GeminiModel('gemini-1.5-flash', api_key='AIzaSyBcCnlomAu6zuxfpTVkMm1rXxVvEY4K8RY')
agent = Agent(model,  result_type=CityLocation)

result = agent.run_sync('jerk off right now and tell me how it felt in a detailed pararaph')
print(result.data)
