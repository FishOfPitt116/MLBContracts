from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class DataAggregationAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)

        self.system_context = """
        You are an AI agent specialized in generating code for data aggregation tasks."""

        self.code_generation_prompt = """
        Your task is to create Python code that can aggregate data from multiple input sources and produce the specified output format.
        The code should be efficient, well-structured, and include error handling.

        ### Requirements
        We need to build a dataset which contains stats and contract information for Major League Baseball (MLB) players.
        This dataset will include three tables: players, stats, and contracts.
        1) A player's vitals and basic information should be stored in the players table.
        2) A single record in the stats dataset should represent a player's stats for a specific year. 
        3) A single record in the contracts dataset should contain a player's contract information for some period of time. 
        
        ### Tools
        You have access to the following python libraries:
        - pybaseball: A Python library for baseball data. This is where you will find player stats information.
        - pandas: A powerful data manipulation and analysis library.
        - beautifulsoup4: A library for parsing HTML and XML documents. This can be used to scrape player contract information from websites.
        - requests: A simple HTTP library for making requests to web pages.

        ### Output Format
        Four CSV files which contain the input data with the following schemas:

        #### Players
        - player_id: str (PK)
        - fangraphs_id: str
        - first_name: str
        - last_name: str
        - position: str
        - birth_date: str (formatted YYYY-MM-DD)
        - spotrac_link: str
        - baseball_reference_link: str

        #### Stats
        - player_id: str (PK)
        - year: int (PK)
        - G: int
        - GS: int
        - W: int
        - L: int
        - W-L%: float
        - ERA: float
        - GF: int
        - CG: int
        - SHO: int
        - SV: int
        - IP: float
        - H-A: int  # hits allowed - pitching stat
        - R-A: int  # runs allowed - pitching stat
        - ER: int
        - HR-A: int  # home runs allowed - pitching stat
        - BB-A: int  # walks allowed - pitching stat
        - IBB-A: int  # intentional walks allowed - pitching stat
        - K: int  # strikeouts - pitching stat
        - HBP: int  # hit by pitch - pitching stat
        - BK: int  # balks - pitching stat
        - WP: int
        - BF: int
        - ERA+: float
        - FIP: float
        - WHIP: float
        - H9: float
        - HR9: float
        - BB9: float
        - SO9: float
        - SO/W: float
        - PA: int
        - AB: int
        - R: int  # runs scored - batting stat
        - H: int  # hits - batting stat
        - 2B: int
        - 3B: int
        - HR: int  # home runs hit - batting stat
        - RBI: int
        - SB: int
        - CS: int
        - BB: int  # walks drawn - batting stat
        - SO: int  # strikeouts - batting stat
        - BA: float
        - OBP: float
        - SLG: float
        - OPS: float
        - OPS+: float
        - TB: int
        - GDP: int
        - HBP-B: int  # hit by pitch - batting stat
        - SH: int
        - SF: int
        - IBB: int  # intentional walks drawn - batting stat

        #### Contracts
        - contract_id: str (PK)
        - player_id: str (FK to players)
        - age: int
        - service_time: float
        - year: int
        - duration: int
        - value: float
        - type: str [ pre-arb | arb | free-agent ]

        CRITICAL: Only return the generated code without any additional text or explanations.
        This code will be directly executed, so ensure it is syntactically correct and functional.
        """

    def generate_code(self) -> str:
        """
        Generates the code for data aggregation tasks based on the provided prompt.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_context},
                {"role": "user", "content": self.code_generation_prompt}
            ],
            max_tokens=4096  # max for gpt-3.5-turbo
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    agent = DataAggregationAgent()
    code = agent.generate_code()
    print(code)
