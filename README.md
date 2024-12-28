# MLB Contracts Repository

## Dataset

### Contracts Dataset

The contracts dataset was assembled using contract information from [Cot's Baseball Contracts](https://legacy.baseballprospectus.com/compensation/cots/). The dataset currently includes player contract information from 2020 through 2024.

To regenerate the dataset, place all spreadsheets from Cot's Contracts in a directory called `contract_data`, and then run the script `python parse.py`. Make sure that the name of each file is structured `MLB {team} {year}.csv`. The dataset will be generated in the `dataset` directory.

Columns:
- first name
- last name
- team
- year
- position
- age
- service time
- agent
- value: salary in millions of dollars

## Models

### Model Building

1) The first step in model building is to divide players based on their positions on the field. Players who play the same position generally get paid on a similar scale, and players who play different positions don't get paid the same amount (center fielders and relief pitchers get paid different amounts on average).

#### Best Models So Far:
Starting Pitchers - SVR with GS, service time, ERA (starter_support_vector_regression.pkl)
Relief Pitchers - SVR with G, service time, SV, ERA, SO (reliever_support_vector_regression.pkl)
Position Players - Linear Regression with G, service time, AB, HR, SB (position_linear_regression.pkl)