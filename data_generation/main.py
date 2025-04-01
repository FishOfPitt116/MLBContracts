"""
This file should be used for putting together the Player, Contracts, and Stats datasets and exporting them to CSV files.
The main function should be used to call the functions in the other files to generate the datasets and save them to CSV files.
This file should only have to do with assembling 

1) Enable pybaseball cache
2) Generate player and contract records via spotrac.
    a) While these records are being generated, use pybaseball to get player ID, start year and end year (from pybaseball import playerid_lookup)
    NOTE: There will be an issue when it comes to two players with the same name. This will need to be addressed.
3) For each generated player record, use pybaseball to get player stats.
4) Save all records to CSV files.
"""