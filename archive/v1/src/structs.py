class Player:
    def __init__(self, name):
        self.name = name

class Pitcher(Player):
    def __init__(self, name, innings_pitched, runs_allowed, hits_allowed, walks_allowed):
        super().__init__(name)
        self.innings_pitched = innings_pitched
        self.runs_allowed = runs_allowed
        self.hits_allowed = hits_allowed
        self.walks_allowed = walks_allowed

class StartingPitcher(Pitcher):
    def __init__(self, name, innings_pitched, runs_allowed, hits_allowed, walks_allowed, wins, losses):
        super().__init__(name, innings_pitched, runs_allowed, hits_allowed, walks_allowed)
        self.wins = wins
        self.losses = losses

    def data_to_list(self):
        return [
            self.innings_pitched,
            self.runs_allowed,
            self.hits_allowed,
            self.walks_allowed,
            self.wins,
            self.losses
        ]

class ReliefPitcher(Pitcher):
    def __init__(self, name, innings_pitched, runs_allowed, hits_allowed, walks_allowed, holds, saves, blown_saves):
        super().__init__(name, innings_pitched, runs_allowed, hits_allowed, walks_allowed)
        self.holds = holds
        self.saves = saves
        self.blown_saves = blown_saves

    def data_to_list(self):
        return [
            self.innings_pitched,
            self.runs_allowed,
            self.hits_allowed,
            self.walks_allowed,
            self.holds,
            self.saves,
            self.blown_saves
        ]
    
class PositionPlayer(Player):
    def __init__(self):
        pass