from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random

app = FastAPI()

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/startgame/numberOfPlayers={player_count}startingDice={dice_count_pp}")
async def start_game(player_count: int, dice_count_pp: int):
    all_hands = []
    for player in range(player_count):
        player_hand = [random.randint(1, 6) for _ in range(dice_count_pp)]
        all_hands.append(player_hand)
    return {"all_hands": all_hands}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)