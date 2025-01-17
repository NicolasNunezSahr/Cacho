from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import asyncio

app = FastAPI()

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/startgame/numberOfPlayers={player_count}startingDice={dice_count_pp}")
# async def start_game(player_count: int, dice_count_pp: int):
#     all_hands = []
#     for _ in enumerate(range(player_count)):
#         player_hand = [random.randint(1, 6) for _ in range(dice_count_pp)]
#         all_hands.append(player_hand)
#
#     player_metadata = {}
#     player_metadata['player_count'] = player_count
#     random.shuffle(all_hands)
#     for i, hand in enumerate(all_hands):
#         player_metadata[f'player_{i + 1}'] = {'player_id': i + 1, 'hand': hand}
#
#     return {"player_metadata": player_metadata}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# Store active connections
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)


async def simulate_player_actions():
    """Simulate your player action loop"""
    players = ["Player1", "Player2", "Player3"]  # Example players
    while True:
        for player in players:
            # Your actual player action logic here
            action = random.randint(1, 6)
            # Broadcast to all connected clients
            for connection in active_connections:
                try:
                    await connection.send_json({
                        "player": player,
                        "action": action
                    })
                except Exception as e:
                    print(f'Exception sending json: {e}')

            await asyncio.sleep(1)  # Add delay between actions

@app.on_event("startup")
async def startup_event():
    # Start the player action simulation in the background
    asyncio.create_task(simulate_player_actions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
