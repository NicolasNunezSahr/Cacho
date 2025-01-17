import asyncio
import json
import random
from websockets.server import serve


class GameState:
    def __init__(self):
        self.players = {}  # Store WebSocket connections with player IDs
        self.game_started = False
        self.all_hands = []


async def register(websocket, game_state):
    # Assign a player ID (just use the length of current players + 1)
    player_id = len(game_state.players) + 1
    game_state.players[player_id] = websocket
    return player_id


async def unregister(websocket, game_state):
    # Remove player when they disconnect
    player_id = None
    for pid, ws in game_state.players.items():
        if ws == websocket:
            player_id = pid
            break
    if player_id:
        del game_state.players[player_id]


async def start_game(player_count: int, dice_count_pp: int) -> list:
    all_hands = []
    for player in range(player_count):
        player_hand = [random.randint(1, 6) for _ in range(dice_count_pp)]
        all_hands.append(player_hand)
    return all_hands


async def handler(websocket, game_state):
    try:
        # Register the new player
        player_id = await register(websocket, game_state)

        # Send the player their ID
        await websocket.send(json.dumps({
            "type": "registration",
            "player_id": player_id
        }))

        # Handle incoming messages
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "start_game":
                player_count = data["numberOfPlayers"]
                dice_count_pp = data["startingDice"]

                # Generate hands for all players
                all_hands = await start_game(player_count, dice_count_pp)
                game_state.all_hands = all_hands

                # Send hands to all connected players
                for pid, ws in game_state.players.items():
                    # Send each player only their hand
                    player_hand = all_hands[pid - 1]  # pid is 1-based
                    await ws.send(json.dumps({
                        "type": "game_started",
                        "your_hand": player_hand,
                        "player_count": player_count
                    }))

    except Exception as e:
        print(f"Error handling connection: {e}")
    finally:
        await unregister(websocket, game_state)


async def main():
    game_state = GameState()
    async with serve(
            lambda ws: handler(ws, game_state),
            "localhost",
            8000
    ):
        print("WebSocket server started on ws://localhost:8000")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())