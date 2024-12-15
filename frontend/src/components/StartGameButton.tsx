'use client';
import { useState } from 'react';
import { Dice1, Dice2, Dice3, Dice4, Dice5, Dice6 } from 'lucide-react';

const StartGameButton = () => {
  const [isGameStarted, setIsGameStarted] = useState(false);
  const [numberOfPlayers, setNumberOfPlayers] = useState(2);
  const [startingDice, setStartingDice] = useState(1);
  const [gameMessage, setGameMessage] = useState('');
  const [diceValues, setDiceValues] = useState([]);

  const getDice = async (numberOfPlayers, startingDice) => {
    try {
      const response = await fetch(
        `http://localhost:8000/startgame/numberOfPlayers=${numberOfPlayers}startingDice=${startingDice}`
      );
      const data = await response.json();
      return data.all_hands;
    } catch (error) {
      console.error('Error:', error);
      return [];
    }
  };

  const handleStartGame = async () => {
    setIsGameStarted(true);
    const hands = await getDice(numberOfPlayers, startingDice);
    console.log('Hands before setting state:', hands);
    setDiceValues(hands);
    setGameMessage(`${numberOfPlayers} players are playing with ${startingDice} dice each`);
  };

  const DiceFace = ({ value }) => {
    const diceProps = { size: 48, className: "text-blue-600" };
    switch (value) {
      case 1: return <Dice1 {...diceProps} />;
      case 2: return <Dice2 {...diceProps} />;
      case 3: return <Dice3 {...diceProps} />;
      case 4: return <Dice4 {...diceProps} />;
      case 5: return <Dice5 {...diceProps} />;
      case 6: return <Dice6 {...diceProps} />;
      default: return null;
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto flex flex-col items-center gap-8 p-4">
      {!isGameStarted ? (
        <div className="flex flex-col items-center gap-6">
          <div className="flex gap-8">
            <div className="flex flex-col gap-2">
              <label htmlFor="players" className="text-sm font-medium">
                Number of Players
              </label>
              <select
                id="players"
                value={numberOfPlayers}
                onChange={(e) => setNumberOfPlayers(Number(e.target.value))}
                className="w-48 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {[2, 3, 4, 5, 6].map(num => (
                  <option key={num} value={num}>{num} Players</option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-2">
              <label htmlFor="dice" className="text-sm font-medium">
                Starting Dice per Player
              </label>
              <select
                id="dice"
                value={startingDice}
                onChange={(e) => setStartingDice(Number(e.target.value))}
                className="w-48 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {[1, 2, 3, 4, 5].map(num => (
                  <option key={num} value={num}>{num} Dice</option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={handleStartGame}
            className="px-6 py-3 text-lg font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Start Cacho Game
          </button>
        </div>
      ) : (
        <div className="w-full">
          <div className="text-center mb-6 text-lg font-medium">
            {gameMessage}
          </div>
          <div className="space-y-6">
            {diceValues.map((playerHand, playerIndex) => (
              <div key={playerIndex}>
                <div className="font-medium mb-2">Player {playerIndex + 1}</div>
                <div style={{ display: 'flex', flexDirection: 'row', gap: '1rem' }}> {/* Inline styles for debugging */}
                  {playerHand.map((value, diceIndex) => (
                    <div
                      key={`${playerIndex}-${diceIndex}`}
                      className="w-16 h-16 flex items-center justify-center bg-white rounded-lg shadow-md"
                    >
                      <DiceFace value={value} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default StartGameButton;