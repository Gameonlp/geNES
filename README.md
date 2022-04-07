# geNES
A genetic algorithm that plays NES games

## Introduction
geNES is a genetic algorithm, which learns sine waves. It is built as two major components, the first being a TCP server and the second being a thin client for the emulator and game.

## Execution
### Requirements
- python
- BizHawk Emulator

### Setup
1. Clone this repository
2. Open the BizHawk Emulator
3. Make a named save state (File -> Save State -> Save Named State...), name it start.State and save it in the repository folder

### How to run
1. Start Bizhawk, load your legally obtained ROM and the repective script
2. Repeat step one as many times as you want and your PC supports (usually number physical cores times)
3. Start GenServer.py script by running ``` python GenServer.py ```

Instances of the emulator can also be added on the fly and the server can be restarted, the instances will reconnect to it.
