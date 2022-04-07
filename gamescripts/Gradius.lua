
socket = require("socket.core")

tcp = socket.tcp()

repeat
	ret = tcp:connect("localhost", 12345)
	emu.frameadvance()
	print(ret)
	client.sleep(100)
until ret ~= nil

buttons = {"A", "B", "Down", "Left", "Right", "Select",	"Start", "Up"}
client.speedmode(5000)

function get_input(x, genomes)
	local map = {}
	for button, genome in pairs(genomes) do
		value = 0
		for i = 1,#genome do
			value = value + genome[i][2] * math.sin(genome[i][1] * x + genome[i][3])
		end
		if value > 0 then
			map[button] = 1
		end
	end
	return map
end

function evaluate(genomes)	
	local fitness = 0
	local no_change = 0
	savestate.load("./start.State")
	while true do
		local new_fitness = memory.read_u32_le(0x07E3)
		if fitness == new_fitness then
			no_change = no_change + 1
		else
			fitness = new_fitness
			no_change = 0
		end
		if no_change == 800 or memory.read_u8(0x0020) == 2 then
			break
		end
		local frame = emu.framecount()
		local inputs = get_input(frame, genomes)
		joypad.set(inputs, 1)
		emu.frameadvance();
	end
	print("Achieving a fitness of ", fitness)
	return fitness
end

while true do
	line, status, partial = tcp:receive()
	if status == "closed" then
		tcp:close()
		tcp = socket.tcp()
		repeat
			ret = tcp:connect("localhost", 12345)
			emu.frameadvance()
			print(ret)
			client.sleep(100)
		until ret ~= nil
	end
	in_button = false
	fitness = 0
	button = ""
	id = -1
	current = {}
	genome = {}
	while line ~= nil do
		if nil ~= string.match(line, "End individual") then
			fitness = evaluate(current)
			tcp:send(fitness .. " " .. id)
			current = {}
			break
		elseif in_button and nil ~= string.match(line, "End button") then
			in_button = false
			current[button] = genome
			genome = {}
		elseif id == -1 then
			id = string.match(line, ".+")
		elseif not in_button then
			in_button = true
			button = string.match(line, ".+")
		else
			freq, mag, phas = string.match(line, "(%S+) (%S+) (%S+)")
			table.insert(genome, {tonumber(freq), tonumber(mag), tonumber(phas)})
		end
		line = tcp:receive()
	end
	emu.frameadvance()
	client.sleep(50)
end
tcp:close()
