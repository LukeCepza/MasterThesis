-- this function is called when the box is initialized
function initialize(box)
end

-- this function is called when the box is uninitialized
function uninitialize(box)
	io.write("uninitialize has been calle d\n")
end

function process(box)
	box:send_stimulation(1,33084,2,0) --Start LSL communication

box:send_stimulation(1,1999,10,0)
box:send_stimulation(1,33026,12,0)
box:send_stimulation(1,0200,13,0)
box:send_stimulation(1,33027,16,0)
box:send_stimulation(1,0200,17,0)
box:send_stimulation(1,33025,20,0)
box:send_stimulation(1,0200,21,0)
box:send_stimulation(1,33024,24,0)
box:send_stimulation(1,0200,25,0)
box:send_stimulation(1,1999,34,0)
box:send_stimulation(1,33024,36,0)
box:send_stimulation(1,0200,37,0)
box:send_stimulation(1,33027,40,0)
box:send_stimulation(1,0200,41,0)
box:send_stimulation(1,33025,44,0)
box:send_stimulation(1,0200,45,0)
box:send_stimulation(1,33026,48,0)
box:send_stimulation(1,0200,49,0)
box:send_stimulation(1,1999,58,0)
box:send_stimulation(1,33024,60,0)
box:send_stimulation(1,0200,61,0)
box:send_stimulation(1,33026,64,0)
box:send_stimulation(1,0200,65,0)
box:send_stimulation(1,33027,68,0)
box:send_stimulation(1,0200,69,0)
box:send_stimulation(1,33025,72,0)
box:send_stimulation(1,0200,73,0)
box:send_stimulation(1,1999,82,0)
box:send_stimulation(1,33027,84,0)
box:send_stimulation(1,0200,85,0)
box:send_stimulation(1,33024,88,0)
box:send_stimulation(1,0200,89,0)
box:send_stimulation(1,33025,92,0)
box:send_stimulation(1,0200,93,0)
box:send_stimulation(1,33026,96,0)
box:send_stimulation(1,0200,97,0)

box:send_stimulation(1,32770,100,0)   -- End stimuli
box:send_stimulation(1,32770,102,0) 
	box:sleep()
	io.write("process has been called\n")

end