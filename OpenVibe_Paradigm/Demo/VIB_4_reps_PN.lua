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
box:send_stimulation(1,33034,12,0)
box:send_stimulation(1,33042,12.5,0)
box:send_stimulation(1,33035,16,0)
box:send_stimulation(1,33042,16.5,0)
box:send_stimulation(1,33032,20,0)
box:send_stimulation(1,33042,20.5,0)
box:send_stimulation(1,33033,24,0)
box:send_stimulation(1,33042,24.5,0)
box:send_stimulation(1,1999,34,0)
box:send_stimulation(1,33033,36,0)
box:send_stimulation(1,33042,36.5,0)
box:send_stimulation(1,33034,40,0)
box:send_stimulation(1,33042,40.5,0)
box:send_stimulation(1,33032,44,0)
box:send_stimulation(1,33042,44.5,0)
box:send_stimulation(1,33035,48,0)
box:send_stimulation(1,33042,48.5,0)
box:send_stimulation(1,1999,58,0)
box:send_stimulation(1,33034,60,0)
box:send_stimulation(1,33042,60.5,0)
box:send_stimulation(1,33033,64,0)
box:send_stimulation(1,33042,64.5,0)
box:send_stimulation(1,33032,68,0)
box:send_stimulation(1,33042,68.5,0)
box:send_stimulation(1,33035,72,0)
box:send_stimulation(1,33042,72.5,0)
box:send_stimulation(1,1999,82,0)
box:send_stimulation(1,33034,84,0)
box:send_stimulation(1,33042,84.5,0)
box:send_stimulation(1,33032,88,0)
box:send_stimulation(1,33042,88.5,0)
box:send_stimulation(1,33035,92,0)
box:send_stimulation(1,33042,92.5,0)
box:send_stimulation(1,33033,96,0)
box:send_stimulation(1,33042,96.5,0)

box:send_stimulation(1,32770,100,0)   -- End stimuli
box:send_stimulation(1,32770,101,0) 
	box:sleep()
	io.write("process has been called\n")

end