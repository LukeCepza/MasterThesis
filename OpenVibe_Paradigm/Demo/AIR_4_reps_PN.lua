-- this function is called when the box is initialized
function initialize(box)
end

-- this function is called when the box is uninitialized
function uninitialize(box)
	io.write("uninitialize has been calle d\n")
end

function process(box)
	box:send_stimulation(1,33084,3,0) --Start LSL communication

box:send_stimulation(1,1998,10,0)
box:send_stimulation(1,33028,13,0)
box:send_stimulation(1,33043,15,0)
box:send_stimulation(1,33030,18,0)
box:send_stimulation(1,33043,20,0)
box:send_stimulation(1,33029,23,0)
box:send_stimulation(1,33043,25,0)
box:send_stimulation(1,33031,28,0)
box:send_stimulation(1,33043,30,0)
box:send_stimulation(1,1998,38,0)
box:send_stimulation(1,33031,41,0)
box:send_stimulation(1,33043,43,0)
box:send_stimulation(1,33028,46,0)
box:send_stimulation(1,33043,48,0)
box:send_stimulation(1,33029,51,0)
box:send_stimulation(1,33043,53,0)
box:send_stimulation(1,33030,56,0)
box:send_stimulation(1,33043,58,0)
box:send_stimulation(1,1998,66,0)
box:send_stimulation(1,33031,69,0)
box:send_stimulation(1,33043,71,0)
box:send_stimulation(1,33029,74,0)
box:send_stimulation(1,33043,76,0)
box:send_stimulation(1,33030,79,0)
box:send_stimulation(1,33043,81,0)
box:send_stimulation(1,33028,84,0)
box:send_stimulation(1,33043,86,0)
box:send_stimulation(1,1998,94,0)
box:send_stimulation(1,33029,97,0)
box:send_stimulation(1,33043,99,0)
box:send_stimulation(1,33030,102,0)
box:send_stimulation(1,33043,104,0)
box:send_stimulation(1,33028,107,0)
box:send_stimulation(1,33043,109,0)
box:send_stimulation(1,33031,112,0)
box:send_stimulation(1,33043,114,0)


box:send_stimulation(1,32770,117,0)   -- End stimuli
box:send_stimulation(1,32770,118,0) 
	box:sleep()
	io.write("process has been called\n")

end