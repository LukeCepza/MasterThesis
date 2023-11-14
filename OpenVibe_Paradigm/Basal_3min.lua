-- this function is called when the box is initialized
function initialize(box)
end

-- this function is called when the box is uninitialized
function uninitialize(box)
	io.write("uninitialize has been calle d\n")
end

function process(box)

box:send_stimulation(1,32770,178,0)   -- End stimuli
box:send_stimulation(1,32770,180,0) 
	box:sleep()
	io.write("process has been called\n")

end