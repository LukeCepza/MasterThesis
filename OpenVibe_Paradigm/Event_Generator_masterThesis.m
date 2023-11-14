rng('default');
s = rng(0);

time_init = 15;
repsair = 20;
repsvib = 20;
repscaress = 20;

stim_air = 0; %Duracion del estimulo
epoch_dur = 4;
gepoch_dur = 24;

feedback = "2000"
pinkNoise = "1999" 
%% Air
combs = perms(["33028","33029","33030","33031"]);
l_stop = "33043";

% Mix rows
c_shu = combs(randperm(size(combs,1)),:);

gep = (0:24:24*repsvib)+time_init
fileID = fopen('Events_KevinThesis.txt','w');

for i = 0:(repsvib-1)
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + pinkNoise +"," + num2str(0 + gep(i+1)) +",0)");    
    for t = 1:4
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + c_shu(i+1,t)+ "," + num2str(4*t - 2 + gep(i+1)) +",0)");
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + l_stop + "," + num2str(4*t + gep(i+1)) +",0)");
    end
end
%% Vibration
combs = perms(["33032","33033","33034","33035"])
l_stop = "33042"
% Mix rows
c_shu= combs(randperm(size(combs,1)),:)
gep = (repsvib*24+time_init):24:((repsvib+repsair)*24)+time_init

for i = 0:(repsair-1)
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + pinkNoise +"," + num2str(0 + gep(i+1)) +",0)");    
    for t = 1:4
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + c_shu(i+1,t)+ "," + num2str(4*t - 2 + gep(i+1)) +",0)");
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + l_stop + "," + num2str(4*t - 1.5 + gep(i+1)) +",0)");
    end
end

%% Caress
combs = perms(["33024","33025","33026","33027"])
l_stop = "0200"
% Mix rows
c_shu= combs(randperm(size(combs,1)),:)

gep = (24*(repsair+repsvib)+time_init):24:24*(repsvib+repsair+repscaress)+time_init

for i = 0:(repscaress-1)
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + pinkNoise +"," + num2str(0 + gep(i+1)) +",0)");    
    for t = 1:4
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + c_shu(i+1,t)+ "," + num2str(4*t-2 + gep(i+1)) +",0)");
    fprintf(fileID,'%s\n',"box:send_stimulation(1," + l_stop + "," + num2str(4*t -1 + gep(i+1)) +",0)");
    end
end
