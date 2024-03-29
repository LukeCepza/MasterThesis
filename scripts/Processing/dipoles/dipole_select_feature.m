% primary_sensory_MNI = [-41 -27 47];         
% secondary_sensory = [-15 -33 48];
% thalamus = [-10 -19 6];
% pathway = [-28 -19 29];
% pathway_2 = [-16 -19 16];

primary_sensory = [-40 -31 52];         
secondary_sensory = [-52 -30 32];
thalamus = [-9 -20 8];
pathway = [-27 -18 28];
pathway_2 = [-15 -19 17];

dips_cords = {EEG.dipfit.model.posxyz};
p1 = zeros(length(dips_cords),3);

for dip = 1:length(dips_cords)
    try
    p1(dip,:) = dips_cords{dip};
    catch
    p1(dip,:) = [inf,inf,inf];
    end
end

p1_l1 = calculateEuclideanDistance3D(p1,primary_sensory);
p1_l2 = calculateEuclideanDistance3D(p1,secondary_sensory);
p1_l3 = calculateEuclideanDistance3D(p1,thalamus);
p1_l4 = calculateEuclideanDistance3D(p1,pathway);
p1_l5 = calculateEuclideanDistance3D(p1,pathway_2);
%%
p1_l1_id = get_minidx(p1_l1);
p1_l2_id = get_minidx(p1_l2);
p1_l3_id = get_minidx(p1_l3);
p1_l4_id = get_minidx(p1_l4);
p1_l5_id = get_minidx(p1_l5);
ids = [p1_l1_id,p1_l2_id,p1_l3_id,p1_l4_id,p1_l5_id];
ids = reshape(ids(1:3,:),[],1);
D1 = mode(ids(1:5)');
ids(ids == D1) = 0;
D2 = mode(ids(ids(1:10)' ~= 0));
ids(ids == D2) = 0;
D3 = mode(ids(ids(1:15)'~= 0));

%%
tag = [EEG.event.edftype];
tag = tag(1:2:end) -4;
exp = ones(length(tag), 4);
exp(:,3) = tag;
exp(:,1) = 5;

IC_obs = squeeze(mean(EEG.icaact,1));
exp = [exp,IC_obs'];

writematrix(exp, fullfile('D:\shared_git\MaestriaThesis\FeaturesTabs','pp01_t7.csv'));
%%

[pxx,freqs] = pwelch(exp(:,5:end)', 250,125,[], 250);
output_pxx = [exp(:,1:4),pxx'];
writematrix(output_pxx, fullfile('D:\shared_git\MaestriaThesis\FeaturesTabs','pp01_t8.csv'));

function Aids = get_minidx(A)
    Aids = zeros(length(A),1);
    for i = 1:length(A)
        [~,Aid] = min(A);
        A(Aid) = inf;
        Aids(i) = Aid;
    end
end

function distance = calculateEuclideanDistance3D(point1, point2)
    % Extract coordinates for easier reading
    x1 = point1(:,1);
    y1 = point1(:,2);
    z1 = point1(:,3);
    x2 = point2(1);
    y2 = point2(2);
    z2 = point2(3);
    
    % Calculate the Euclidean distance
    distance = sqrt((x2 - x1).^2 + (y2 - y1).^2 + (z2 - z1).^2);
end
