
 function M2T = mni2tal_matrix()
 
 % rotn  = spm_matrix([0 0 0 0.05]); % similar to Rx(eye(3),-0.05), DLW
 M2T.rotn  = [      1         0         0         0;
                    0    0.9988    0.0500         0;
                    0   -0.0500    0.9988         0;
                    0         0         0    1.0000 ];
 
 
 % upz   = spm_matrix([0 0 0 0 0 0 0.99 0.97 0.92]);
 M2T.upZ   = [ 0.9900         0         0         0;
                    0    0.9700         0         0;
                    0         0    0.9200         0;
                    0         0         0    1.0000 ];
 
 
 % downz = spm_matrix([0 0 0 0 0 0 0.99 0.97 0.84]);
 M2T.downZ = [ 0.9900         0         0         0;
                    0    0.9700         0         0;
                    0         0    0.8400         0;
                    0         0         0    1.0000 ];
 
 return