 function outpoints = tal2mni(inpoints)
 

 [s1,s2] = size(inpoints);
 if s1 ~= 3
     error('input must be a 3xN matrix')
 end
 
 % Transformation matrices, different zooms above/below AC
 M2T =  mni2tal_matrix;
 
 inpoints = [inpoints; ones(1, size(inpoints, 2))];
 
 tmp = inpoints(3,:) < 0;  % 1 if below AC
 
inpoints(:,  tmp) = (M2T.rotn * M2T.downZ) \ inpoints(:,  tmp);
inpoints(:, ~tmp) = (M2T.rotn * M2T.upZ  ) \ inpoints(:, ~tmp);
 
outpoints = inpoints(1:3, :);
 end
 