function defuzzfun = customdefuzz(xmf, ymf)
    %%COS is the implementation of the Center of Sums (COS) for Mamdani
    %%deffuzification method for matlab
    % xmf is the vector of values in the membership function input range
    % ymf is the value of the membership function at xmf
 
    % find the mf's maximum value and it's location
    % the pks will give us the flat surface
    % For the flat peak, the function returns only the point with lowest index
    [pks, locs] = findpeaks(ymf, xmf);
 
    % The following plot will gifizve us a better understanding of the way the
    % method works. It is a comment in order to speed up the execution and
    % ib better to be used for demonstration
    %     findpeaks(ymf,xmf);
    %     hold on;
    %     title('Membership functions - input to COS deffuzifier');
    %     xlabel('xmf');
    %     ylabel('ymf');
    %     saveas(gcf, 'input_to_cos.png');
 
    weightedAreaSum = 0;
    areaSum = 0;
 
    for i = 1:length(pks)
        % find the position in the xmf/ymf array that the peak occurs
        start_index = find(xmf == locs(i));
     
        % find the distance of the flat area. It is the length that the ymf
        % value is the same. If it changed it is not flat anymore
        % Index of the last xmf that is in the flat surface
        end_index = start_index + length(ymf(ymf == ymf(start_index))) - 1; % minus one, since one time is the start_index already
 
        upper = abs(xmf(end_index) - xmf(start_index)); % calculate the length of the small base of the trapezoid
 
        area = (0.66 + upper) * pks(i) / 2; % trapezoidal surface
 
        % Calculate center of area for the trapezoid
        center_of_area = (xmf(end_index) + xmf(start_index) ) / 2;
 
        % COS formula
        areaSum = areaSum + area;
        weightedAreaSum = weightedAreaSum + area * center_of_area;
    end

    % Check the NL and PL as findpeaks cannot locate them
    if (ymf(1) ~= 0) % NL
        start_index = 1;
        end_index = start_index + length(ymf(ymf == ymf(start_index))) - 1; % calculate the number of the times that ymf(1) appears in the array

        upper = abs(xmf(end_index) - xmf(start_index)); % upper base of the trapezoid
    
        area = (0.33 + upper) * ymf(1) / 2; % trapezoidal surface

        function1 = @(w) (ymf(1) * w); % constant function
        function2 = @(w) ((ymf(1) / (xmf(end_index) + 0.66)) .* (w + 0.66) .* w); % y = ax + b, with values for a and b

        % the total value is the sum of the integrals
        q = integral(function1, - 1, xmf(end_index)) + integral(function2, xmf(end_index), - 0.66);

        center_of_area =  q / area;
        areaSum = areaSum + area;
        weightedAreaSum = weightedAreaSum + center_of_area * area;
    end

    if (ymf(end) ~= 0) % PL
        start_index = 101;
        end_index = start_index - length(ymf(ymf == ymf(start_index))) - 1; % same as before
        upper = abs(- xmf(end_index) + xmf(start_index));

        area = (0.33 + upper) * ymf(end) / 2;

        function1 = @(w) ((ymf(end) / (xmf(end_index) - 0.66)) .* (w - 0.66) .* w);
        function2 = @(w) (ymf(end) * w);

        q = integral(function1, 0.66, xmf(end_index)) + integral(function2, xmf(end_index), 1);

        center_of_area = q / area;
        areaSum = areaSum + area;
        weightedAreaSum = weightedAreaSum + center_of_area * area;
    end

    % Defuzzified value
    defuzzfun = weightedAreaSum / areaSum;

end
