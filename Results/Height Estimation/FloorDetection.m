clear
clc
opts = detectImportOptions('Floor_Detect_2020-8-19_14-08-02.csv');
M2 = readmatrix('Floor_Detect_2020-8-19_14-08-02.csv',opts);
opts = detectImportOptions('Floor_Detect_2020-8-19_14-05-52.csv');
M1 = readmatrix('Floor_Detect_2020-8-19_14-05-52.csv',opts);
input_data = [M1(:,1)' M2(:,1)']';
avg_readings = 0;
avgT_0 = 0;
avgT_2 = 0;
enterLoop2 = false;
currentTime = 0;
h0 = 3.52;
temperature = 27.0;
segma = 1/273.15;
eqConst = 18410.183;
thetaT = 0.01;
N1 = 0; 
N0 = 5; 
num2 = 0;
num1 = 0;
firstRun = true;
currentFloor = 0;
allFloors = 0;
pstart = 0;
pend = 0;
prevI = 1;
output_file(:,1) = input_data;
output_file(:,2) = 0;
avg_countHelper = 0;
for i = 1:length(input_data)
    avg_readings(abs(i - avg_countHelper)) = input_data(i);
    if(length(avg_readings) >= 6)
        avg_countHelper = avg_countHelper + 6;
        avgT_0 = 0;
        avgT_0 = mean(avg_readings);
        avg_readings = 0;
        if(firstRun)
            avgT_2 = avgT_0;
            firstRun = false;
        end
        if (currentTime == 2)
            if(~enterLoop2)
                if(abs(avgT_0 - avgT_2) > thetaT)
                    if(num1 == 0)
                        avgT_5 = avgT_0;
                    end
                    if(num1 == N0)
                        pstart = avgT_5;
                        enterLoop2 = true;
                        output_file(i,2) = pstart;
                        num1 = 0;
                    else
                        num1 = num1 +1;
                    end
                else
                    output_file(i,4) = abs(avgT_0 - avgT_2);
                    output_file(i,5) = 10;
                    num1 = 0;
                end
                else
                    if(abs(avgT_0 - avgT_2) < thetaT)
                        if(num2 ==0)
                            avgT_5 = avgT_0;
                        end
                        if (num2 == N1)
                            pend = avgT_5;
                            enterLoop2 = false;
                            num2 = 0;
                            output_file(i,3) = pend;
                            output_file(i,4) = abs(avgT_0 - avgT_2);
                            allFloors(prevI:i) = currentFloor;
                            currentFloor = currentFloor + round((1/h0) .* eqConst .* (1+segma * temperature) * log10(pstart/pend))
                            output_file(i,6) = currentFloor;
                            prevI = i;
                        else
                            num2 = num2 +1;
                        end
                    else
                        output_file(i,4) = abs(avgT_0 - avgT_2);
                        output_file(i,5) = 20;
                        num2 = 0;
                    end
                end
                avgT_2 = avgT_0;
                currentTime = 0;
        end
            currentTime = currentTime + 1;
    end
end
time_sample = 1: length(input_data);
allFloors(prevI:i) = currentFloor;
t = tiledlayout(2,1)
lines_x = [127 246 378 534 687 821 977 1108];
ax1 = nexttile;
plot(time_sample, input_data)
ylabel('Pressure[hPa]')
% for i = 1:length(lines_x)
%     xline(lines_x(i),'r')
% end
nexttile
plot(time_sample, allFloors)
ylabel('Floor Number')

title(t,'Floor Detection')
xlabel(t,'Sample')

% Move plots closer together
xticklabels(ax1,{})
t.TileSpacing = 'compact';                    