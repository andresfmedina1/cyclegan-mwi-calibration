function [s_tot, timePausa] = meas2x2_trigger_MGDR(instrObj, settingsVNA)
% MEAS2X2_TRIGGER_MGDR
% Function to measure a 2x2 S-parameter measurement using a
% Vector Network Analyzer (VNA) with a manual trigger (CO & RM 2024).
%
% INPUTS:
%   instrObj     - VISA/TCPIP instrument object (VNA connection)
%   settingsVNA  - Struct containing VNA settings:
%                  .numPoints -> number of frequency points
%
% OUTPUTS:
%   s_tot        - 2x2xN S-parameters:
%                  s_tot(1,1,:) = S11
%                  s_tot(2,1,:) = S21
%                  s_tot(2,2,:) = S22
%                  s_tot(1,2,:) = S12
%
%   timePausa    - Wait time to ensure measurement completion
 

%% Get sweep time and define waiting time
sweep_time = query(instrObj, 'SENS:SWE:TIME?');
sweep_time = str2double(sweep_time);
timePausa = 3 * sweep_time;

numPoints = settingsVNA.numPoints;

%% Initialize output matrix
s_tot = zeros(2, 2, numPoints);

%% Start measurement
fprintf(instrObj, 'OUTP:STAT ON');
fprintf(instrObj, 'INIT:IMM');

pause(timePausa);

fprintf(instrObj, 'OUTP:STAT OFF');

%% ----------- S11 -----------
fprintf(instrObj, 'CALC:PAR:SEL ''s11''');

fprintf(instrObj, 'CALC:DATA? SDATA');
raw_data = binblockread(instrObj, 'double');
fread(instrObj, 1);

data_complex = complex(raw_data(1:2:end), raw_data(2:2:end));
s_tot(1,1,:) = reshape(data_complex, 1, 1, numPoints);

%% ----------- S21 -----------
fprintf(instrObj, 'CALC:PAR:SEL ''s21''');

fprintf(instrObj, 'CALC:DATA? SDATA');
raw_data = binblockread(instrObj, 'double');
fread(instrObj, 1);

data_complex = complex(raw_data(1:2:end), raw_data(2:2:end));
s_tot(2,1,:) = reshape(data_complex, 1, 1, numPoints);

%% ----------- S22 -----------
fprintf(instrObj, 'CALC:PAR:SEL ''s22''');

fprintf(instrObj, 'CALC:DATA? SDATA');
raw_data = binblockread(instrObj, 'double');
fread(instrObj, 1);

data_complex = complex(raw_data(1:2:end), raw_data(2:2:end));
s_tot(2,2,:) = reshape(data_complex, 1, 1, numPoints);

%% ----------- S12 -----------
fprintf(instrObj, 'CALC:PAR:SEL ''s12''');

fprintf(instrObj, 'CALC:DATA? SDATA');
raw_data = binblockread(instrObj, 'double');
fread(instrObj, 1);

data_complex = complex(raw_data(1:2:end), raw_data(2:2:end));
s_tot(1,2,:) = reshape(data_complex, 1, 1, numPoints);

%% Restore output state
fprintf(instrObj, 'OUTP:STAT ON');

end