%%
% Setting VNA 09.11.2023 DR MG
%
%%
function instrObj = setVNA_v3_2ports(settings)

instrumentVISAAddress = "TCPIP0::localhost::hislip1::INSTR"; % hislip0 (check in connection expert)
instrObj = visadev(instrumentVISAAddress);


instrObj.InputBufferSize = 10e6;
instrObj.OutputBufferSize = 10e6;
instrObj.ByteOrder = 'little-endian';
fopen(instrObj);
clrdevice(instrObj);

fprintf(instrObj,'*CLS');
fprintf(instrObj, 'FORMat REAL,64');
fprintf(instrObj, 'FORMat:BORDer SWAP');

%% Set power port
%By default the VNA is set to 0dBm
char=['SOURce:POWer ',num2str(settings.dBmPower,2)];
fprintf(instrObj,char);

%% set average
fprintf(instrObj,'SENS:AVERage:STATe OFF');

%% set bandwidth
fprintf(instrObj,['SENS:BWID ',num2str(settings.BW)]);

%% Set the number of points
fprintf(instrObj, sprintf('SENSe:SWEep:POINts %s',num2str(settings.numPoints)));

% Set the frequency ranges
fprintf(instrObj, sprintf('SENSe:FREQuency:STARt %sHz',num2str(settings.frequencyRange(1))));
fprintf(instrObj, sprintf('SENSe:FREQuency:STOP %sHz',num2str(settings.frequencyRange(2))));

%% set average
fprintf(instrObj,'SENS:AVERage:STATe ON');
fprintf(instrObj,sprintf('SENS:AVERage:Count %s',  num2str(settings.avg_N)));

%% Turn on windows
fprintf(instrObj,sprintf('DISPlay:WINDow1:STATE ON'));
fprintf(instrObj,sprintf('DISPlay:WINDow2:STATE ON'));


%% Delete all meas
fprintf(instrObj,'CALCulate:PARameter:DELete:ALL');

% antenna 1
fprintf(instrObj,'CALCulate:PARameter:DEFine:EXT ''s11'',''S11''');
fprintf(instrObj,'CALCulate:PARameter:DEFine:EXT ''s21'',''S21''');

% antenna 2
fprintf(instrObj,'CALCulate:PARameter:DEFine:EXT ''s12'',''S12''');
fprintf(instrObj,'CALCulate:PARameter:DEFine:EXT ''s22'',''S22''');


%% Associate FEED to the windows and traces
fprintf(instrObj,sprintf('DISPlay:WINDow1:TRACe1:FEED ''s11'''));
fprintf(instrObj,sprintf('DISPlay:WINDow1:TRACe2:FEED ''s22'''));

fprintf(instrObj,sprintf('DISPlay:WINDow2:TRACe5:FEED ''s12'''));

end