% Design reflector-backed dipole antenna element
fq = 10e9; % 10 GHz
myelement = design(reflector,fq);
myelement.Exciter = design(myelement.Exciter,fq);

% Tilt antenna element to radiate in xy-plane, with boresight along x-axis
myelement.Tilt = 90;
myelement.TiltAxis = 'y';
myelement.Exciter.Tilt = 90;
myelement.Exciter.TiltAxis = 'y';

Create a 7-by-7 Rectangular Antenna Array
Use Phased Array System Toolbox to create a 7-by-7 rectangular array from the antenna element. Specify the array normal to direct radiation in the x-axis direction.
% Create 7-by-7 antenna array                            
nrow = 7;
ncol = 7;
myarray = phased.URA('Size',[nrow ncol],'Element',myelement);
    
% Define element spacing to be half-wavelength at 10 GHz, and specify 
% array plane as yz-plane, which directs radiation in x-axis direction
lambda = physconst('lightspeed')/fq;
drow = lambda/2;
dcol = lambda/2; 
myarray.ElementSpacing = [drow dcol];
myarray.ArrayNormal = 'x';
    
% Display radiation pattern
f = figure;
az = -180:1:180;
el = -90:1:90;  
pattern(myarray,fq,az,el)
tx = txsite('Name','Washington Monument',...
    'Latitude',38.88949, ...
    'Longitude',-77.03523, ...
    'Antenna',myarray,...
    'AntennaHeight',169', ...
    'TransmitterFrequency',fq,...
    'TransmitterPower',1);
if isvalid(f)
    close(f)
end
viewer = siteviewer;
show(tx)
rxNames = {...
    'Brentwood Hamilton Field', ...
    'Nationals Park', ...
    'Union Station', ...
    'Georgetown University', ...
    'Arlington Cemetery'};

% Define coordinates for receiver sites
rxLocations = [...
    38.9080 -76.9958; ...
    38.8731 -77.0075; ...
    38.8976 -77.0062; ...
    38.9076 -77.0722; ...
    38.8783 -77.0685];

% Create array of receiver sites. Each receiver has a sensitivity of -75 dBm.
rxs = rxsite('Name',rxNames, ...
    'Latitude',rxLocations(:,1), ...
    'Longitude',rxLocations(:,2), ...
    'ReceiverSensitivity',-75);
startTaper = myarray.Taper;

% Define angles over which to perform sweep
azsweep = -30:10:30;

% Set up tapering window and steering vector
N = nrow*ncol;
nbar = 5; 
sll = -20;
sltaper = taylorwin(N,nbar,sll)';
steeringVector = phased.SteeringVector('SensorArray',myarray);

% Sweep the angles and show the antenna pattern for each
for az = azsweep
    sv = steeringVector(fq,[az; 0]);    
    myarray.Taper = sltaper.*sv';
    
    % Update the radiation pattern. Use a larger size so the pattern is visible among the antenna sites.
    pattern(tx, 'Size', 2500,'Transparency',1);
end
myarray.Taper = startTaper;

% Define signal strength levels (dBm) and corresponding colors
strongSignal = -65;
mediumSignal = -70;
weakSignal = -75;
sigstrengths = [strongSignal mediumSignal weakSignal];
sigcolors = {'red' 'yellow' 'green'};

% Show the tx pattern
pattern(tx,'Size',500)

% Display coverage map out to 6 km
maxRange = 6000;
coverage(tx, ...
    'SignalStrengths',sigstrengths, ...
    'Colors',sigcolors, ...
    'MaxRange',maxRange)
for az = azsweep
    % Calculate and assign taper from steering vector
    sv = steeringVector(fq,[az; 0]);    
    myarray.Taper = sltaper.*sv';
    
    % Update the tx pattern
    pattern(tx,'Size',500)

    % Update coverage map
    coverage(tx, ...
        'SignalStrengths',sigstrengths, ...
        'Colors',sigcolors, ...
        'MaxRange',maxRange)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
centerSite = txsite('Name','MathWorks Glasgow', ...
    'Latitude',55.862787,...
    'Longitude',-4.258523);

% Initialize arrays for distance and angle from center location to each cell site, where
% each site has 3 cells
numCellSites = 19;
siteDistances = zeros(1,numCellSites);
siteAngles = zeros(1,numCellSites);

% Define distance and angle for inner ring of 6 sites (cells 4-21)
isd = 200; % Inter-site distance
siteDistances(2:7) = isd;
siteAngles(2:7) = 30:60:360;

% Define distance and angle for middle ring of 6 sites (cells 22-39)
siteDistances(8:13) = 2*isd*cosd(30);
siteAngles(8:13) = 0:60:300;

% Define distance and angle for outer ring of 6 sites (cells 40-57)
siteDistances(14:19) = 2*isd;
siteAngles(14:19) = 30:60:360;

Define Cell Parameters
Each cell site has three transmitters corresponding to each cell. Create arrays to define the names, latitudes, longitudes, and antenna angles of each cell transmitter. 
% Initialize arrays for cell transmitter parameters
numCells = numCellSites*3;
cellLats = zeros(1,numCells);
cellLons = zeros(1,numCells);
cellNames = strings(1,numCells);
cellAngles = zeros(1,numCells);

% Define cell sector angles
cellSectorAngles = [30 150 270];

% For each cell site location, populate data for each cell transmitter
cellInd = 1;
for siteInd = 1:numCellSites
    % Compute site location using distance and angle from center site
    [cellLat,cellLon] = location(centerSite, siteDistances(siteInd), siteAngles(siteInd));
    
    % Assign values for each cell
    for cellSectorAngle = cellSectorAngles
        cellNames(cellInd) = "Cell " + cellInd;
        cellLats(cellInd) = cellLat;
        cellLons(cellInd) = cellLon;
        cellAngles(cellInd) = cellSectorAngle;
        cellInd = cellInd + 1;
    end
end
fq = 4e9; % Carrier frequency (4 GHz) for Dense Urban-eMBB
antHeight = 25; % m
txPowerDBm = 44; % Total transmit power in dBm
txPower = 10.^((txPowerDBm-30)/10); % Convert dBm to W

% Create cell transmitter sites
txs = txsite('Name',cellNames, ...
    'Latitude',cellLats, ...
    'Longitude',cellLons, ...
    'AntennaAngle',cellAngles, ...
    'AntennaHeight',antHeight, ...
    'TransmitterFrequency',fq, ...
    'TransmitterPower',txPower);

% Launch Site Viewer
viewer = siteviewer;

% Show sites on a map
show(txs);
viewer.Basemap = 'topographic';
azvec = -180:180;
elvec = -90:90;
Am = 30; % Maximum attenuation (dB)
tilt = 0; % Tilt angle
az3dB = 65; % 3 dB bandwidth in azimuth
el3dB = 65; % 3 dB bandwidth in elevation

% Define antenna pattern
[az,el] = meshgrid(azvec,elvec);
azMagPattern = -12*(az/az3dB).^2;
elMagPattern = -12*((el-tilt)/el3dB).^2;
combinedMagPattern = azMagPattern + elMagPattern;
combinedMagPattern(combinedMagPattern<-Am) = -Am; % Saturate at max attenuation
phasepattern = zeros(size(combinedMagPattern));

% Create antenna element
antennaElement = phased.CustomAntennaElement(...
    'AzimuthAngles',azvec, ...
    'ElevationAngles',elvec, ...
    'MagnitudePattern',combinedMagPattern, ...
    'PhasePattern',phasepattern);
   
% Display radiation pattern
f = figure;
pattern(antennaElement,fq);
for tx = txs
    tx.Antenna = antennaElement;
end

% Define receiver parameters using Table 8-2 (b) of Report ITU-R M.[IMT-2020.EVAL] 
bw = 20e6; % 20 MHz bandwidth
rxNoiseFigure = 7; % dB
rxNoisePower = -174 + 10*log10(bw) + rxNoiseFigure;
rxGain = 0; % dBi
rxAntennaHeight = 1.5; % m

% Display SINR map
if isvalid(f)
    close(f)
end
sinr(txs,'freespace', ...
    'ReceiverGain',rxGain, ...
    'ReceiverAntennaHeight',rxAntennaHeight, ...
    'ReceiverNoisePower',rxNoisePower, ...    
    'MaxRange',isd, ...
    'Resolution',isd/20)
nrow = 8;
ncol = 8;

% Define element spacing
lambda = physconst('lightspeed')/fq;
drow = lambda/2;
dcol = lambda/2;

% Define taper to reduce sidelobes 
dBdown = 30;
taperz = chebwin(nrow,dBdown);
tapery = chebwin(ncol,dBdown);
tap = taperz*tapery.'; % Multiply vector tapers to get 8-by-8 taper values

% Create 8-by-8 antenna array
cellAntenna = phased.URA('Size',[nrow ncol], ...
    'Element',antennaElement, ...
    'ElementSpacing',[drow dcol], ...
    'Taper',tap, ...
    'ArrayNormal','x');
    
% Display radiation pattern
f = figure;
pattern(cellAntenna,fq);
downtilt = 15;
for tx = txs
    tx.Antenna = cellAntenna;
    tx.AntennaAngle = [tx.AntennaAngle; -downtilt];
end

% Display SINR map
if isvalid(f)
    close(f)
end
sinr(txs,'freespace', ...
    'ReceiverGain',rxGain, ...
    'ReceiverAntennaHeight',rxAntennaHeight, ...
    'ReceiverNoisePower',rxNoisePower, ...    
    'MaxRange',isd, ...
    'Resolution',isd/20)
sinr(txs,'close-in', ...
    'ReceiverGain',rxGain, ...
    'ReceiverAntennaHeight',rxAntennaHeight, ...
    'ReceiverNoisePower',rxNoisePower, ...    
    'MaxRange',isd, ...
    'Resolution',isd/20)
patchElement = design(patchMicrostrip,fq);
patchElement.Width = patchElement.Length;
patchElement.Tilt = 90;
patchElement.TiltAxis = [0 1 0];

% Display radiation pattern
f = figure;
pattern(patchElement,fq)
cellAntenna.Element = patchElement;

% Display SINR map
if isvalid(f)
    close(f)
end
sinr(txs,'close-in',...
    'ReceiverGain',rxGain, ...
    'ReceiverAntennaHeight',rxAntennaHeight, ...
    'ReceiverNoisePower',rxNoisePower, ...    
    'MaxRange',isd, ...
    'Resolution',isd/20)