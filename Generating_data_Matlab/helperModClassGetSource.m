function src = helperModClassGetSource(modType, sps, spf)
%helperModClassGetSource Source selector for modulation types
%    SRC = helperModClassGetSource(TYPE,SPS,SPF,FS) returns the data source
%    for the modulation type TYPE, with the number of samples per symbol
%    SPS, the number of samples per frame SPF, and the sampling frequency
%    FS.
%   
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

switch modType
  case {"BPSK","GFSK","CPFSK"}
    M = 2;
    src = @()randi([0 M-1],spf/sps,1);
  case {"QPSK","PAM_4"}
    M = 4;
    src = @()randi([0 M-1],spf/sps,1);
  case "PSK_8"
    M = 8;
    src = @()randi([0 M-1],spf/sps,1);
  case "QAM16"
    M = 16;
    src = @()randi([0 M-1],spf/sps,1);
  case "QAM64"
    M = 64;
    src = @()randi([0 M-1],spf/sps,1);
end
end