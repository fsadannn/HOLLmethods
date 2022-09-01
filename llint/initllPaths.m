function initllPaths(shouldIncludeTests)

if nargin==0
    shouldIncludeTests=false;
end

abspath = which('initllPaths');
pos = strfind(abspath, filesep);
pos = pos(end);
abspath = abspath(1:pos - 1);
if exist([ abspath filesep 'llint' ], 'dir')
    relpath = [ abspath filesep ];
else
    relpath = [ abspath filesep '..' filesep ];
end
addpath([relpath 'llint']);

if shouldIncludeTests
%      TODO: add path that are only for testing propouses
    krilov_pth = {'krilov'};
else
    krilov_pth = {'krilov'};
end

c_code_pth = {'c_code'};

dirs = cat(2,krilov_pth,c_code_pth);

for i = length(dirs):-1:1
    dir = [ relpath 'llint' filesep dirs{i} ];
    if isempty(strfind(path, [ dir pathsep ]))
        if length(dir) > 1 && strcmp(dir(end-1:end), '//')
            addpath(genpath(dir(1:end - 2)));
        else
            addpath(dir);
        end
    end
end

end
