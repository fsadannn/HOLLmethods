lapacklib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft','libmwlapack.lib');
blaslib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft','libmwblas.lib');

mex('-v', '-largeArrayDims', 'expm64v4.c', blaslib, lapacklib)
file = sprintf('expm64v4.%s',mexext);
[status,message,messageId] = movefile(['.',filesep,file],['.',filesep,'..',filesep,file ],'f');

mex('-v', '-largeArrayDims', 'expm64v41.c', blaslib, lapacklib)
file = sprintf('expm64v41.%s',mexext);
[status,message,messageId] = movefile(['.',filesep,file],['.',filesep,'..',filesep,file ],'f');

mex('-v', '-R2018a', 'expm64v41_complex.c', blaslib, lapacklib)
file = sprintf('expm64v41_complex.%s',mexext);
[status,message,messageId] = movefile(['.',filesep,file],['.',filesep,'..',filesep,file ],'f');

mex('-v', '-R2018a', 'expm64v42_complex.c', blaslib, lapacklib)
file = sprintf('expm64v42_complex.%s',mexext);
[status,message,messageId] = movefile(['.',filesep,file],['.',filesep,'..',filesep,file ],'f');

mex('-v', '-R2018a', 'expm64v42p_complex.c', blaslib, lapacklib)
file = sprintf('expm64v42p_complex.%s',mexext);
[status,message,messageId] = movefile(['.',filesep,file],['.',filesep,'..',filesep,file ],'f');
