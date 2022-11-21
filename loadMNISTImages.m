% Author:       Clum, Mixon, Villar, Xie.
% Filename:     loadMNISTImages.m
% Last edited:  21 November 2022 
% Description:  loadMNISTImages returns a 28x28x[number of MNIST images] 
%               matrix containing the raw MNIST images [1].
% 
%
% Input: 
%               -filename: 
%               A MNIST data file.
%
% Output: 
%               -images: 
%               A d x n data matrix, where d = 28 x 28 denotes the 
%               dimension of the data and n denotes the number of images.
%               
% References:
% [1] Y. LeCun, C. Cortes, MNIST handwritten digit database.
% -------------------------------------------------------------------------
function images = loadMNISTImages(filename)


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end
