% Edge detection quantitative evaluation :
clear;
addpath(genpath('D:\Evaluation_TUD\edge_detection_related\bench_script\pdollar_edges'));
addpath(genpath('D:\Evaluation_TUD\edge_detection_related\bench_script\pdollar_toolbox'));% Give necessary path here having the pdollar 
dataset_name = 'BIPED';
model = 'dxn';

img_dir = 'D:\Evaluation_TUD\edge_detection_related\result\img';
gt_dir = 'D:\Evaluation_TUD\edge_detection_related\result\gt';
pred_dir = 'D:\Evaluation_TUD\edge_detection_related\result\pred';
nms_pred_dir = 'D:\Evaluation_TUD\edge_detection_related\result\nms_pred';
values_dir = 'D:\Evaluation_TUD\edge_detection_related\result\values';
results = {};

list_pred = dir(fullfile(pred_dir, '*.png'));
list_gt = dir(fullfile(gt_dir, '*.png'));
list_values = dir(fullfile(values_dir, '*.txt'));
list_pp = dir(fullfile(nms_pred_dir, '*.png'));

n_res = length(list_pred);

%nms performed on the predicted edge maps
for i=1:n_res
    edge = imread(fullfile(pred_dir, list_pred(i).name));
    edge = 1-single(edge)/255;
    [Ox, Oy] = gradient2(convTri(edge,4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    edge = edgesNmsMex(edge,O,1,5,1.03,8);
    %edge = imcomplement(edge);
    imwrite(edge, fullfile(nms_pred_dir, [list_pred(i).name(1:end-4) '.png']));
end

%Evaluate model and find ODS OIS AP values
results = cell(1,9);
gtdir = gt_dir;
resDir = nms_pred_dir;

assert(exist(img_dir,'dir')==7); assert(exist(gt_dir,'dir')==7);

[results{1:9}] = edgesEvalDir_x('resDir', resDir, 'gtDir', gtdir,...
    'pDistr', {{'type','parfor'}}, 'cleanup', 0, 'thrs',99, 'maxDist', 0.0075, 'thin', 1);
disp(results);



