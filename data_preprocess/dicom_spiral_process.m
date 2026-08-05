function dicom_spiral_process()
%DICOM_SPIRAL_PROCESS Extract spiral CT projections + geometry from DICOM CT-PD.
%
%   This script follows the workflow described in
%   "DICOM-CT-PD-User-Manual_Version-3". It walks through a directory with
%   spiral (helical) CT projection DICOM files, extracts the mandatory
%   scanner parameters (angle, axial position, source-detector distances,
%   etc.), writes each projection to a MAT file, and stores the associated
%   metadata inside a JSON file. The Python pipeline can then reuse the
%   existing `generate_data.py` implementation with minimal changes.
%
%   To keep the original code base untouched, configure the folder paths
%   below and run this script inside Matlab before calling the Python
%   tooling.
%
%   Notes:
%     * The Siemens private dictionary shipped with the manual must be
%       reachable via `dicomDictionary` or supplied through the `dict_path`
%       variable.
%     * Projections are saved using the same numeric stem (0001, 0002, …)
%       so that downstream scripts operate exactly as they did for the FIPS
%       dataset.
%
%   r2_gaussian 一致化（与 dataset_readers.readCTameras 中 coord_left==true 等价）：
%     在导出 MAT/JSON 时完成原先仅在 Python 里对实拍数据做的变换，便于下游将
%     meta_data.json 里 scanner.coord_left 统一设为 false（与合成数据同路径）。
%     对应关系：角度 frame_angle = -frame_angle + 2*pi；强度 image *= 6（与
%     readCTameras 中注释一致，用于抵消 generate_data 中 /proj_rescale*object_scale
%     与训练时 scene_scale 的组合）；探测器列方向可选 fliplr（readCTameras 中
%     曾注释掉，但 CT-PD 单帧导出常与 angle2pose 列约定不一致，默认开启）。

%% Configuration -----------------------------------------------------------
dicom_root = "1.000000-Full dose projections-24362";        % Folder containing *.dcm files.
save_root = "SPIRAL_processed_t";   % Destination for MAT + JSON.
dict_path = "dict.txt";                     % Custom DICOM dictionary (from manual).
dicom_image_root = "1.000000-Full Dose Images-63186";

slice_files = dir(fullfile(dicom_image_root,"**","*.dcm"));
slice_files = slice_files(~[slice_files.isdir]);
z = zeros(1,numel(slice_files));
dsinf = dicominfo(fullfile(dicom_image_root,slice_files(1).name),"dictionary","dict.txt");
slice_thickness = getfield_with_default(dsinf, "SliceThickness", 0);
rows=getfield_with_default(dsinf, "Rows", 0);
cols =getfield_with_default(dsinf, "Columns", 0);
pixel_spacing = getfield_with_default(dsinf,"PixelSpacing",0);
svo1=rows*pixel_spacing(1);
svo2= cols*pixel_spacing(2);

for i = 1:numel(slice_files)
    dsinf = dicominfo(fullfile(dicom_image_root,slice_files(i).name),"dictionary","dict.txt");
    z(i) = getfield_with_default(dsinf, "SliceLocation", 0.0);
end

svo3 = max(z)-min(z)+slice_thickness;
    

save_proj = "SPIRAL_processed_t/proj";
if ~exist(save_root, "dir")
    mkdir(save_root);
end
if ~exist(save_proj, "dir")
    mkdir(save_proj);
end

% --- 与 r2_gaussian coord_left 等价烘焙（下游请将 coord_left 设为 false）---
bake_r2_coord_left = true;
bake_r2_img_tr = true;
% 与 readCTameras 一致：frame_angle = -frame_angle + 2*pi
bake_negate_source_angle = false;
% 与 readCTameras 一致：image = image * 6（若改 proj_rescale/object_scale 请相应调整）
bake_coord_left_intensity_scale = 6;
apply_intensity_bake = false;
% 实拍 CT-PD 单帧常与合成数据的列方向约定不一致；若与合成对齐后横向反了，改为 false
fliplr_detector_columns = false;

%% Enumerate DICOM projections --------------------------------------------
files = dir(fullfile(dicom_root, "**", "*.dcm"));
files = files(~[files.isdir]);
assert(~isempty(files), "No DICOM files were found under %s.", dicom_root);

% Sort by InstanceNumber (or filename as fallback) to keep projection order stable.
instance_numbers = zeros(numel(files), 1);
for i = 1:numel(files)
    if isempty(dict_path)
        info = dicominfo(fullfile(files(i).folder, files(i).name));
    else
        info = dicominfo(fullfile(files(i).folder, files(i).name), ...
            "dictionary", dict_path);
    end
    if isfield(info, "InstanceNumber")
        instance_numbers(i) = double(info.InstanceNumber);
    else
        instance_numbers(i) = i;
    end
    files(i).info = info;
end
[~, order] = sort(instance_numbers);
files = files(order);

%% Extract projections + geometry -----------------------------------------
num_proj = numel(files);
geometry = struct();
geometry.scanner = struct();
geometry.projections = cell(num_proj, 1);
geometry.svo = [svo1,svo2,svo3];
for idx = 1:num_proj
    info = files(idx).info;
    dcm_path = fullfile(files(idx).folder, files(idx).name);
    raw = dicomread(dcm_path);
    slope = getfield_with_default(info, "RescaleSlope", 1.0);
    intercept = getfield_with_default(info, "RescaleIntercept", 0.0);


    img = double(raw) * double(slope) + double(intercept);
    raw_angle = double(info.DetectorFocalCenterAngularPosition);
    if bake_r2_coord_left
        if bake_r2_img_tr
            img = transpose(img);
            
        end
        if fliplr_detector_columns
            img = fliplr(img);
        end
        if apply_intensity_bake
            img = img * double(bake_coord_left_intensity_scale);
        end
        if bake_negate_source_angle
            angle_use = -raw_angle + 2 * pi;
        else
            angle_use = raw_angle;
        end
    else
        angle_use = raw_angle;
    end


    proj_id = sprintf("%04d", idx);
    save(fullfile(save_proj,proj_id + ".mat"), "img", "-v7");

    proj_meta = struct();
    proj_meta.file_stem = proj_id;
    proj_meta.original_file = files(idx).name;
    proj_meta.angle_rad = angle_use;
    proj_meta.table_z_mm = double(info.DetectorFocalCenterAxialPosition);
    %proj_meta.source_z_mm = getfield_with_default(info, "SourceFocalSpotPositionZ", NaN);
%     if isfield(info, "ContentTime")
%         proj_meta.timestamp = info.ContentTime;
%     end
    geometry.projections{idx} = proj_meta;

    if idx == 1
        scanner = struct();
        scanner.DSO_mm = double(info.DetectorFocalCenterRadialDistance);
        scanner.DSD_mm = double(info.ConstantRadialDistance);
        spacing_tr = getfield_with_default(info, ...
            "DetectorElementTransverseSpacing", []);
        spacing_ax = getfield_with_default(info, ...
            "DetectorElementAxialSpacing", []);
        spacing = [];
        if ~isempty(spacing_tr) && ~isempty(spacing_ax)
            % 与读取后、transpose 前的 img 维度顺序一致：[dim1, dim2]
            spacing = [double(spacing_tr(1)); double(spacing_ax(1))];
        elseif ~isempty(spacing_tr)
            spacing = double(spacing_tr(:));
        elseif ~isempty(spacing_ax)
            spacing = double(spacing_ax(:));
        end
        if isempty(spacing) && isfield(info, "PixelSpacing")
            spacing = double(info.PixelSpacing(:));
        end
        if bake_r2_coord_left && bake_r2_img_tr && numel(spacing) >= 2
            spacing = spacing([2, 1]);
        end
        scanner.detector_pixel_size_mm = spacing(:);
        scanner.detector_pixels = size(img);
        scanner.mode = "cone";
        scanner.r2_gaussian_coord_left_baked = bake_r2_coord_left;
        if bake_r2_coord_left
            scanner.r2_gaussian_coord_left = false;
            if apply_intensity_bake
                intensity_scale_note = bake_coord_left_intensity_scale;
            else
                intensity_scale_note = 1;
            end
            scanner.r2_preprocess = struct();
            scanner.r2_preprocess.negate_source_angle_plus_2pi = ...
                bake_negate_source_angle;
            scanner.r2_preprocess.intensity_scale = intensity_scale_note;
            scanner.r2_preprocess.fliplr_detector_columns = ...
                fliplr_detector_columns;
        end
        geometry.scanner = scanner;
    end

    if mod(idx, 50) == 0 || idx == num_proj
        fprintf("Saved %d/%d projections\n", idx, num_proj);
    end
end

geometry.projections = vertcat(geometry.projections{:});
geometry.notes = struct();
geometry.notes.generated_with = 'dicom_spiral_process.m';
% char() 同时兼容 string 与 char，避免旧版对 struct("a",b,...) 解析报错
geometry.notes.dictionary = char(dict_path);
geometry.notes.r2_gaussian_coord_left_baked = bake_r2_coord_left;

json_path = fullfile(save_root, "scanner_geometry.json");
fid = fopen(json_path, "w");
cleaner = onCleanup(@() fclose(fid));
% 旧版 MATLAB 不支持 jsonencode(..., PrettyPrint = true)，须用逗号对形式；
% R2016b–R2019 等版本可能无 PrettyPrint 选项，回退为紧凑 JSON。
try
    json_txt = jsonencode(geometry, 'PrettyPrint', true);
catch
    json_txt = jsonencode(geometry);
end
fprintf(fid, "%s", json_txt);
fprintf("Scanner geometry saved to %s\n", json_path);
end

function value = getfield_with_default(s, field_name, default_value)
if isfield(s, field_name)
    value = s.(field_name);
else
    value = default_value;
end
end

