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

%% Configuration -----------------------------------------------------------
dicom_root = "SPIRAL_raw/example_scan";        % Folder containing *.dcm files.
save_root = "SPIRAL_processed/example_scan";   % Destination for MAT + JSON.
dict_path = "../dict.txt";                     % Custom DICOM dictionary (from manual).

if ~exist(save_root, "dir")
    mkdir(save_root);
end

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
    files(i).info = info; %#ok<AGROW>
end
[~, order] = sort(instance_numbers);
files = files(order);

%% Extract projections + geometry -----------------------------------------
num_proj = numel(files);
geometry = struct();
geometry.scanner = struct();
geometry.projections = cell(num_proj, 1);

for idx = 1:num_proj
    info = files(idx).info;
    dcm_path = fullfile(files(idx).folder, files(idx).name);
    raw = dicomread(dcm_path);
    slope = getfield_with_default(info, "RescaleSlope", 1.0);
    intercept = getfield_with_default(info, "RescaleIntercept", 0.0);
    img = double(raw) * double(slope) + double(intercept);

    proj_id = sprintf("%04d", idx);
    save(fullfile(save_root, proj_id + ".mat"), "img", "-v7.3");

    proj_meta = struct();
    proj_meta.file_stem = proj_id;
    proj_meta.original_file = files(idx).name;
    proj_meta.angle_rad = double(info.DetectorFocalCenterAngularPosition);
    proj_meta.table_z_mm = double(info.DetectorFocalCenterAxialPosition);
    proj_meta.source_z_mm = getfield_with_default(info, "SourceFocalSpotPositionZ", NaN);
    if isfield(info, "ContentTime")
        proj_meta.timestamp = info.ContentTime;
    end
    geometry.projections{idx} = proj_meta;

    if idx == 1
        scanner = struct();
        scanner.DSO_mm = double(info.DetectorFocalCenterRadialDistance);
        scanner.DSD_mm = double(info.ConstantRadialDistance);
        spacing = getfield_with_default(info, "DetectorElementTransverseSpacing", []);
        if isempty(spacing) && isfield(info, "PixelSpacing")
            spacing = info.PixelSpacing;
        end
        scanner.detector_pixel_size_mm = double(spacing(:));
        scanner.detector_pixels = size(img);
        scanner.mode = "fan";
        geometry.scanner = scanner;
    end

    if mod(idx, 50) == 0 || idx == num_proj
        fprintf("Saved %d/%d projections\n", idx, num_proj);
    end
end

geometry.projections = vertcat(geometry.projections{:});
geometry.notes = struct("generated_with", "dicom_spiral_process.m", ...
    "dictionary", dict_path);

json_path = fullfile(save_root, "scanner_geometry.json");
fid = fopen(json_path, "w");
cleaner = onCleanup(@() fclose(fid));
json_txt = jsonencode(geometry, PrettyPrint=true);
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

