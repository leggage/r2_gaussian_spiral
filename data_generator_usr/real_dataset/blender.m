clear

file = dir(fullfile(".","*.dcm"));
info = dicominfo(file(1).name,"dictionary","../dict.txt");
scanner = struct();
%%上每个像素在z轴方向的长度
width_row = info.DetectorElementAxialSpacing;
height_col = info.DetectorElementTransverseSpacing;
col = info.NumberofDetectorColumns;
row = info.NumberofDetectorRows;
dso = info.DetectorFocalCenterRadialDistance;
dsd = info.ConstantRadialDistance;

%%scanner旋转一圈，采集的投影数量
samples_cycle = info.NumberofSourceAngularSteps;

pitch = info.SpiralPitchFactor;

%%把所有投影按沿着第三维存储，初始化为0
proj_all = zeros(col,row,length(file));

for i=1:length(file)
    filename=fullfile(".",file(i).name);
    dsinf=dicominfo(filename,"dictionary","../dict.txt");
    z(i)=dsinf.DetectorFocalCenterAxialPosition;
    angle(i)=dsinf.DetectorFocalCenterAngularPosition;
    pixel=dicomread(filename);
    proj_all(:,:,i)=double(pixel)*dsinf.RescaleSlope+dsinf.RescaleIntercept;
end
%%根据z排序对投影重排,升序，
[zsorted,index] = sort(z);
angle = angle(index);
proj_all = proj_all(:,:,index);
%%螺旋投影最长的物理距离,根据此求出如果要完全投影整个volume需要多少rows
max_pd = max(zsorted) - min(zsorted);
num_bl_rows = round(max_pd/width_row);
%%每个角度都有一个拼接后的投影，尺寸为col*num_bl_rows
proj_blender = cell(samples_cycle, 1);

consider_pitch = 1;


%%%拼接实现，在矩阵水平方向拼接（也即投影的rows方向）%%%
for i = 1:length(file)
    

    % 修正 mod，保证从 1 到 samples_cycle
    stem = mod(i-1, samples_cycle) + 1;
    %%第一次旋转时对应的是最底下的投影,记录它的zmin和angle
    if i<=samples_cycle
        proj_blender{i}.zmin=zsorted(i);
        proj_blender{i}.angle_rad = angle(i);
        proj_blender{i}.pixel = proj_all(:,:,i);
    else
        %%每个角度最大的z偏移量逐次更新，取最后一次
        proj_blender{stem}.zmax=zsorted(i);
        if consider_pitch
            effect_rows = round(row * pitch);
            % proj_all: [nRow × nCol × nProj]
            % 这里是水平拼接，第二维在变长
            proj_blender{stem}.pixel = [proj_blender{stem}.pixel, proj_all(:, 1:effect_rows, i)];
        else
            proj_blender{stem}.pixel = [proj_blender{stem}.pixel, proj_all(:,:,i)];
        end
    end

end
    

%%%裁切，范围以中心角度（不一定是pi)的z轴范围为基准
cut=[proj_blender{samples_cycle/2}.zmin,proj_blender{samples_cycle/2}.zmax];
nor_size = size(proj_blender{samples_cycle/2}.pixel);
for j=1:samples_cycle
    shift = proj_blender{j}.zmin-cut(1);
    pixel_shift = round(abs(shift)/width_row);
    if pixel_shift==0
        pixel_shift=1;
    end
    if shift<=0
        proj_blender{j}.pixel = proj_blender{j}.pixel(:,pixel_shift:end);
    else
        proj_blender{j}.pixel = proj_blender{j}.pixel(:,1:end-pixel_shift);
    end
        proj_blender{j}.pixel = imresize(proj_blender{j}.pixel,nor_size,"bicubic");
        proj_id = sprintf("%04d", j);
        proj_dir = fullfile(".","proj");
        if ~exist(proj_dir, 'dir')
            mkdir(proj_dir);
        end
        proj = fullfile(proj_dir, proj_id+".mat");
        % Save with variable name 'img' for easier loading in Python
        img = proj_blender{j}.pixel;
        save(proj, "img");
end

% Build JSON structure
json_data = struct();
json_data.scanner = struct();
json_data.scanner.DSO_mm = dso;
json_data.scanner.DSD_mm = dsd;
json_data.scanner.detector_pixel_size_mm = [height_col, width_row];
json_data.scanner.detector_pixels = nor_size;
json_data.scanner.mode = "fan";  % or "cone" depending on your scanner

json_data.projections = struct([]);
for j = 1:samples_cycle
    proj_id = sprintf("%04d", j);
    mat_file = fullfile("proj", proj_id + ".mat");
    angle_deg = proj_blender{j}.angle_rad * 180 / pi;
    
    proj_entry = struct();
    proj_entry.file_stem = proj_id;
    proj_entry.mat_file = mat_file;
    proj_entry.angle_deg = angle_deg;
    proj_entry.angle_rad = proj_blender{j}.angle_rad;
    
    json_data.projections = [json_data.projections; proj_entry];
end

% Save JSON file
json_file = fullfile(".", "scanner_geometry.json");
try
    % Try using jsonencode (requires MATLAB R2016b+)
    json_str = jsonencode(json_data, 'PrettyPrint', true);
    fid = fopen(json_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('JSON file saved to: %s\n', json_file);
catch
    % Fallback: manual JSON writing (for older MATLAB versions)
    fprintf('Warning: jsonencode not available, using manual JSON writing\n');
    fid = fopen(json_file, 'w');
    fprintf(fid, '{\n');
    fprintf(fid, '  "scanner": {\n');
    fprintf(fid, '    "DSO_mm": %.10f,\n', dso);
    fprintf(fid, '    "DSD_mm": %.10f,\n', dsd);
    fprintf(fid, '    "detector_pixel_size_mm": [%.10f, %.10f],\n', height_col, width_row);
    fprintf(fid, '    "detector_pixels": [%d, %d],\n', col, row);
    fprintf(fid, '    "mode": "fan"\n');
    fprintf(fid, '  },\n');
    fprintf(fid, '  "projections": [\n');
    for j = 1:samples_cycle
        proj_id = sprintf("%04d", j);
        mat_file = fullfile("proj", proj_id + ".mat");
        angle_deg = proj_blender{j}.angle_rad * 180 / pi;
        fprintf(fid, '    {\n');
        fprintf(fid, '      "file_stem": "%s",\n', proj_id);
        fprintf(fid, '      "mat_file": "%s",\n', mat_file);
        fprintf(fid, '      "angle_deg": %.10f,\n', angle_deg);
        fprintf(fid, '      "angle_rad": %.10f', proj_blender{j}.angle_rad);
        if j < samples_cycle
            fprintf(fid, '\n    },\n');
        else
            fprintf(fid, '\n    }\n');
        end
    end
    fprintf(fid, '  ]\n');
    fprintf(fid, '}\n');
    fclose(fid);
    fprintf('JSON file saved to: %s\n', json_file);
end
