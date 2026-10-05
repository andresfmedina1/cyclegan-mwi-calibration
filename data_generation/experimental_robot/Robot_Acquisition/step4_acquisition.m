clc; clear; close all;

%% ================================================================
%  STEP 4 — Adquisición automática robot XY + VNA 12×12
%  Robot de adquisición automática de targets — POLITO
%  ----------------------------------------------------------------
%  Mueve el robot a cada posición válida del target seleccionado,
%  mide la matriz S 12×12 con el VNA y los dos switches MCL,
%  y guarda cada posición en measurements_database.h5 (mismo
%  formato que acquisition_gui.m).
%
%  REQUISITOS PREVIOS:
%    ✔  step1_calibrate_robot.m  (robot_config.mat)
%    ✔  step2_test_movement.m    (motores verificados)
%    ✔  step3_homing.m           (stroke en (0,0) del phantom)
%    ✔  stroke_positions.m       (valid_positions.mat)
%    ✔  VNA y switches encendidos y calibrados
%% ================================================================

%% ── Parámetros del VNA ──────────────────────────────────────────
settingsVNA.frequencyRange = [0.8e9, 1.8e9];
settingsVNA.numPoints      = 11;
settingsVNA.dBmPower       = 0;
settingsVNA.BW             = 100;
settingsVNA.avg_N          = 12;
fstart = settingsVNA.frequencyRange(1);
fstop  = settingsVNA.frequencyRange(2);
settingsVNA.freq = linspace(fstart, fstop, settingsVNA.numPoints)';

%% ── Parámetros del robot ────────────────────────────────────────
ACK_TIMEOUT_S  = 30;
SWITCH_PAUSE_S = 0.05;   % pausa tras cada comando de switch [s]

%% ── Carpetas ────────────────────────────────────────────────────
root_folder    = fileparts(mfilename('fullpath'));
config_file    = fullfile(root_folder, 'config', 'robot_config.mat');
meas_folder    = fullfile(root_folder, '..');
pos_file       = fullfile(root_folder, '..', 'valid_positions.mat');
db_folder      = fullfile(root_folder, '..', 'Meas_Switch_12x12');
db_file        = fullfile(db_folder, 'measurements_database_robot_pueba.h5');

addpath(meas_folder);   % acceso a meas2x2_trigger_MGDR, setVNA_v3_2ports

%% ================================================================
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   Robot XY — Adquisición automática  (Step 4)   ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

%% ── 1. Cargar configuración del robot ───────────────────────────
if ~isfile(config_file)
    error('No se encontró robot_config.mat.\nEjecuta step1_calibrate_robot.m primero.\n');
end

load(config_file, 'arduino_port', 'cmd_per_mm_x', 'cmd_per_mm_y', ...
                  'DWELL_S', 'OFFSET_X_MM', 'OFFSET_Y_MM', ...
                  'calibration_date');

% Verificar que se hizo homing
cfg = load(config_file);
if ~isfield(cfg, 'homing_date')
    error(['No se encontró homing_date en robot_config.mat.\n' ...
           'Ejecuta step3_homing.m primero para definir el origen.\n']);
end

fprintf('Calibración  : %s\n', calibration_date);
fprintf('Homing       : %s\n', cfg.homing_date);
fprintf('OFFSET_X_MM  : %.1f mm\n', OFFSET_X_MM);
fprintf('OFFSET_Y_MM  : %.1f mm\n\n', OFFSET_Y_MM);

if OFFSET_Y_MM == 0
    fprintf('  ⚠️  OFFSET_Y_MM = 0 — asegúrate de haber actualizado este valor.\n\n');
end

%% ── 2. Cargar posiciones válidas ────────────────────────────────
if ~isfile(pos_file)
    error(['No se encontró valid_positions.mat.\n' ...
           'Ejecuta primero positions/stroke_positions.m\n']);
end

vp = load(pos_file);
vp = vp.valid_positions;

fprintf('Targets disponibles:\n');
for t = 1:numel(vp.target)
    fprintf('  [%d]  %s  —  %d posiciones válidas\n', ...
            t, vp.target(t).nombre, vp.target(t).N);
end
fprintf('\n');
tgt_idx = input('Selecciona target [1/2/3]: ');
if tgt_idx < 1 || tgt_idx > numel(vp.target)
    error('Selección inválida.');
end

tgt      = vp.target(tgt_idx);
ang_opts = vp.angulos_deg;         % [0, 90, 45, -45]
fprintf('\nOrientaciones disponibles:\n');
for a = 1:numel(ang_opts)
    fprintf('  [%d]  %d deg\n', a, ang_opts(a));
end
ang_idx = input('Selecciona orientación: ');
rot_deg = ang_opts(ang_idx);

positions = tgt.xy_mm;   % N×2 [x_mm, y_mm]

% ── Transformación de coordenadas phantom → robot ────────────────
% El phantom está montado rotado respecto al robot:
%   X_robot = -Y_phantom
%   Y_robot = -X_phantom
positions = [-positions(:,2), -positions(:,1)];

N_pos     = size(positions, 1);

fprintf('\nTarget   : %s\n', tgt.nombre);
fprintf('rx / ry  : %.0f / %.0f mm\n', tgt.rx_mm, tgt.ry_mm);
fprintf('Ángulo   : %d deg\n', rot_deg);
fprintf('Posiciones: %d\n\n', N_pos);

%% ── 3. Parámetros de medida ─────────────────────────────────────
fprintf('Tipo de medio dieléctrico:\n');
EPS_OPTS = {'Hem 1  (eps = 52.0)', 52.0; ...
            'Hem 2  (eps = 58.0)', 58.0; ...
            'Isc 1  (eps = 72.0)', 72.0; ...
            'Isc 2  (eps = 80.0)', 80.0};
for e = 1:size(EPS_OPTS,1)
    fprintf('  [%d]  %s\n', e, EPS_OPTS{e,1});
end
eps_idx = input('Selecciona medio: ');
epsilon = EPS_OPTS{eps_idx, 2};
eps_lbl = EPS_OPTS{eps_idx, 1};
notes   = input('Notas opcionales (Enter para ninguna): ', 's');

%% ── 4. Ordenar posiciones en boustrophedon ──────────────────────
positions = boustrophedon(positions);

%% ── 5. Estimación de tiempo ─────────────────────────────────────
meas_per_pos  = 36;                     % 6×6 combinaciones A×B
t_meas_est    = meas_per_pos * (3 * (settingsVNA.numPoints / settingsVNA.BW) ...
                                + 2 * SWITCH_PAUSE_S);
t_move_est    = 20 * max(cmd_per_mm_x, cmd_per_mm_y)^(-1) * 2;  % ~20 mm promedio
t_total_est   = N_pos * (DWELL_S + t_meas_est + t_move_est);

fprintf('\n══════════════════════════════════════════════════\n');
fprintf('  RESUMEN DE ADQUISICIÓN\n');
fprintf('══════════════════════════════════════════════════\n');
fprintf('  Target    : %s  —  %d deg\n', tgt.nombre, rot_deg);
fprintf('  Medio     : %s\n', eps_lbl);
fprintf('  Posiciones: %d\n', N_pos);
fprintf('  T estimado: ~%.0f min\n', t_total_est / 60);
fprintf('══════════════════════════════════════════════════\n\n');

conf = input('¿Comenzar adquisición? [S/n]: ', 's');
if strcmpi(strtrim(conf), 'n')
    fprintf('Adquisición cancelada.\n');
    return;
end

%% ── 6. Inicializar hardware ─────────────────────────────────────

% VNA
fprintf('\nConectando al VNA...\n');
instrObj = setVNA_v3_2ports(settingsVNA);
fprintf(instrObj, 'INIT:CONT OFF');
fprintf('✔️ VNA listo.\n');

% Switches MCL
fprintf('Cargando librería de switches...\n');
NET.addAssembly('C:\Image_to_Image\mcl_SolidStateSwitch_NET45.dll');
SW1 = mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
SW2 = mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
SW1.Connect('12410080206');
SW2.Connect('12410080235');
fprintf('✔️ Switches conectados.\n');

% Arduino
fprintf('Conectando al Arduino (%s)...\n', arduino_port);
ser = serialport(arduino_port, 9600);
configureTerminator(ser, "LF");
ser.Timeout = ACK_TIMEOUT_S;
pause(2);
fprintf('✔️ Arduino listo.\n\n');

% HDF5
if ~exist(db_folder, 'dir'), mkdir(db_folder); end
n_prev = count_acquisitions(db_file);
n_acq  = n_prev;

% ── Figura del atenuador (live, una traza por posición medida) ───
freq_GHz_atten = settingsVNA.freq / 1e9;
fig_atten = figure('Name', 'Atenuador — S21 vs frecuencia', 'NumberTitle', 'off', ...
                   'Position', [10 -20 800 420]);
ax_atten = axes(fig_atten);
hold(ax_atten, 'on'); grid(ax_atten, 'on'); box(ax_atten, 'on');
xlabel(ax_atten, 'Frecuencia (GHz)', 'FontSize', 10);
ylabel(ax_atten, '|S_{21}| (dB)',    'FontSize', 10);
title(ax_atten, 'Atenuador (SW1:7 \rightarrow SW2:7) — en espera de primera medida...', ...
      'FontSize', 10, 'FontWeight', 'bold');
h_atten_cur = plot(ax_atten, freq_GHz_atten, zeros(1, settingsVNA.numPoints), ...
                   'k-', 'LineWidth', 2, 'DisplayName', 'Última medida');
legend(ax_atten, 'Location', 'best', 'FontSize', 9);
drawnow;

%% ── 7. Bucle de adquisición ─────────────────────────────────────
robot_x = 0;   % posición actual del carrito en X [mm]
robot_y = 0;   % posición actual del carrito en Y [mm]

t_start = tic;

fprintf('══════════════════════════════════════════════════\n');
fprintf('  INICIO DEL ESCANEO  (%s)\n', datestr(now,'HH:MM:SS'));
fprintf('══════════════════════════════════════════════════\n\n');

for ip = 1:N_pos

    pos_x = positions(ip, 1);
    pos_y = positions(ip, 2);

    fprintf('[%d/%d]  Stroke → (X=%+.0f, Y=%+.0f) mm  ...', ...
            ip, N_pos, pos_x, pos_y);

    %% 7a. Mover robot a la posición
    dx = pos_x - robot_x;
    dy = pos_y - robot_y;

    if dx ~= 0
        cmd_x  = ternary(dx > 0, 'X+', 'X-');
        n_x    = cmd_for_mm(abs(dx), cmd_per_mm_x);
        move_axis(ser, cmd_x, n_x);
        robot_x = robot_x + sign(dx) * abs(dx);
    end
    if dy ~= 0
        cmd_y  = ternary(dy > 0, 'Y+', 'Y-');
        n_y    = cmd_for_mm(abs(dy), cmd_per_mm_y);
        move_axis(ser, cmd_y, n_y);
        robot_y = robot_y + sign(dy) * abs(dy);
    end

    %% 7b. Esperar estabilización
    pause(DWELL_S);

    %% 7c. Medir S 12×12
    N_pts = settingsVNA.numPoints;
    Sm    = zeros(12, 12, N_pts);

    for A = 1:6
        SW1.Send_SCPI(sprintf(':SP8T:STATE:%d', A), '');
        pause(SWITCH_PAUSE_S);
        for B = 1:6
            SW2.Send_SCPI(sprintf(':SP8T:STATE:%d', B), '');
            pause(SWITCH_PAUSE_S);

            [s_tot, ~] = meas2x2_trigger_MGDR(instrObj, settingsVNA);

            idx_A = A;
            idx_B = B + 6;
            Sm(idx_A, idx_A, :) = s_tot(1, 1, :);
            Sm(idx_B, idx_A, :) = s_tot(2, 1, :);
            Sm(idx_A, idx_B, :) = s_tot(1, 2, :);
            Sm(idx_B, idx_B, :) = s_tot(2, 2, :);
        end
    end

    %% 7c-bis. Medir atenuador (SW1:7 → SW2:7)
    SW1.Send_SCPI(':SP8T:STATE:7', '');
    pause(SWITCH_PAUSE_S);
    SW2.Send_SCPI(':SP8T:STATE:7', '');
    pause(SWITCH_PAUSE_S);
    [s_atten, ~]  = meas2x2_trigger_MGDR(instrObj, settingsVNA);
    S21_att_dB    = 20 * log10(abs(squeeze(s_atten(2,1,:))') + eps);  % 1×N_pts

    if isvalid(fig_atten)
        set(h_atten_cur, 'Color', [0.75 0.75 0.75], 'LineWidth', 0.8, ...
                         'HandleVisibility', 'off');
        h_atten_cur = plot(ax_atten, freq_GHz_atten, S21_att_dB, ...
                           'k-', 'LineWidth', 2, 'DisplayName', 'Última medida');
        f_ctr = round(N_pts / 2);
        title(ax_atten, ...
              sprintf('Atenuador  |  pos %d/%d  (X=%+.0f, Y=%+.0f) mm  —  S21=%.1f dB @ %.2f GHz', ...
                      ip, N_pos, pos_x, pos_y, S21_att_dB(f_ctr), freq_GHz_atten(f_ctr)), ...
              'FontSize', 9, 'FontWeight', 'bold');
        drawnow;
    end

    %% 7d. Guardar en HDF5
    n_acq = n_acq + 1;
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    save_to_hdf5(db_file, n_acq, Sm, s_atten, settingsVNA.freq, ...
                 timestamp, 'target', ...
                 pos_x, pos_y, rot_deg, ...
                 tgt.nombre, tgt.rx_mm, tgt.ry_mm, ...
                 epsilon, eps_lbl, notes, settingsVNA);

    t_elapsed = toc(t_start);
    t_remain  = t_elapsed / ip * (N_pos - ip);
    fprintf('  ✔️  acq #%04d  (%.0f min restantes)\n', n_acq, t_remain/60);
end

%% ── 8. Volver al origen ─────────────────────────────────────────
fprintf('\nVolviendo al origen...\n');
if robot_x ~= 0
    cmd_x = ternary(robot_x > 0, 'X-', 'X+');
    move_axis(ser, cmd_x, cmd_for_mm(abs(robot_x), cmd_per_mm_x));
end
if robot_y ~= 0
    cmd_y = ternary(robot_y > 0, 'Y-', 'Y+');
    move_axis(ser, cmd_y, cmd_for_mm(abs(robot_y), cmd_per_mm_y));
end
fprintf('✔️ Robot en origen.\n\n');

%% ── 9. Resumen y cierre ─────────────────────────────────────────
t_total = toc(t_start);
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   ADQUISICIÓN COMPLETADA  ✔️                      ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n');
fprintf('  Posiciones medidas : %d\n', N_pos);
fprintf('  Adquisiciones #    : %04d → %04d\n', n_prev+1, n_acq);
fprintf('  Tiempo total       : %.1f min\n', t_total/60);
fprintf('  Base de datos      : %s\n\n', db_file);

try
    fprintf(instrObj, 'OUTP:STAT OFF');
    fclose(instrObj);
    delete(instrObj);
catch
end
clear ser instrObj;
fprintf('Conexiones cerradas.\n');


%% ================================================================
%  FUNCIONES LOCALES
%% ================================================================

function pos = boustrophedon(pos)
%BOUSTROPHEDON  Ordena posiciones en patrón serpiente para minimizar recorrido.
    y_vals    = unique(pos(:,2));
    pos_sorted = [];
    for iy = 1:numel(y_vals)
        mask = pos(:,2) == y_vals(iy);
        row  = pos(mask, :);
        if mod(iy,2)==1
            row = sortrows(row,  1);   % X ascendente
        else
            row = sortrows(row, -1);   % X descendente
        end
        pos_sorted = [pos_sorted; row]; %#ok<AGROW>
    end
    pos = pos_sorted;
end

function n = cmd_for_mm(dist_mm, cmd_per_mm)
    n = max(1, round(dist_mm * cmd_per_mm));
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function move_axis(ser, cmd, n_steps)
    for i = 1:n_steps
        writeline(ser, cmd);
        ack = readline(ser);
        if ~strcmp(strtrim(ack), 'OK')
            warning('Respuesta inesperada en paso %d/%d: "%s"', i, n_steps, strtrim(ack));
        end
    end
end

function save_to_hdf5(db_file, acq_num, Sm, s_atten, freq, timestamp, meas_type, ...
                      pos_x, pos_y, rot_deg, tgt_lbl, rx_mm, ry_mm, ...
                      epsilon, eps_lbl, notes, settingsVNA)
    N   = numel(freq);
    grp = sprintf('/acq_%04d', acq_num);

    % Matriz S 12×12
    h5create(db_file, [grp '/Sm_real'], [12 12 N], 'Datatype', 'double');
    h5write( db_file, [grp '/Sm_real'], real(Sm));
    h5create(db_file, [grp '/Sm_imag'], [12 12 N], 'Datatype', 'double');
    h5write( db_file, [grp '/Sm_imag'], imag(Sm));
    h5create(db_file, [grp '/freq'],    [N 1],      'Datatype', 'double');
    h5write( db_file, [grp '/freq'],    freq(:));

    % Atenuador (SW1:7 → SW2:7) — matriz S 2×2
    h5create(db_file, [grp '/s_atten_real'], [2 2 N], 'Datatype', 'double');
    h5write( db_file, [grp '/s_atten_real'], real(s_atten));
    h5create(db_file, [grp '/s_atten_imag'], [2 2 N], 'Datatype', 'double');
    h5write( db_file, [grp '/s_atten_imag'], imag(s_atten));

    h5writeatt(db_file, grp, 'timestamp',         timestamp);
    h5writeatt(db_file, grp, 'meas_type',         meas_type);
    h5writeatt(db_file, grp, 'pos_x_mm',          pos_x);
    h5writeatt(db_file, grp, 'pos_y_mm',          pos_y);
    h5writeatt(db_file, grp, 'rotation_deg',      rot_deg);
    h5writeatt(db_file, grp, 'target_type',       tgt_lbl);
    h5writeatt(db_file, grp, 'target_rx_mm',      rx_mm);
    h5writeatt(db_file, grp, 'target_ry_mm',      ry_mm);
    h5writeatt(db_file, grp, 'target_epsilon_re', epsilon);
    h5writeatt(db_file, grp, 'target_eps_label',  eps_lbl);
    h5writeatt(db_file, grp, 'notes',             notes);
    h5writeatt(db_file, grp, 'acquired_by',       'step4_acquisition');
    h5writeatt(db_file, grp, 'vna_fstart_Hz',     settingsVNA.frequencyRange(1));
    h5writeatt(db_file, grp, 'vna_fstop_Hz',      settingsVNA.frequencyRange(2));
    h5writeatt(db_file, grp, 'vna_numPoints',     double(settingsVNA.numPoints));
    h5writeatt(db_file, grp, 'vna_BW_Hz',         double(settingsVNA.BW));
    h5writeatt(db_file, grp, 'vna_power_dBm',     double(settingsVNA.dBmPower));
end

function n = count_acquisitions(db_file)
    n = 0;
    if ~isfile(db_file), return; end
    try
        info = h5info(db_file, '/');
        n    = numel(info.Groups);
    catch
    end
end
