% main_generate_simulation_dataset.m
%
% Generates a FEM simulation dataset in HDF5 format matching the metadata
% schema of measurements_database_robot.h5.
%
% Mesh       : mesh_p01.mat (original mesh, unmodified)
% Frequency  : 1.3 GHz (single frequency)
% Output     : results/simulation_dataset.h5
%
% Each group 'sim_XXXX' contains:
%   Esc_real  [12×12 float64]  — real( E_total - E_background ) at probes [V/m]
%   Esc_imag  [12×12 float64]  — imag( E_total - E_background ) at probes [V/m]
%   freq      [scalar float64] — simulation frequency [Hz]
%
% Attributes per group (same names as experimental HDF5):
%   meas_type          : 'healthy' | 'target'
%   pos_x_mm           : robot x position [mm]
%   pos_y_mm           : robot y position [mm]
%   rotation_deg       : target rotation angle [deg]
%   target_type        : 'Target 1 (30x20mm)' | ... | '(ninguno)'
%   target_rx_mm       : semi-axis rx [mm]  (along x_mesh at rot=0)
%   target_ry_mm       : semi-axis ry [mm]  (along y_mesh at rot=0)
%   target_epsilon_re  : real part of target permittivity
%   target_eps_label   : permittivity label string
%   generated_by       : 'main_generate_simulation_dataset'
%
% Coordinate transform (simulation — no phantom rotation):
%   x_mesh = pos_x_mm / 1000
%   y_mesh = pos_y_mm / 1000 + dy_offset
%
% Rotation convention:
%   At rot=0°  : rx along x_mesh, ry along y_mesh
%   Positive   : counterclockwise in mesh frame
%
% !! IMPORTANT — FILL IN sigma VALUES BEFORE RUNNING !!
%   See PERMITTIVITY TABLE section.
%
% Total FEM calls: 1 background + 1216 total-field = 1217.
% Estimated runtime: depends on mesh size and hardware.

clc; close all; clear;
addpath('../../Forward_solvers/lib');
addpath('lib');

% ── Visualización de muestra: poner true para ver el zone map
%    del primer escenario de cada tipo de target antes del loop completo
plot_sample = true;

%% ============================================================
%  SIMULATION PARAMETERS
%% ============================================================
sim_f     = 1.3e9;
sim_omega = 2 * pi * sim_f;
eps0      = 8.854187817e-12;
dy_offset = 0.00423;   % [m]
current   = 1e-3;      % [A]

phys2fem   = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7];
nA         = 6;   N = 12;
meas_mask  = false(N);
meas_mask(1:nA,   nA+1:N) = true;
meas_mask(nA+1:N, 1:nA  ) = true;

%% ============================================================
%  BACKGROUND MATERIAL PERMITTIVITIES
%% ============================================================
coupling_epsr  = 13;    coupling_sigma  = 0.2;
brain_epsr     = 47.37;    brain_sigma     = 0.98;
pla_epsr       = 3;     pla_sigma       = 1.7e-3;
ring_in_mm     = 1;
ring_out_mm    = 5;

bg_epsc    = coupling_epsr - 1i * coupling_sigma / (sim_omega * eps0);
brain_epsc = brain_epsr    - 1i * brain_sigma    / (sim_omega * eps0);
pla_epsc   = pla_epsr      - 1i * pla_sigma      / (sim_omega * eps0);

%% ============================================================
%  PERMITTIVITY TABLE  ← FILL IN sigma VALUES BEFORE RUNNING
%
%  eps_re values match experimental labels:
%    52 = Hem 1  (hemorrhagic stroke type 1)
%    58 = Hem 2  (hemorrhagic stroke type 2)
%    72 = Isc 1  (ischemic stroke type 1)
%    80 = Isc 2  (ischemic stroke type 2)
%
%  sigma [S/m] at 1.3 GHz — SPECIFY CORRECT VALUES
%% ============================================================
tgt_eps_re = [63.69,                  77.7,                  37.47,                  15.1];
tgt_sigma  = [1.32,                   0.36,                   1.0,                   0.80];  % <-- FILL IN
tgt_labels = {'Hem 1  (eps = 63.69)', 'Hem 2  (eps = 77.7)', ...
              'Isc 1  (eps = 37.47)', 'Isc 2  (eps = 15.1)'};

assert(~any(isnan(tgt_sigma)), ...
    ['Fill in sigma values for each permittivity label in the ' ...
     'PERMITTIVITY TABLE section before running.']);

tgt_epsc = zeros(1, 4);
for k = 1:4
    tgt_epsc(k) = tgt_eps_re(k) - 1i * tgt_sigma(k) / (sim_omega * eps0);
end

%% ============================================================
%  TARGET DEFINITIONS (3 physical phantoms)
%  rx, ry = semi-axes [mm]  (rx along x_mesh at rot=0°)
%% ============================================================
T(1).name   = 'Target 1 (30x20mm)';   T(1).rx_mm = 15;   T(1).ry_mm = 10;
T(2).name   = 'Target 2 (60x50mm)';   T(2).rx_mm = 30;   T(2).ry_mm = 25;
T(3).name   = 'Target 3 (110x80mm)';  T(3).rx_mm = 55;   T(3).ry_mm = 40;

%% ============================================================
%  POSITION GRIDS [mm, robot frame] — extracted from experimental HDF5
%  Transform to mesh: x_mesh = pos_x/1000, y_mesh = pos_y/1000 + dy_offset
%% ============================================================
pos_T1 = [
   -50, -40;  -50, -20;  -50,   0;  -50,  20;  -50,  40;
   -30, -60;  -30, -40;  -30, -20;  -30,   0;  -30,  20;  -30,  40;  -30,  60;
   -10, -80;  -10, -60;  -10, -40;  -10, -20;  -10,   0;  -10,  20;  -10,  40;  -10,  60;  -10,  80;
    10, -80;   10, -60;   10, -40;   10, -20;   10,   0;   10,  20;   10,  40;   10,  60;   10,  80;
    30, -60;   30, -40;   30, -20;   30,   0;   30,  20;   30,  40;   30,  60;
    50, -40;   50, -20;   50,   0;   50,  20;   50,  40;
];  % 42 positions

pos_T2 = [
   -50, -20;  -50,   0;
   -30, -40;  -30, -20;  -30,   0;  -30,  20;  -30,  40;
   -10, -60;  -10, -40;  -10, -20;  -10,   0;  -10,  20;  -10,  40;  -10,  60;
    10, -60;   10, -40;   10, -20;   10,   0;   10,  20;   10,  40;   10,  60;
    30, -40;   30, -20;   30,   0;   30,  20;   30,  40;
];  % 26 positions

pos_T3 = [
   -10, -40;  -10, -20;  -10,   0;  -10,  20;
    10, -40;   10, -20;   10,   0;   10,  20;
];  % 8 positions

pos_grids = {pos_T1, pos_T2, pos_T3};

%% ============================================================
%  ROTATIONS [degrees]
%% ============================================================
rotations_deg = [-45, 0, 45, 90];

%% ============================================================
%  LOAD MESH
%% ============================================================
fprintf('Loading mesh_p01.mat ...\n');
md            = load('Mesh/mesh_p01.mat');
verts         = md.verts;   verts(:,3) = [];   % N×2  [x,y] in metres
cells         = md.cells;                       % M×3  node indices (1-based)
idx_cells_DoI = md.idx_cells_DoI(:);

n_nodes = size(verts, 1);
n_cells = size(cells, 1);
fprintf('  Nodes: %d   Cells: %d   DOI cells: %d\n', n_nodes, n_cells, numel(idx_cells_DoI));

% Cell centroids (all cells)
cx_all = (verts(cells(:,1),1) + verts(cells(:,2),1) + verts(cells(:,3),1)) / 3;
cy_all = (verts(cells(:,1),2) + verts(cells(:,2),2) + verts(cells(:,3),2)) / 3;

%% ============================================================
%  SKULL CONTOUR AND RING
%% ============================================================
bnd           = load('MRI_labels/boundaries.mat');
skull_raw     = bnd.boundaries{337};
theta_sk      = -90 * pi/180;
Rot_sk        = [cos(theta_sk) -sin(theta_sk); sin(theta_sk) cos(theta_sk)];
skull_contour = (Rot_sk * skull_raw')';
skull_contour(:,2) = skull_contour(:,2) + dy_offset;

skull_center = mean(skull_contour, 1);
unit_dir     = skull_contour - skull_center;
unit_dir     = unit_dir ./ sqrt(sum(unit_dir.^2, 2));
skull_inner  = skull_contour - (ring_in_mm  / 1e3) * unit_dir;
skull_outer  = skull_contour + (ring_out_mm / 1e3) * unit_dir;

%% ============================================================
%  ANTENNA POSITIONS
%% ============================================================
ant_scad = [
    0.1232854000,  0.0000000000;
    0.0999189800,  0.0605022300;
    0.0479169300,  0.0901708300;
    0.0009053600,  0.1001224300;
   -0.0511411600,  0.0906341800;
   -0.0946068200,  0.0555665000;
   -0.1124909000,  0.0000000000;
   -0.0940806400, -0.0561425300;
   -0.0514699300, -0.0900297700;
    0.0011043200, -0.0998655400;
    0.0469120000, -0.0871032100;
    0.0982543800, -0.0607438200;
];
r_vec    = sqrt(sum(ant_scad.^2, 2));
ant_scad = ant_scad - 0.003 * (ant_scad ./ r_vec);

antennaPos       = zeros(12, 2);
antennaPos(:,1)  = -ant_scad(:,2);
antennaPos(:,2)  = -ant_scad(:,1) + dy_offset;

probeNodes = findProbeNodes(verts', antennaPos);

%% ============================================================
%  BACKGROUND PERMITTIVITY MAP (shared across all scenarios)
%% ============================================================
perm_bg          = bg_epsc * ones(n_cells, 1);
perm_bg(idx_cells_DoI) = brain_epsc;

cx_doi = cx_all(idx_cells_DoI);  cy_doi = cy_all(idx_cells_DoI);
in_skull_doi = inpolygon(cx_doi, cy_doi, skull_contour(:,1), skull_contour(:,2));
in_inner_doi = inpolygon(cx_doi, cy_doi, skull_inner(:,1),   skull_inner(:,2));
idx_inner    = idx_cells_DoI(in_skull_doi & ~in_inner_doi);

in_skull_all = inpolygon(cx_all, cy_all, skull_contour(:,1), skull_contour(:,2));
in_outer_all = inpolygon(cx_all, cy_all, skull_outer(:,1),   skull_outer(:,2));
idx_outer    = find(in_outer_all & ~in_skull_all);

perm_bg(idx_inner) = pla_epsc;
perm_bg(idx_outer) = pla_epsc;

%% ============================================================
%  SOLVE BACKGROUND FIELD (once, shared)
%% ============================================================
fprintf('\nSolving background field (1 FEM call) ...\n');
t_bg = tic;
[Ev_bg, ~, ~] = fem2d(verts, cells, perm_bg, sim_f, probeNodes, bg_epsc, [], current);
fprintf('  Done in %.1f s\n', toc(t_bg));

Ep_bg = Ev_bg(probeNodes, :);   % 12×12 FEM order

%% ============================================================
%  BUILD SCENARIO LIST
%  Each row: [target_idx, pos_x_mm, pos_y_mm, rot_deg, eps_idx]
%  target_idx = 0 → healthy (no target)
%% ============================================================
scenarios = [];

% Target scenarios only (healthy = Esc all-zeros, redundant)
for ti = 1:3
    pos_grid = pos_grids{ti};
    for pi = 1:size(pos_grid, 1)
        for ri = 1:numel(rotations_deg)
            for ei = 1:numel(tgt_eps_re)
                scenarios = [scenarios; ti, pos_grid(pi,1), pos_grid(pi,2), ...
                                         rotations_deg(ri), ei]; %#ok<AGROW>
            end
        end
    end
end

n_scenarios = size(scenarios, 1);
fprintf('\nTotal scenarios: %d\n', n_scenarios);

% Primer escenario de cada tipo de target para la visualización de muestra
plot_sample_sids = arrayfun(@(ti) find(scenarios(:,1)==ti, 1), 1:3);
phys_lbl     = {'A1','A2','A3','A4','A5','A6','B1','B2','B3','B4','B5','B6'};
idx_ring_vis = union(idx_inner, idx_outer);
cmap_zones   = [0.18 0.45 0.75; 0.95 0.90 0.20; 0.80 0.22 0.10; 0.85 0.10 0.85];

%% ============================================================
%  OUTPUT HDF5
%% ============================================================
out_file = 'results/simulation_dataset_V2.h5';
%% 
if exist(out_file, 'file'), delete(out_file); end

%% ============================================================
%  MAIN SIMULATION LOOP
%% ============================================================
t_total = tic;
fprintf('\n--- Starting simulation loop ---\n');

for s = 1:n_scenarios
    ti      = scenarios(s, 1);   % target index (0 = healthy)
    pos_x   = scenarios(s, 2);   % [mm] robot x
    pos_y   = scenarios(s, 3);   % [mm] robot y
    rot_deg = scenarios(s, 4);   % [deg]
    ei      = scenarios(s, 5);   % eps index (0 = healthy)

    group = sprintf('/sim_%04d', s);

    %% --- Build total permittivity map ---
    if ti == 0
        % Healthy: same as background
        Esc = zeros(N);
        meas_type    = 'healthy';
        target_name  = '(ninguno)';
        rx_mm = 0;   ry_mm = 0;
        eps_re_val   = 0;
        eps_lbl      = 'N/A';
    else
        meas_type   = 'target';
        target_name = T(ti).name;
        rx_mm       = T(ti).rx_mm;
        ry_mm       = T(ti).ry_mm;
        eps_re_val  = tgt_eps_re(ei);
        eps_lbl     = tgt_labels{ei};

        % Target center in mesh coordinates (simulation: no phantom rotation)
        cx0 = -pos_x / 1000;              % x_mesh = pos_x_mm / 1000
        cy0 = -pos_y / 1000 + dy_offset;  % y_mesh = pos_y_mm / 1000 + dy_offset

        rx  = rx_mm / 1000;   % [m]
        ry  = ry_mm / 1000;   % [m]
        th  = rot_deg * pi / 180;

        % Rotated ellipse inclusion test on all cell centroids
        dx     = cx_all - cx0;
        dy     = cy_all - cy0;
        dx_rot = dx * cos(th) + dy * sin(th);
        dy_rot = -dx * sin(th) + dy * cos(th);
        ell_d  = (dx_rot / rx).^2 + (dy_rot / ry).^2;

        idx_target = intersect(find(ell_d <= 1), idx_cells_DoI);

        % ── Zone map de muestra (primer escenario de cada target type) ──
        if plot_sample && ismember(s, plot_sample_sids)
            zone_map = ones(size(cells, 1), 1);
            zone_map(idx_cells_DoI) = 3;
            zone_map(idx_ring_vis)  = 2;
            zone_map(idx_target)    = 4;

            figure('Name', sprintf('Zone map — sim #%04d  Target %d', s, ti), ...
                   'Position', [250 150 680 640]);
            patch('Faces', cells, 'Vertices', verts, ...
                  'FaceVertexCData', zone_map, 'FaceColor', 'flat', 'EdgeColor', 'none');
            colormap(gca, cmap_zones);  clim([1 4]);
            colorbar('Ticks', [1.375, 2.125, 2.875, 3.625], ...
                'TickLabels', {'Matching  \epsilon_r=13', 'PLA ring  \epsilon_r=3', ...
                               sprintf('Brain  \\epsilon_r=%.1f', brain_epsr), 'Target (lesión)'});
            hold on;
            plot(skull_contour(:,1), skull_contour(:,2), 'w-',  'LineWidth', 1.6);
            plot(skull_inner(:,1),   skull_inner(:,2),   'w:',  'LineWidth', 1.2);
            plot(skull_outer(:,1),   skull_outer(:,2),   'w--', 'LineWidth', 1.2);

            probeCoords = verts(probeNodes, :);
            for k = 1:12
                fk = phys2fem(k);
                xk = probeCoords(fk,1);  yk = probeCoords(fk,2);
                mc = [0.00 0.80 0.20];
                if k > 6,  mc = [1.00 0.50 0.00];  end
                plot(xk, yk, '^', 'MarkerSize', 8, 'Color', mc, 'MarkerFaceColor', mc, ...
                     'HandleVisibility', 'off');
                text(xk+0.006, yk+0.006, phys_lbl{k}, ...
                     'FontSize', 7, 'FontWeight', 'bold', 'Color', mc);
            end
            plot(cx0, cy0, 'w+', 'MarkerSize', 14, 'LineWidth', 2.5, ...
                 'DisplayName', 'target center');

            axis equal;  xlim([-0.15 0.15]);  ylim([-0.15 0.15]);
            xlabel('x  [m]');  ylabel('y  [m]');
            title(sprintf('sim #%04d — Target %d | pos=[%.0f, %.0f] mm | rot=%.0f° | \\epsilon_r=%.2f  (%d cells)', ...
                s, ti, pos_x, pos_y, rot_deg, tgt_eps_re(ei), numel(idx_target)), 'FontSize', 11);
            legend('target center', 'Location', 'southeastoutside');
            hold off;
            drawnow;
        end

        perm_tot             = perm_bg;
        perm_tot(idx_target) = tgt_epsc(ei);

        % Solve total field
        [Ev_tot, ~, ~] = fem2d(verts, cells, perm_tot, sim_f, probeNodes, bg_epsc, [], current);

        % Scattered field at probes (physical order, masked)
        Ep_tot = Ev_tot(probeNodes, :);
        Esc    = (Ep_tot(phys2fem, phys2fem) - Ep_bg(phys2fem, phys2fem)) .* meas_mask;
    end

    %% --- Write HDF5 ---
    h5create(out_file, [group '/Esc_real'], [N N]);
    h5write( out_file, [group '/Esc_real'], real(Esc));
    h5create(out_file, [group '/Esc_imag'], [N N]);
    h5write( out_file, [group '/Esc_imag'], imag(Esc));
    h5create(out_file, [group '/freq'],     [1 1]);
    h5write( out_file, [group '/freq'],     sim_f);

    h5writeatt(out_file, group, 'meas_type',         meas_type);
    h5writeatt(out_file, group, 'pos_x_mm',          pos_x);
    h5writeatt(out_file, group, 'pos_y_mm',          pos_y);
    h5writeatt(out_file, group, 'rotation_deg',      rot_deg);
    h5writeatt(out_file, group, 'target_type',       target_name);
    h5writeatt(out_file, group, 'target_rx_mm',      rx_mm);
    h5writeatt(out_file, group, 'target_ry_mm',      ry_mm);
    h5writeatt(out_file, group, 'target_epsilon_re', eps_re_val);
    h5writeatt(out_file, group, 'target_eps_label',  eps_lbl);
    h5writeatt(out_file, group, 'generated_by',      'main_generate_simulation_dataset');

    %% --- Progress ---
    if mod(s, 50) == 0 || s == 1 || s == n_scenarios
        elapsed  = toc(t_total);
        rate     = s / elapsed;
        eta_s    = (n_scenarios - s) / rate;
        fprintf('  [%4d/%d] elapsed=%.0fs  ETA=%.0fs (%.1f min)\n', ...
            s, n_scenarios, elapsed, eta_s, eta_s/60);
    end
end

elapsed_total = toc(t_total);
fprintf('\nDone. %d scenarios in %.1f min → %s\n', ...
    n_scenarios, elapsed_total/60, out_file);
