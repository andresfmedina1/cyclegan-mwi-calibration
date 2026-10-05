function step4_acquisition_gui()
%STEP4_ACQUISITION_GUI  GUI de adquisición automática robot XY + VNA 12×12
%  Ejecutar directamente:  step4_acquisition_gui()

%% ── Rutas ───────────────────────────────────────────────────────────
root_folder = fileparts(mfilename('fullpath'));
config_file = fullfile(root_folder, 'config', 'robot_config.mat');
pos_file    = fullfile(root_folder, '..', 'valid_positions.mat');
db_folder   = fullfile(root_folder, '..', 'Meas_Switch_12x12');
db_file     = fullfile(db_folder,   'measurements_database_robot_V2.h5');
addpath(fullfile(root_folder, '..'));

%% ── robot_config.mat ────────────────────────────────────────────────
if ~isfile(config_file)
    error('No se encontró robot_config.mat.\nEjecuta step1_calibrate_robot.m primero.\n');
end
cfg          = load(config_file);
arduino_port = char(cfg.arduino_port);   % char forzado para serialport()
cmd_per_mm_x = cfg.cmd_per_mm_x;
cmd_per_mm_y = cfg.cmd_per_mm_y;
DWELL_S_def  = cfg.DWELL_S;

%% ── Posiciones válidas ──────────────────────────────────────────────
has_positions  = false;
all_valid_pos  = {};
phantom_x_real = [];
phantom_y_real = [];
angulos_valid  = [0, 90, 45, -45];
TGT            = {'Custom', 20, 20};

if isfile(pos_file)
    try
        vp = load(pos_file);
        vp = vp.valid_positions;
        phantom_x_real = vp.phantom_x;
        phantom_y_real = vp.phantom_y;
        angulos_valid  = vp.angulos_deg;
        n_tgt_vp       = numel(vp.target);
        all_valid_pos  = cell(n_tgt_vp, 1);
        TGT            = cell(n_tgt_vp + 1, 3);
        for t = 1:n_tgt_vp
            TGT{t,1} = vp.target(t).nombre;
            TGT{t,2} = vp.target(t).rx_mm;
            TGT{t,3} = vp.target(t).ry_mm;
            all_valid_pos{t} = vp.target(t).xy_mm;
        end
        TGT{end,1} = 'Custom'; TGT{end,2} = 20; TGT{end,3} = 20;
        has_positions = true;
    catch ME
        fprintf('Aviso: valid_positions.mat no cargado (%s)\n', ME.message);
    end
end

%% ── Constantes ──────────────────────────────────────────────────────
EPS_OPTS = {'Hem 1  (eps = 52.0)', 52.0; ...
            'Hem 2  (eps = 58.0)', 58.0; ...
            'Isc 1  (eps = 72.0)', 72.0; ...
            'Isc 2  (eps = 80.0)', 80.0};
ang_labels   = {'0 deg', '90 deg', '45 deg', '-45 deg'};
ant_tick_lbl = [arrayfun(@(x) sprintf('A%d',x),1:6,'UniformOutput',false), ...
                arrayfun(@(x) sprintf('B%d',x),1:6,'UniformOutput',false)];
SWITCH_PAUSE_S = 0.05;
ACK_TIMEOUT_S  = 30;

%% ── HDF5 ────────────────────────────────────────────────────────────
if ~exist(db_folder,'dir'), mkdir(db_folder); end
n_prev = count_acquisitions(db_file);

%% ── Tema oscuro ─────────────────────────────────────────────────────
BG      = [0.13 0.13 0.16];
FG      = [0.93 0.93 0.93];
IBG     = [0.20 0.20 0.25];
COL_OK  = [0.25 0.95 0.42];
COL_ERR = [1.00 0.30 0.30];
COL_BUS = [1.00 0.82 0.22];

%% ── Tamaño adaptativo a la pantalla ─────────────────────────────────
scr   = get(0, 'ScreenSize');          % [1 1 W H] en píxeles lógicos
FIG_W = min(1060, scr(3) - 50);
FIG_H = min(900,  max(620, scr(4) - 90));
fig_x = max(10, floor((scr(3) - FIG_W) / 2));
fig_y = max(40, scr(4) - FIG_H - 60);

%% ── Geometría del panel ──────────────────────────────────────────────
AX_W  = max(380, round(FIG_W * 0.445));  % ancho del phantom
AX_L  = 20;
AX_B  = 40;
AX_H  = FIG_H - 62;
PX    = AX_L + AX_W + 18;   % borde izquierdo del panel derecho
PW    = FIG_W - PX - 16;    % ancho del panel derecho

%% ================================================================
%  FIGURA PRINCIPAL
%% ================================================================
fig = figure('Name',            'Robot XY — Adquisición Automática', ...
             'NumberTitle',     'off', ...
             'Color',           BG, ...
             'Position',        [fig_x fig_y FIG_W FIG_H], ...
             'Resize',          'on', ...
             'CloseRequestFcn', @on_close);

%% ── Axes: phantom ────────────────────────────────────────────────────
ax = axes('Parent',fig,'Units','pixels','Position',[AX_L AX_B AX_W AX_H], ...
          'Color',[0.09 0.09 0.12],'XColor',[0.55 0.55 0.55], ...
          'YColor',[0.55 0.55 0.55],'GridColor',[0.22 0.22 0.25],'FontSize',8);
hold(ax,'on'); axis(ax,'equal'); grid(ax,'on');

if has_positions && ~isempty(phantom_x_real)
    fill(ax,phantom_x_real,phantom_y_real,[0.15 0.32 0.52], ...
         'FaceAlpha',0.28,'EdgeColor',[0.32 0.62 1.00],'LineWidth',1.6);
    xlim(ax,[min(phantom_x_real)-25, max(phantom_x_real)+25]);
    ylim(ax,[min(phantom_y_real)-25, max(phantom_y_real)+25]);
else
    th=linspace(0,2*pi,400);
    fill(ax,90*cos(th),105*sin(th),[0.15 0.32 0.52], ...
         'FaceAlpha',0.28,'EdgeColor',[0.32 0.62 1.00],'LineWidth',1.6);
    xlim(ax,[-155 155]); ylim(ax,[-165 165]);
end
xlabel(ax,'X  (mm)','Color',[0.72 0.72 0.72],'FontSize',8);
ylabel(ax,'Y  (mm)','Color',[0.72 0.72 0.72],'FontSize',8);
title(ax,'Phantom — posiciones válidas', ...
      'Color',[0.92 0.92 0.92],'FontSize',9,'FontWeight','bold');

h_valid = scatter(ax,NaN,NaN,30,[0.30 0.75 0.45],'filled', ...
                  'MarkerEdgeColor',[0.15 0.50 0.25],'LineWidth',0.7);
h_done  = scatter(ax,NaN,NaN,50,[0.20 0.60 1.00],'filled', ...
                  'MarkerEdgeColor',[0.10 0.40 0.80],'LineWidth',1.2);
h_cur   = scatter(ax,NaN,NaN,120,[1.00 0.85 0.10],'filled', ...
                  'MarkerEdgeColor',[1.00 0.45 0.00],'LineWidth',2.0);
h_target_ell = plot(ax,NaN,NaN,'-','Color',[1.00 0.85 0.10],'LineWidth',1.8);

%% ================================================================
%  PANEL DERECHO — layout compacto
%  DY = paso estándar entre filas (29 px)
%% ================================================================
DY = 29;
y  = FIG_H - 42;    % coordenada Y del primer control (desde arriba)

% ── Título ───────────────────────────────────────────────────────────
uicontrol('Style','text','String','ROBOT XY — CONFIGURACIÓN', ...
    'Position',[PX y PW 22],'BackgroundColor',BG, ...
    'ForegroundColor',[0.38 0.78 1.00], ...
    'FontSize',11,'FontWeight','bold','HorizontalAlignment','left');
y = y - 26;

% ═══ Parámetros VNA (2 filas compactas) ══════════════════════════════
sec_lbl('─── VNA ───────────────────────────────────────',y);
y = y - 20;

% Fila A: inicio / fin / Pts / BW
lbl_s('inicio GHz:',y, 0,  72);
ed_fstart = edt(0.8, y,  74, 62);
lbl_s('fin:',       y,139,  28);
ed_fstop  = edt(1.8, y, 169, 62);
lbl_s('Pts:',       y,234,  28);
ed_npts   = edt(11,  y, 264, 48);
lbl_s('BW (Hz):',  y,315,  52);
ed_bw     = edt(100, y, 369, round(PW-369));
y = y - DY;

% Fila B: Potencia / Avg / Dwell
lbl_s('Pot(dBm):',y,  0, 62);
ed_pwr   = edt(0,           y,  64, 48);
lbl_s('Avg:',     y,115,    32);
ed_avg   = edt(12,           y, 149, 40);
lbl_s('Dwell(s):', y, 193,  58);
ed_dwell = edt(DWELL_S_def, y, 253, 48);
y = y - (DY + 2);

% ═══ Parámetros de medida ════════════════════════════════════════════
sec_lbl('─── Medida ─────────────────────────────────────',y);
y = y - 20;

% ── Modo: Target vs Healthy (background) ─────────────────────────────
lbl_s('Modo:',y,0,40);
mw = floor((PW-42)/2) - 1;
btn_mode_target = uicontrol('Style','togglebutton','String','◉  Target', ...
    'Position',[PX+42 y mw 24],'Value',1, ...
    'BackgroundColor',[0.18 0.50 0.32],'ForegroundColor',FG,'FontSize',8.5, ...
    'Callback',@(~,~)on_meas_toggle('target'));
btn_mode_healthy = uicontrol('Style','togglebutton','String','◎  Healthy', ...
    'Position',[PX+42+mw+1 y mw 24],'Value',0, ...
    'BackgroundColor',IBG,'ForegroundColor',[0.55 0.95 0.60],'FontSize',8.5, ...
    'Callback',@(~,~)on_meas_toggle('healthy'));
y = y - DY;

y_stroke_top = y;   % inicio del área de controles de target

% ── Target (visible sólo en modo Target) ─────────────────────────────
lbl_target_ctrl = uicontrol('Style','text','String','Target:', ...
    'Position',[PX y 52 19],'BackgroundColor',BG, ...
    'ForegroundColor',[0.68 0.68 0.68],'FontSize',8,'HorizontalAlignment','left');
dd_target = uicontrol('Style','popupmenu','String',TGT(:,1), ...
    'Position',[PX+54 y PW-56 24],'BackgroundColor',IBG,'ForegroundColor',FG, ...
    'FontSize',8.5,'Callback',@on_target_change);
y = y - DY;

% ── Orientación (visible sólo en modo Target) ────────────────────────
lbl_orient_ctrl = uicontrol('Style','text','String','Orientación:', ...
    'Position',[PX y 74 19],'BackgroundColor',BG, ...
    'ForegroundColor',[0.68 0.68 0.68],'FontSize',8,'HorizontalAlignment','left');
btn_ang = gobjects(4,1);
bw4 = floor((PW-76)/4) - 2;
for bi = 1:4
    btn_ang(bi) = uicontrol('Style','togglebutton','String',ang_labels{bi}, ...
        'Position',[PX+76+(bi-1)*(bw4+2), y-1, bw4, 25], ...
        'BackgroundColor',IBG,'ForegroundColor',FG,'FontSize',8,'Value',bi==1, ...
        'Callback',@(~,~) on_angle_btn(bi));
end
y = y - DY;

% ── Líquido / epsilon (visible sólo en modo Target) ──────────────────
lbl_eps_ctrl = uicontrol('Style','text','String','Líquido:', ...
    'Position',[PX y 50 19],'BackgroundColor',BG, ...
    'ForegroundColor',[0.68 0.68 0.68],'FontSize',8,'HorizontalAlignment','left');
dd_eps = uicontrol('Style','popupmenu','String',EPS_OPTS(:,1), ...
    'Position',[PX+52 y PW-54 24],'BackgroundColor',IBG,'ForegroundColor',FG,'FontSize',8.5);
y = y - DY;

y_stroke_bot = y;   % fin del área de controles de target

% Controles que se ocultan en modo Healthy
ctrl_stroke = [lbl_target_ctrl; dd_target; lbl_orient_ctrl; btn_ang; lbl_eps_ctrl; dd_eps];

% Label informativo que aparece en modo Healthy (cubre el mismo espacio)
lbl_healthy_info = uicontrol('Style','text', ...
    'String', sprintf('Medida de fondo  (sin target)\nEl robot permanece en la posición actual'), ...
    'Position', [PX y_stroke_bot PW y_stroke_top - y_stroke_bot], ...
    'BackgroundColor', BG, 'ForegroundColor', [0.45 0.95 0.55], ...
    'FontSize', 9.5, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'Visible', 'off');

% ── Notas (siempre visible) ───────────────────────────────────────────
lbl_s('Notas:',y,0,44);
ed_notes = uicontrol('Style','edit','String','', ...
    'Position',[PX+46 y-1 PW-48 24],'BackgroundColor',IBG,'ForegroundColor',FG, ...
    'FontSize',8.5,'HorizontalAlignment','left');
y = y - (DY + 2);

% ── Separador ────────────────────────────────────────────────────────
uicontrol('Style','text','String',repmat(char(8211),1,55), ...
    'Position',[PX y PW 12],'BackgroundColor',BG, ...
    'ForegroundColor',[0.26 0.26 0.30],'FontSize',8);

% ═══ Homing / Jog ════════════════════════════════════════════════════
y = y - 2;
sec_lbl('─── Homing / Jog del robot ──────────────────────',y);
y = y - 20;

% [CONECTAR ARDUINO]
btn_jog_connect = uicontrol('Style','pushbutton', ...
    'String','⚡  CONECTAR ARDUINO', ...
    'Position',[PX y PW 26],'BackgroundColor',[0.28 0.16 0.03], ...
    'ForegroundColor',[1.00 0.88 0.55],'FontSize',9.5,'FontWeight','bold', ...
    'Callback',@on_jog_connect);
y = y - 30;

% Paso (mm) + botones rápidos
lbl_s('Paso (mm):',y,0,62);
ed_step = uicontrol('Style','edit','String','1', ...
    'Position',[PX+64 y-1 52 23],'BackgroundColor',IBG,'ForegroundColor',FG, ...
    'FontSize',9.5,'HorizontalAlignment','center');
sv = [1,5,10]; sw = floor((PW-122)/3) - 2;
step_btns = gobjects(3,1);
for si = 1:3
    step_btns(si) = uicontrol('Style','pushbutton','String',sprintf('×%d',sv(si)), ...
        'Position',[PX+122+(si-1)*(sw+2), y-1, sw, 23], ...
        'BackgroundColor',IBG,'ForegroundColor',[0.72 0.88 1.00],'FontSize',8.5, ...
        'Callback',@(~,~) on_step_quick(sv(si)));
end
y = y - 28;

% Cruz de jog
CW = floor(PW/3);   BH = 24;   JS = 26;   % column width, button height, jog step

btn_y_plus = jog_btn(CW, y, CW, BH, '▲  Y+', @(~,~)on_jog('Y+'));
y = y - JS;

btn_x_minus = jog_btn( 0, y, CW, BH, '◄  X-', @(~,~)on_jog('X-'));
lbl_jog_pos = uicontrol('Style','text','String',sprintf('X=%+.1f\nY=%+.1f',0,0), ...
    'Position',[PX+CW, y+3, CW, BH-4],'BackgroundColor',IBG, ...
    'ForegroundColor',[0.75 0.75 0.75],'FontSize',7.5,'HorizontalAlignment','center');
btn_x_plus  = jog_btn(2*CW, y, CW, BH, 'X+  ►', @(~,~)on_jog('X+'));
y = y - JS;

btn_y_minus = jog_btn(CW, y, CW, BH, '▼  Y-', @(~,~)on_jog('Y-'));
y = y - 30;

% [ESTABLECER ORIGEN]
btn_set_origin = uicontrol('Style','pushbutton', ...
    'String','✓  ESTABLECER ORIGEN  (X=0, Y=0)', ...
    'Position',[PX y PW 26],'BackgroundColor',[0.07 0.34 0.16], ...
    'ForegroundColor',COL_OK,'FontSize',10,'FontWeight','bold', ...
    'Callback',@on_set_origin);
y = y - 6;

% Agrupar controles de jog
jog_btns = [btn_jog_connect; btn_y_plus; btn_x_minus; btn_x_plus; btn_y_minus; btn_set_origin];

% ── Separador ────────────────────────────────────────────────────────
uicontrol('Style','text','String',repmat(char(8211),1,55), ...
    'Position',[PX y PW 12],'BackgroundColor',BG, ...
    'ForegroundColor',[0.26 0.26 0.30],'FontSize',8);

% ═══ START ═══════════════════════════════════════════════════════════
y = y - 48;
btn_start = uicontrol('Style','pushbutton', ...
    'String','▶  INICIAR ADQUISICIÓN', ...
    'Position',[PX y PW 44],'BackgroundColor',[0.10 0.50 0.20],'ForegroundColor',[1 1 1], ...
    'FontSize',12.5,'FontWeight','bold','Callback',@on_start);
y = y - 26;

% Estado + progreso (una sola línea)
lbl_status = uicontrol('Style','text','String','Estado:  Listo', ...
    'Position',[PX y PW 20],'BackgroundColor',BG,'ForegroundColor',COL_OK, ...
    'FontSize',8.5,'FontWeight','bold','HorizontalAlignment','left');
y = y - 18;

% Barra de progreso
ax_bar = axes('Parent',fig,'Units','pixels','Position',[PX y PW 12], ...
              'XLim',[0 1],'YLim',[0 1],'XTick',[],'YTick',[], ...
              'Color',[0.17 0.17 0.21],'Box','on', ...
              'XColor',[0.26 0.26 0.30],'YColor',[0.26 0.26 0.30]);
h_bar = fill(ax_bar,[0 0 0 0],[0 0 1 1],[0.18 0.72 0.38],'EdgeColor','none');
y = y - 22;

% Historial
uicontrol('Style','text','String',sprintf('HISTORIAL  (%d adq. en BD)',n_prev), ...
    'Position',[PX y PW 16],'BackgroundColor',BG, ...
    'ForegroundColor',[0.42 0.68 1.00],'FontSize',8,'FontWeight','bold', ...
    'HorizontalAlignment','left');
y_hist = 15;
lb_h   = max(30, y - y_hist - 18);   % altura mínima garantizada
lb_hist = uicontrol('Style','listbox','String',{}, ...
    'Position',[PX y_hist PW lb_h],'BackgroundColor',[0.14 0.14 0.18], ...
    'ForegroundColor',[0.85 0.85 0.85],'FontSize',7.5,'FontName','Courier');

%% ── Estado compartido ────────────────────────────────────────────────
acquiring = false;
ang_idx   = 1;
meas_type = 'target';   % 'target' | 'healthy'
n_acq     = n_prev;
instrObj  = [];
SW1_h     = [];  SW2_h = [];
ser_h     = [];
jog_x     = 0;   jog_y = 0;
ard_ready = false;

set(btn_ang(1),'BackgroundColor',[0.18 0.50 0.32]);
update_valid_dots();
load_db_history();
uiwait(fig);

%% ================================================================
%  NESTED CALLBACKS
%% ================================================================

    function on_target_change(~,~)
        update_valid_dots();
    end

    function on_angle_btn(idx)
        ang_idx = idx;
        for k=1:4, set(btn_ang(k),'Value',k==idx,'BackgroundColor',IBG); end
        set(btn_ang(idx),'BackgroundColor',[0.18 0.50 0.32]);
    end

    %-- Modo Target / Healthy (background) --------------------------------
    function on_meas_toggle(mode)
        meas_type = mode;
        if strcmp(mode,'target')
            set(btn_mode_target, 'Value',1,'BackgroundColor',[0.18 0.50 0.32]);
            set(btn_mode_healthy,'Value',0,'BackgroundColor',IBG);
            set(ctrl_stroke,'Visible','on');
            set(lbl_healthy_info,'Visible','off');
        else   % healthy
            set(btn_mode_target, 'Value',0,'BackgroundColor',IBG);
            set(btn_mode_healthy,'Value',1,'BackgroundColor',[0.08 0.36 0.22]);
            set(ctrl_stroke,'Visible','off');
            set(lbl_healthy_info,'Visible','on');
        end
        update_valid_dots();
    end

    function update_valid_dots()
        % En modo Healthy: ocultar puntos y elipse de target
        if strcmp(meas_type,'healthy')
            set(h_valid,'XData',NaN,'YData',NaN);
            set(h_target_ell,'XData',NaN,'YData',NaN);
            return;
        end
        ti = get(dd_target,'Value');
        if has_positions && ti<=numel(all_valid_pos) && ~isempty(all_valid_pos{ti})
            p = all_valid_pos{ti};
            set(h_valid,'XData',p(:,1),'YData',p(:,2));
            rx=TGT{ti,2}; ry=TGT{ti,3}; rot=angulos_valid(ang_idx);
            th=linspace(0,2*pi,120); a=deg2rad(rot);
            xe=rx*cos(th)*cos(a)-ry*sin(th)*sin(a);
            ye=rx*cos(th)*sin(a)+ry*sin(th)*cos(a);
            set(h_target_ell,'XData',[xe xe(1)],'YData',[ye ye(1)]);
        else
            set(h_valid,'XData',NaN,'YData',NaN);
            set(h_target_ell,'XData',NaN,'YData',NaN);
        end
        drawnow;
    end

    function on_step_quick(val)
        set(ed_step,'String',num2str(val));
    end

    %-- Conectar / desconectar Arduino --------------------------------
    function on_jog_connect(~,~)
        if acquiring, return; end
        if ard_ready
            try; delete(ser_h); catch; end
            ser_h=[]; ard_ready=false;
            set(btn_jog_connect,'String','⚡  CONECTAR ARDUINO', ...
                'BackgroundColor',[0.28 0.16 0.03]);
            set(lbl_status,'String','Arduino desconectado.','ForegroundColor',[0.62 0.62 0.62]);
            return;
        end
        set(btn_jog_connect,'Enable','off');
        set(lbl_status,'String','Conectando Arduino...','ForegroundColor',COL_BUS);
        drawnow;
        try
            port_str = char(arduino_port);    % garantizar char para serialport()
            ser_h    = serialport(port_str, 9600);
            configureTerminator(ser_h,'LF');
            ser_h.Timeout = ACK_TIMEOUT_S;
            pause(2);
            ard_ready = true;
            set(btn_jog_connect,'String','✕  DESCONECTAR ARDUINO', ...
                'BackgroundColor',[0.42 0.06 0.06]);
            set(lbl_status,'String', ...
                sprintf('✔  Arduino OK  (%s)', port_str),'ForegroundColor',COL_OK);
        catch ME
            ard_ready = false;
            set(lbl_status,'String',sprintf('ERROR Arduino: %s',ME.message),'ForegroundColor',COL_ERR);
        end
        set(btn_jog_connect,'Enable','on');
    end

    %-- Joystick X/Y --------------------------------------------------
    function on_jog(dir)
        if acquiring || ~ard_ready
            if ~ard_ready
                set(lbl_status,'String','Conecta el Arduino primero.','ForegroundColor',COL_ERR);
            end
            return;
        end
        step_mm = str2double(get(ed_step,'String'));
        if isnan(step_mm)||step_mm<=0
            set(lbl_status,'String','Paso (mm) inválido.','ForegroundColor',COL_ERR); return;
        end
        set(jog_btns,'Enable','off'); drawnow;
        try
            switch dir
                case 'X+', move_axis(ser_h,'X+',cmd_for_mm(step_mm,cmd_per_mm_x)); jog_x=jog_x+step_mm;
                case 'X-', move_axis(ser_h,'X-',cmd_for_mm(step_mm,cmd_per_mm_x)); jog_x=jog_x-step_mm;
                case 'Y+', move_axis(ser_h,'Y+',cmd_for_mm(step_mm,cmd_per_mm_y)); jog_y=jog_y+step_mm;
                case 'Y-', move_axis(ser_h,'Y-',cmd_for_mm(step_mm,cmd_per_mm_y)); jog_y=jog_y-step_mm;
            end
            set(lbl_jog_pos,'String',sprintf('X=%+.1f\nY=%+.1f',jog_x,jog_y));
            set(lbl_status,'String', ...
                sprintf('Jog  X=%+.1f mm  |  Y=%+.1f mm',jog_x,jog_y),'ForegroundColor',COL_BUS);
        catch ME
            set(lbl_status,'String',sprintf('ERROR jog: %s',ME.message),'ForegroundColor',COL_ERR);
        end
        set(jog_btns,'Enable','on');
    end

    %-- Establecer origen ---------------------------------------------
    function on_set_origin(~,~)
        if acquiring, return; end
        jog_x=0; jog_y=0;
        set(lbl_jog_pos,'String',sprintf('X=%+.1f\nY=%+.1f',0,0));
        set(h_cur,'XData',NaN,'YData',NaN);
        try
            cfg2=load(config_file);
            cfg2.homing_date=datestr(now,'yyyy-mm-dd HH:MM:SS');
            save(config_file,'-struct','cfg2');
            set(lbl_status,'String', ...
                sprintf('✔  Origen fijado  (%s)',cfg2.homing_date),'ForegroundColor',COL_OK);
        catch ME
            set(lbl_status,'String',sprintf('ERROR origen: %s',ME.message),'ForegroundColor',COL_ERR);
        end
    end

    %-- START ─────────────────────────────────────────────────────────
    function on_start(~,~)
        if acquiring, return; end

        % ── Parámetros VNA (comunes a ambos modos) ─────────────────────
        fstart_ghz = str2double(get(ed_fstart,'String'));
        fstop_ghz  = str2double(get(ed_fstop, 'String'));
        n_pts      = round(str2double(get(ed_npts,'String')));
        bw_val     = str2double(get(ed_bw,   'String'));
        avg_val    = round(str2double(get(ed_avg,'String')));
        pwr_val    = str2double(get(ed_pwr,  'String'));
        dwell_val  = str2double(get(ed_dwell,'String'));
        notes      = strtrim(get(ed_notes,'String'));

        if any(isnan([fstart_ghz fstop_ghz n_pts bw_val avg_val pwr_val dwell_val]))
            set(lbl_status,'String','ERROR: parámetros VNA inválidos.','ForegroundColor',COL_ERR); return;
        end

        settingsVNA.frequencyRange = [fstart_ghz*1e9, fstop_ghz*1e9];
        settingsVNA.numPoints      = n_pts;
        settingsVNA.BW             = bw_val;
        settingsVNA.avg_N          = avg_val;
        settingsVNA.dBmPower       = pwr_val;
        settingsVNA.freq           = linspace(fstart_ghz*1e9, fstop_ghz*1e9, n_pts)';

        % ══════════════════════════════════════════════════════════════
        if strcmp(meas_type,'healthy')
        % ── RAMA HEALTHY: medida única, sin robot, sin target ─────────

            msg = sprintf(['MEDIDA DE FONDO  (Healthy)\n\n' ...
                           'Frec   : %.2f – %.2f GHz  (%d pts)\n' ...
                           'BW / Avg : %.0f Hz / %d\n\n' ...
                           'El robot permanece en la posición actual.\n' ...
                           'No se necesita Arduino.\n\n¿Comenzar?'], ...
                           fstart_ghz,fstop_ghz,n_pts,bw_val,avg_val);
            if ~strcmp(questdlg(msg,'Confirmar','Sí, comenzar','Cancelar','Cancelar'),'Sí, comenzar')
                return;
            end

            acquiring=true;
            set(btn_start,'Enable','off','BackgroundColor',[0.32 0.32 0.36]);
            set(jog_btns,'Enable','off');
            set(lbl_status,'String','Conectando VNA y switches...','ForegroundColor',COL_BUS);
            set(h_bar,'XData',[0 0 0 0]); drawnow;

            % VNA
            try
                instrObj=setVNA_v3_2ports(settingsVNA);
                fprintf(instrObj,'INIT:CONT OFF');
            catch ME
                set(lbl_status,'String',sprintf('ERROR VNA: %s',ME.message),'ForegroundColor',COL_ERR);
                acquiring=false; set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
                set(jog_btns,'Enable','on'); return;
            end
            % Switches
            try
                NET.addAssembly('C:\Image_to_Image\mcl_SolidStateSwitch_NET45.dll');
                SW1_h=mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
                SW2_h=mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
                SW1_h.Connect('12410080206');
                SW2_h.Connect('12410080235');
            catch ME
                set(lbl_status,'String',sprintf('ERROR Switches: %s',ME.message),'ForegroundColor',COL_ERR);
                try;fclose(instrObj);delete(instrObj);instrObj=[];catch;end
                acquiring=false; set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
                set(jog_btns,'Enable','on'); return;
            end
            % NO se usa Arduino en modo Healthy

            % Ventana de progreso
            set(lbl_status,'String','Midiendo S 12×12 — fondo...','ForegroundColor',COL_BUS); drawnow;
            fig_prog_h=figure('Name','S 12×12 — Healthy (fondo)','NumberTitle','off','Position',[50 50 600 520]);
            auxFill_h=zeros(12,12); Splot_h=imagesc(auxFill_h);
            colormap(fig_prog_h,gray); colorbar; clim([0 1]);
            title('Fondo — celdas medidas'); xlabel('Antena j'); ylabel('Antena i');
            set(gca,'XTick',1:12,'YTick',1:12,'XTickLabel',ant_tick_lbl,'YTickLabel',ant_tick_lbl,'TickLabelInterpreter','none');
            drawnow;

            Sm_h=zeros(12,12,n_pts); err_msg=''; t_start=tic;
            try
                for A=1:6
                    SW1_h.Send_SCPI(sprintf(':SP8T:STATE:%d',A),''); pause(SWITCH_PAUSE_S);
                    for B=1:6
                        SW2_h.Send_SCPI(sprintf(':SP8T:STATE:%d',B),''); pause(SWITCH_PAUSE_S);
                        [s_tot,~]=meas2x2_trigger_MGDR(instrObj,settingsVNA);
                        iA=A; iB=B+6;
                        Sm_h(iA,iA,:)=s_tot(1,1,:); Sm_h(iB,iA,:)=s_tot(2,1,:);
                        Sm_h(iA,iB,:)=s_tot(1,2,:); Sm_h(iB,iB,:)=s_tot(2,2,:);
                        auxFill_h(iA,iA)=1; auxFill_h(iB,iA)=1;
                        auxFill_h(iA,iB)=1; auxFill_h(iB,iB)=1;
                        pct_h=((A-1)*6+B)/36;
                        set(h_bar,'XData',[0 pct_h pct_h 0]);
                        if isvalid(fig_prog_h), set(Splot_h,'CData',auxFill_h); drawnow; end
                    end
                end
            catch ME
                err_msg=ME.message;
            end

            % Atenuador
            s_atten_h=zeros(2,2,n_pts);
            try
                SW1_h.Send_SCPI(':SP8T:STATE:7',''); pause(SWITCH_PAUSE_S);
                SW2_h.Send_SCPI(':SP8T:STATE:7',''); pause(SWITCH_PAUSE_S);
                [s_atten_h,~]=meas2x2_trigger_MGDR(instrObj,settingsVNA);
            catch; end

            % Cerrar VNA
            try;fprintf(instrObj,'OUTP:STAT OFF');fclose(instrObj);delete(instrObj);instrObj=[];catch;end

            if isempty(err_msg)
                set(h_bar,'XData',[0 1 1 0]);
                n_acq=n_acq+1;
                ts=datestr(now,'yyyy-mm-dd HH:MM:SS');
                save_to_hdf5(db_file,n_acq,Sm_h,s_atten_h,settingsVNA.freq,ts,'healthy', ...
                    0,0,0,'(ninguno)',0,0,0,'N/A',notes,settingsVNA);

                % Mostrar resultado
                fm_h=round(n_pts/2);
                Sm_h_dB=20*log10(abs(Sm_h(:,:,fm_h))+eps);
                Sm_h_dB(Sm_h(:,:,fm_h)==0)=-90;
                figure('Name',sprintf('Fondo (Healthy) acq#%04d',n_acq), ...
                    'NumberTitle','off','Position',[680 50 600 520]);
                imagesc(Sm_h_dB); colormap(); colorbar; clim([-90 0]);
                title(sprintf('|Sij| dB  Fondo @ %.2f GHz  acq#%04d', ...
                    settingsVNA.freq(fm_h)/1e9, n_acq));
                set(gca,'XTick',1:12,'YTick',1:12,'XTickLabel',ant_tick_lbl, ...
                    'YTickLabel',ant_tick_lbl,'TickLabelInterpreter','none');
                xlabel('Antena j'); ylabel('Antena i'); drawnow;

                t_tot_h=toc(t_start);
                set(lbl_status,'String', ...
                    sprintf('✔  Healthy #%04d guardado | %.0f s',n_acq,t_tot_h), ...
                    'ForegroundColor',COL_OK);
                entry=sprintf('#%04d[H]|(%+6.1f,%+6.1f)mm|%3.0fdeg|%-14s|%s', ...
                    n_acq,0.0,0.0,0.0,'(ninguno)',ts(1:10));
                prev=get(lb_hist,'String');
                if ischar(prev),prev={prev};end
                if isempty(prev)||(numel(prev)==1&&isempty(prev{1})),prev={};end
                set(lb_hist,'String',[prev;{entry}],'Value',numel(prev)+1);
            else
                set(lbl_status,'String',sprintf('ERROR: %s',err_msg),'ForegroundColor',COL_ERR);
            end
            acquiring=false;
            set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
            set(jog_btns,'Enable','on');

        else
        % ══════════════════════════════════════════════════════════════
        % ── RAMA TARGET: scan completo de posiciones con robot ─────────

            eps_idx = get(dd_eps,'Value');
            epsilon = EPS_OPTS{eps_idx,2};
            eps_lbl = EPS_OPTS{eps_idx,1};
            t_idx   = get(dd_target,'Value');
            tgt_lbl = TGT{t_idx,1};
            rx_mm   = TGT{t_idx,2};
            ry_mm   = TGT{t_idx,3};
            rot_deg = angulos_valid(ang_idx);

            if ~has_positions||t_idx>numel(all_valid_pos)||isempty(all_valid_pos{t_idx})
                set(lbl_status,'String','ERROR: no hay posiciones válidas para este target.','ForegroundColor',COL_ERR); return;
            end

            % Transformación phantom → robot (sin negación)
            raw_pos     = all_valid_pos{t_idx};
            pos_robot   = [raw_pos(:,2), raw_pos(:,1)];
            pos_robot   = boustrophedon(pos_robot);
            N_pos       = size(pos_robot,1);
            pos_phantom = [pos_robot(:,2), pos_robot(:,1)];  % inversa para display

            msg = sprintf(['ADQUISICIÓN — TARGET\n\n' ...
                           'Target     : %s   (%d°)\n' ...
                           'Medio      : %s\n' ...
                           'Posiciones : %d\n' ...
                           'Frec       : %.2f–%.2f GHz  (%d pts)\n' ...
                           'BW/Avg/Dwell: %.0fHz / %d / %.1fs\n\n¿Comenzar?'], ...
                           tgt_lbl,rot_deg,eps_lbl,N_pos, ...
                           fstart_ghz,fstop_ghz,n_pts,bw_val,avg_val,dwell_val);
            if ~strcmp(questdlg(msg,'Confirmar','Sí, comenzar','Cancelar','Cancelar'),'Sí, comenzar')
                return;
            end

            acquiring=true;
            set(btn_start,'Enable','off','BackgroundColor',[0.32 0.32 0.36]);
            set(jog_btns,'Enable','off');
            set(lbl_status,'String','Conectando hardware...','ForegroundColor',COL_BUS);
            set(h_bar,'XData',[0 0 0 0]); drawnow;

            % VNA
            try
                instrObj=setVNA_v3_2ports(settingsVNA);
                fprintf(instrObj,'INIT:CONT OFF');
            catch ME
                set(lbl_status,'String',sprintf('ERROR VNA: %s',ME.message),'ForegroundColor',COL_ERR);
                acquiring=false; set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
                set(jog_btns,'Enable','on'); return;
            end
            % Switches
            try
                NET.addAssembly('C:\Image_to_Image\mcl_SolidStateSwitch_NET45.dll');
                SW1_h=mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
                SW2_h=mcl_SolidStateSwitch_NET45.USB_Digital_Switch;
                SW1_h.Connect('12410080206');
                SW2_h.Connect('12410080235');
            catch ME
                set(lbl_status,'String',sprintf('ERROR Switches: %s',ME.message),'ForegroundColor',COL_ERR);
                try;fclose(instrObj);delete(instrObj);instrObj=[];catch;end
                acquiring=false; set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
                set(jog_btns,'Enable','on'); return;
            end
            % Arduino (reusar si ya estaba conectado por jog)
            try
                if isempty(ser_h)||~isvalid(ser_h)
                    ser_h=serialport(char(arduino_port),9600);
                    configureTerminator(ser_h,'LF');
                    ser_h.Timeout=ACK_TIMEOUT_S;
                    pause(2); ard_ready=true;
                end
            catch ME
                set(lbl_status,'String',sprintf('ERROR Arduino: %s',ME.message),'ForegroundColor',COL_ERR);
                try;fclose(instrObj);delete(instrObj);instrObj=[];catch;end
                acquiring=false; set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
                set(jog_btns,'Enable','on'); return;
            end

            set(lbl_status,'String',sprintf('Escaneando %d posiciones...',N_pos),'ForegroundColor',COL_BUS);
            drawnow;

            % Ventanas auxiliares
            fig_prog=figure('Name','S 12×12 — progreso','NumberTitle','off','Position',[50 50 600 520]);
            auxFill=zeros(12,12); Splot=imagesc(auxFill);
            colormap(fig_prog,gray); colorbar; clim([0 1]);
            title('Celdas medidas (blanco=completado)'); xlabel('Antena j'); ylabel('Antena i');
            set(gca,'XTick',1:12,'YTick',1:12,'XTickLabel',ant_tick_lbl,'YTickLabel',ant_tick_lbl,'TickLabelInterpreter','none');
            drawnow;

            fig_result=figure('Name','S 12×12 — resultado','NumberTitle','off','Position',[680 50 600 520]);
            Splot_res=imagesc(-90*ones(12,12));
            colormap(fig_result); colorbar; clim([-90 0]);
            title('En espera...'); xlabel('Antena j'); ylabel('Antena i');
            set(gca,'XTick',1:12,'YTick',1:12,'XTickLabel',ant_tick_lbl,'YTickLabel',ant_tick_lbl,'TickLabelInterpreter','none');
            drawnow;

            freq_GHz_atten=settingsVNA.freq/1e9;
            fig_atten=figure('Name','Atenuador S21','NumberTitle','off','Position',[50 600 680 280]);
            ax_atten=axes(fig_atten); hold(ax_atten,'on'); grid on; box on;
            xlabel(ax_atten,'Freq (GHz)'); ylabel(ax_atten,'|S21| (dB)');
            title(ax_atten,'Atenuador  SW1:7 → SW2:7','FontSize',10,'FontWeight','bold');
            h_atten_cur=plot(ax_atten,freq_GHz_atten,zeros(1,n_pts),'k-','LineWidth',2,'DisplayName','Última');
            legend(ax_atten,'Location','best','FontSize',8.5); drawnow;

            robot_x=0; robot_y=0;
            done_x=nan(N_pos,1); done_y=nan(N_pos,1);
            err_msg=''; t_start=tic;

            try
                for ip=1:N_pos
                    pos_x=pos_robot(ip,1); pos_y=pos_robot(ip,2);
                    ph_x=pos_phantom(ip,1); ph_y=pos_phantom(ip,2);

                    set(h_cur,'XData',ph_x,'YData',ph_y); drawnow;

                    dx=pos_x-robot_x; dy=pos_y-robot_y;
                    if dx~=0
                        move_axis(ser_h,ternary(dx>0,'X+','X-'),cmd_for_mm(abs(dx),cmd_per_mm_x));
                        robot_x=robot_x+dx;
                    end
                    if dy~=0
                        move_axis(ser_h,ternary(dy>0,'Y+','Y-'),cmd_for_mm(abs(dy),cmd_per_mm_y));
                        robot_y=robot_y+dy;
                    end
                    pause(dwell_val);

                    Sm=zeros(12,12,n_pts); auxFill=zeros(12,12);
                    if isvalid(fig_prog), set(Splot,'CData',auxFill); drawnow; end

                    for A=1:6
                        SW1_h.Send_SCPI(sprintf(':SP8T:STATE:%d',A),''); pause(SWITCH_PAUSE_S);
                        for B=1:6
                            SW2_h.Send_SCPI(sprintf(':SP8T:STATE:%d',B),''); pause(SWITCH_PAUSE_S);
                            [s_tot,~]=meas2x2_trigger_MGDR(instrObj,settingsVNA);
                            iA=A; iB=B+6;
                            Sm(iA,iA,:)=s_tot(1,1,:); Sm(iB,iA,:)=s_tot(2,1,:);
                            Sm(iA,iB,:)=s_tot(1,2,:); Sm(iB,iB,:)=s_tot(2,2,:);
                            auxFill(iA,iA)=1; auxFill(iB,iA)=1;
                            auxFill(iA,iB)=1; auxFill(iB,iB)=1;
                            pct=(ip-1)/N_pos+((A-1)*6+B)/(36*N_pos);
                            set(h_bar,'XData',[0 pct pct 0]);
                            if isvalid(fig_prog), set(Splot,'CData',auxFill); drawnow; end
                        end
                    end

                    % Atenuador
                    SW1_h.Send_SCPI(':SP8T:STATE:7',''); pause(SWITCH_PAUSE_S);
                    SW2_h.Send_SCPI(':SP8T:STATE:7',''); pause(SWITCH_PAUSE_S);
                    [s_atten,~]=meas2x2_trigger_MGDR(instrObj,settingsVNA);
                    S21_dB=20*log10(abs(squeeze(s_atten(2,1,:))') + eps);
                    if isvalid(fig_atten)
                        set(h_atten_cur,'Color',[0.75 0.75 0.75],'LineWidth',0.8,'HandleVisibility','off');
                        h_atten_cur=plot(ax_atten,freq_GHz_atten,S21_dB,'k-','LineWidth',2,'DisplayName','Última');
                        fc=round(n_pts/2);
                        title(ax_atten,sprintf('Atenuador | pos%d/%d (X=%+.0f,Y=%+.0f)mm | S21=%.1fdB@%.2fGHz', ...
                            ip,N_pos,pos_x,pos_y,S21_dB(fc),freq_GHz_atten(fc)),'FontSize',8.5,'FontWeight','bold');
                        drawnow;
                    end

                    % Guardar HDF5
                    n_acq=n_acq+1;
                    ts=datestr(now,'yyyy-mm-dd HH:MM:SS');
                    save_to_hdf5(db_file,n_acq,Sm,s_atten,settingsVNA.freq,ts,'target', ...
                        pos_x,pos_y,rot_deg,tgt_lbl,rx_mm,ry_mm,epsilon,eps_lbl,notes,settingsVNA);

                    done_x(ip)=ph_x; done_y(ip)=ph_y;
                    set(h_done,'XData',done_x(~isnan(done_x)),'YData',done_y(~isnan(done_y)));

                    % Resultado
                    fm=round(n_pts/2); Sm_dB=20*log10(abs(Sm(:,:,fm))+eps);
                    Sm_dB(Sm(:,:,fm)==0)=-90;
                    if isvalid(fig_result)
                        set(fig_result,'Name',sprintf('Resultado pos %d/%d  acq#%04d',ip,N_pos,n_acq));
                        set(Splot_res,'CData',Sm_dB);
                        title(get(fig_result,'CurrentAxes'), ...
                            sprintf('|Sij| dB @ %.2fGHz — pos%d/%d (X=%+.0f,Y=%+.0f)mm', ...
                                settingsVNA.freq(fm)/1e9,ip,N_pos,pos_x,pos_y)); drawnow;
                    end

                    t_el=toc(t_start); t_rem=t_el/ip*(N_pos-ip);
                    set(h_bar,'XData',[0 ip/N_pos ip/N_pos 0]);
                    set(lbl_status,'String', ...
                        sprintf('[%d/%d]  acq#%04d  |  ~%.0f min restantes',ip,N_pos,n_acq,t_rem/60), ...
                        'ForegroundColor',COL_BUS);
                    entry=sprintf('#%04d[T]|(%+6.1f,%+6.1f)mm|%3.0fdeg|%-14s|%s', ...
                        n_acq,pos_x,pos_y,rot_deg,tgt_lbl,ts(1:10));
                    prev=get(lb_hist,'String');
                    if ischar(prev),prev={prev};end
                    if isempty(prev)||(numel(prev)==1&&isempty(prev{1})),prev={};end
                    set(lb_hist,'String',[prev;{entry}],'Value',numel(prev)+1);
                    drawnow;
                end
            catch ME
                err_msg=ME.message;
            end

            % Volver al origen
            set(lbl_status,'String','Volviendo al origen...','ForegroundColor',COL_BUS); drawnow;
            try
                if robot_x~=0, move_axis(ser_h,ternary(robot_x>0,'X-','X+'),cmd_for_mm(abs(robot_x),cmd_per_mm_x)); end
                if robot_y~=0, move_axis(ser_h,ternary(robot_y>0,'Y-','Y+'),cmd_for_mm(abs(robot_y),cmd_per_mm_y)); end
            catch; end
            set(h_cur,'XData',NaN,'YData',NaN);

            % Cerrar VNA
            try;fprintf(instrObj,'OUTP:STAT OFF');fclose(instrObj);delete(instrObj);instrObj=[];catch;end

            if isempty(err_msg)
                t_tot=toc(t_start);
                set(h_bar,'XData',[0 1 1 0]);
                set(lbl_status,'String', ...
                    sprintf('✔  Completado: %d pos. | Acq#%04d→#%04d | %.1f min', ...
                        N_pos,n_prev+1,n_acq,t_tot/60),'ForegroundColor',COL_OK);
            else
                set(lbl_status,'String',sprintf('ERROR: %s',err_msg),'ForegroundColor',COL_ERR);
            end
            acquiring=false;
            set(btn_start,'Enable','on','BackgroundColor',[0.10 0.50 0.20]);
            set(jog_btns,'Enable','on');
        end   % if healthy/target
    end   % on_start

    %-- Historial -------------------------------------------------------
    function load_db_history()
        if ~isfile(db_file)||n_prev==0, return; end
        try
            info=h5info(db_file); ng=numel(info.Groups); entries=cell(ng,1);
            for g=1:ng
                gn=info.Groups(g).Name;
                try
                    ts_g=h5readatt(db_file,gn,'timestamp');
                    px_g=h5readatt(db_file,gn,'pos_x_mm');
                    py_g=h5readatt(db_file,gn,'pos_y_mm');
                    rd_g=h5readatt(db_file,gn,'rotation_deg');
                    tt_g=h5readatt(db_file,gn,'target_type');
                    ep_g=h5readatt(db_file,gn,'target_epsilon_re');
                    try; mt_g=h5readatt(db_file,gn,'meas_type'); catch; mt_g='target'; end
                    mt_c='T'; if strcmpi(strtrim(mt_g),'healthy'), mt_c='H'; end
                    nm_g=str2double(gn(end-3:end)); if isnan(nm_g),nm_g=g;end
                    entries{g}=sprintf('#%04d[%s]|(%+6.1f,%+6.1f)mm|%3.0fdeg|%-14s|eps=%.0f|%s', ...
                        nm_g,mt_c,px_g,py_g,rd_g,tt_g,ep_g,ts_g(1:10));
                catch
                    entries{g}=sprintf('%-20s  [sin metadatos]',gn);
                end
            end
            set(lb_hist,'String',entries,'Value',ng);
        catch; end
    end

    %-- Cierre ---------------------------------------------------------
    function on_close(~,~)
        if acquiring
            if ~strcmp(questdlg('Adquisición en curso. ¿Cerrar?','Cerrar', ...
                    'Sí, cerrar','No','No'),'Sí, cerrar'), return; end
        end
        try
            if ~isempty(instrObj)&&isvalid(instrObj)
                fprintf(instrObj,'OUTP:STAT OFF');fclose(instrObj);delete(instrObj);
            end
        catch; end
        delete(fig);
    end

    %-- Helpers de layout (acceden a PX, BG, IBG del parent) ----------
    function sec_lbl(txt,y)
        uicontrol('Style','text','String',txt,'Position',[PX y PW 15], ...
            'BackgroundColor',BG,'ForegroundColor',[0.50 0.50 0.58], ...
            'FontSize',7.5,'HorizontalAlignment','left');
    end
    function lbl_s(txt,y,xo,w)
        uicontrol('Style','text','String',txt,'Position',[PX+xo y w 19], ...
            'BackgroundColor',BG,'ForegroundColor',[0.68 0.68 0.68], ...
            'FontSize',8,'HorizontalAlignment','left');
    end
    function h = edt(val,y,xo,w)
        h = uicontrol('Style','edit','String',num2str(val), ...
            'Position',[PX+xo y-1 w 22],'BackgroundColor',IBG,'ForegroundColor',FG, ...
            'FontSize',9,'HorizontalAlignment','center');
    end
    function h = jog_btn(xo,y,w,bh,str,cb)
        h = uicontrol('Style','pushbutton','String',str, ...
            'Position',[PX+xo y w bh],'BackgroundColor',[0.12 0.24 0.40], ...
            'ForegroundColor',[0.55 0.88 1.00],'FontSize',9.5,'FontWeight','bold', ...
            'Callback',cb);
    end

end   % step4_acquisition_gui

%% ================================================================
%  FUNCIONES LOCALES
%% ================================================================

function pos = boustrophedon(pos)
    y_vals=unique(pos(:,2)); ps=[];
    for iy=1:numel(y_vals)
        mask=pos(:,2)==y_vals(iy); row=pos(mask,:);
        if mod(iy,2)==1,row=sortrows(row,1);else,row=sortrows(row,-1);end
        ps=[ps;row]; %#ok<AGROW>
    end
    pos=ps;
end

function n = cmd_for_mm(d,c)
    n=max(1,round(d*c));
end

function out = ternary(cond,a,b)
    if cond,out=a;else,out=b;end
end

function move_axis(ser,cmd,n)
    for i=1:n
        writeline(ser,cmd);
        ack=readline(ser);
        if ~strcmp(strtrim(ack),'OK')
            warning('ACK inesperado paso %d/%d: "%s"',i,n,strtrim(ack));
        end
    end
end

function save_to_hdf5(db_file,acq_num,Sm,s_atten,freq,timestamp,meas_type, ...
                      pos_x,pos_y,rot_deg,tgt_lbl,rx_mm,ry_mm, ...
                      epsilon,eps_lbl,notes,settingsVNA)
    N=numel(freq); grp=sprintf('/acq_%04d',acq_num);
    h5create(db_file,[grp '/Sm_real'],      [12 12 N],'Datatype','double');
    h5write( db_file,[grp '/Sm_real'],      real(Sm));
    h5create(db_file,[grp '/Sm_imag'],      [12 12 N],'Datatype','double');
    h5write( db_file,[grp '/Sm_imag'],      imag(Sm));
    h5create(db_file,[grp '/freq'],         [N 1],    'Datatype','double');
    h5write( db_file,[grp '/freq'],         freq(:));
    h5create(db_file,[grp '/s_atten_real'], [2 2 N],  'Datatype','double');
    h5write( db_file,[grp '/s_atten_real'], real(s_atten));
    h5create(db_file,[grp '/s_atten_imag'], [2 2 N],  'Datatype','double');
    h5write( db_file,[grp '/s_atten_imag'], imag(s_atten));
    h5writeatt(db_file,grp,'timestamp',         timestamp);
    h5writeatt(db_file,grp,'meas_type',         meas_type);
    h5writeatt(db_file,grp,'pos_x_mm',          pos_x);
    h5writeatt(db_file,grp,'pos_y_mm',          pos_y);
    h5writeatt(db_file,grp,'rotation_deg',      rot_deg);
    h5writeatt(db_file,grp,'target_type',       tgt_lbl);
    h5writeatt(db_file,grp,'target_rx_mm',      rx_mm);
    h5writeatt(db_file,grp,'target_ry_mm',      ry_mm);
    h5writeatt(db_file,grp,'target_epsilon_re', epsilon);
    h5writeatt(db_file,grp,'target_eps_label',  eps_lbl);
    h5writeatt(db_file,grp,'notes',             notes);
    h5writeatt(db_file,grp,'acquired_by',       'step4_acquisition_gui');
    h5writeatt(db_file,grp,'vna_fstart_Hz',     settingsVNA.frequencyRange(1));
    h5writeatt(db_file,grp,'vna_fstop_Hz',      settingsVNA.frequencyRange(2));
    h5writeatt(db_file,grp,'vna_numPoints',     double(settingsVNA.numPoints));
    h5writeatt(db_file,grp,'vna_BW_Hz',         double(settingsVNA.BW));
    h5writeatt(db_file,grp,'vna_power_dBm',     double(settingsVNA.dBmPower));
end

function n = count_acquisitions(db_file)
    n=0; if ~isfile(db_file),return;end
    try;info=h5info(db_file,'/');n=numel(info.Groups);catch;end
end
