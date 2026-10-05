clc; clear; close all;

%% ================================================================
%  STEP 3 — Homing del robot XY
%  Robot de adquisición automática de targets — POLITO
%  ----------------------------------------------------------------
%  Posiciona el stroke en el centro del phantom (0,0) y confirma
%  ese punto como origen para la adquisición automática.
%
%  ANTES DE CORRER ESTE SCRIPT:
%    1. Haber ejecutado step1 y step2.
%    2. Instalar el soporte del stroke y actualizar OFFSET_Y_MM
%       en step1_calibrate_robot.m (valor en consola de OpenSCAD).
%    3. Insertar el stroke en el collarin y bloquearlo.
%    4. Colocar el phantom en posición de medida.
%% ================================================================

%% ── Parámetros ─────────────────────────────────────────────────
JOG_MM        = 1.0;    % paso de jog por defecto [mm]
ACK_TIMEOUT_S = 30;

%% ── Carpetas ────────────────────────────────────────────────────
root_folder  = fileparts(mfilename('fullpath'));
config_file  = fullfile(root_folder, 'config', 'robot_config.mat');

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   Robot XY — Homing  (Step 3)                   ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

if ~isfile(config_file)
    error('No se encontró robot_config.mat.\nEjecuta primero step1_calibrate_robot.m\n');
end

load(config_file, 'arduino_port', 'cmd_per_mm_x', 'cmd_per_mm_y', ...
                  'OFFSET_X_MM', 'OFFSET_Y_MM', 'calibration_date');

fprintf('Calibración cargada: %s\n', calibration_date);
fprintf('  cmd/mm X : %.5f\n', cmd_per_mm_x);
fprintf('  cmd/mm Y : %.5f\n', cmd_per_mm_y);
fprintf('\n');

%% ── Recordatorio sobre el offset mecánico ──────────────────────
if OFFSET_Y_MM == 0
    fprintf(['  ⚠  OFFSET_Y_MM = 0 en la configuración.\n' ...
             '     Si el soporte ya está impreso, mide COLLARIN_OFFSET\n' ...
             '     (valor en consola de OpenSCAD) y actualiza OFFSET_Y_MM\n' ...
             '     en step1_calibrate_robot.m antes de adquirir.\n\n']);
else
    fprintf('  OFFSET_Y_MM = %.1f mm  ← stroke desplazado en Y del carrito\n\n', OFFSET_Y_MM);
end
fprintf(['  Al hacer homing, alinea el STROKE (no el carrito) con el\n' ...
         '  centro del phantom — el stroke está a %.1f mm del carrito en Y.\n\n'], OFFSET_Y_MM);

input('  Pulsa ENTER cuando el phantom esté en posición de medida: ', 's');

%% ── Conexión al Arduino ─────────────────────────────────────────
fprintf('\nPuerto guardado: %s\n', arduino_port);
use_saved = input('¿Usar ese puerto? [S/n]: ', 's');

if strcmpi(strtrim(use_saved), 'n')
    ports = serialportlist("available");
    if isempty(ports), error('No se encontraron puertos serie disponibles.'); end
    fprintf('\nPuertos disponibles:\n');
    for k = 1:numel(ports), fprintf('  [%d]  %s\n', k, ports(k)); end
    sel = input('Selecciona número: ');
    arduino_port = ports(sel);
end

fprintf('Conectando a %s...\n', arduino_port);
s = serialport(arduino_port, 9600);
configureTerminator(s, "LF");
s.Timeout = ACK_TIMEOUT_S;
pause(2);
fprintf('✔ Conectado.\n\n');

%% ── Menú de jog fino ────────────────────────────────────────────
pos_x = 0;
pos_y = 0;

print_menu(JOG_MM);

homed   = false;
running = true;
while running

    fprintf('  Desplazamiento: X=%+6.1f mm  Y=%+6.1f mm\n', pos_x, pos_y);
    opt = input('  Opción: ');

    switch opt

        case 1   % X+
            n = cmd_for_mm(JOG_MM, cmd_per_mm_x);
            move_axis(s, 'X+', n);
            pos_x = pos_x + JOG_MM;
            fprintf('  ✔ X+\n\n');

        case 2   % X-
            n = cmd_for_mm(JOG_MM, cmd_per_mm_x);
            move_axis(s, 'X-', n);
            pos_x = pos_x - JOG_MM;
            fprintf('  ✔ X-\n\n');

        case 3   % Y+
            n = cmd_for_mm(JOG_MM, cmd_per_mm_y);
            move_axis(s, 'Y+', n);
            pos_y = pos_y + JOG_MM;
            fprintf('  ✔ Y+\n\n');

        case 4   % Y-
            n = cmd_for_mm(JOG_MM, cmd_per_mm_y);
            move_axis(s, 'Y-', n);
            pos_y = pos_y - JOG_MM;
            fprintf('  ✔ Y-\n\n');

        case 5   % Confirmar HOME
            fprintf('\n  ¿El stroke está centrado en el phantom?\n');
            conf = input('  Confirmar HOME aquí [S/n]: ', 's');
            if ~strcmpi(strtrim(conf), 'n')
                homed   = true;
                running = false;
            else
                fprintf('  Homing cancelado — continúa posicionando.\n\n');
            end

        case 7   % Cambiar paso
            JOG_MM = input('  Nuevo paso (mm): ');
            fprintf('  ✔ Paso = %.2f mm\n\n', JOG_MM);
            print_menu(JOG_MM);

        case 0
            running = false;

        otherwise
            fprintf('  Opción no reconocida.\n\n');
    end
end

%% ── Guardar y cerrar ────────────────────────────────────────────
if homed
    homing_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    save(config_file, '-append', 'homing_date');
    fprintf('\n╔══════════════════════════════════════════════════╗\n');
    fprintf('║   HOMING COMPLETADO  ✔                           ║\n');
    fprintf('╚══════════════════════════════════════════════════╝\n');
    fprintf('  Fecha : %s\n', homing_date);
    fprintf('  Stroke en (0, 0) mm del phantom.\n\n');
    fprintf('Próximo paso:  step4_acquisition.m\n');
else
    fprintf('\n⚠ Homing no confirmado. Ejecuta de nuevo cuando estés listo.\n');
end

clear s;
fprintf('Conexión cerrada.\n');


%% ================================================================
%  FUNCIONES LOCALES
%% ================================================================

function print_menu(paso)
    fprintf('══════════════════════════════════════════════════\n');
    fprintf('  JOG FINO  (paso = %.2f mm por acción)\n', paso);
    fprintf('══════════════════════════════════════════════════\n');
    fprintf('  [1] X+   [2] X-   [3] Y+   [4] Y-\n');
    fprintf('  [5] Confirmar HOME  (stroke = centro phantom)\n');
    fprintf('  [7] Cambiar paso de jog\n');
    fprintf('  [0] Salir sin hacer homing\n');
    fprintf('══════════════════════════════════════════════════\n\n');
end

function n = cmd_for_mm(dist_mm, cmd_per_mm)
    n = max(1, round(dist_mm * cmd_per_mm));
end

function move_axis(ser, cmd, n_steps)
    for i = 1:n_steps
        writeline(ser, cmd);
        ack = readline(ser);
        if ~strcmp(strtrim(ack), 'OK')
            warning('Respuesta inesperada en paso %d/%d: "%s"', i, n_steps, strtrim(ack));
        end
        if mod(i, 10) == 0
            fprintf('    %d / %d\n', i, n_steps);
        end
    end
end
