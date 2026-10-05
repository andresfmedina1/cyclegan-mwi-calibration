clc; clear; close all;

%% ================================================================
%  STEP 2 — Test de movimiento del robot XY
%  Robot de adquisición automática de targets — POLITO
%  ----------------------------------------------------------------
%  Verifica que los tres motores responden correctamente ANTES de
%  realizar cualquier adquisición:
%
%    ✔  X+/X- mueven el carro cruzado en la dirección correcta
%    ✔  Y+/Y- mueven ambos rieles paralelos en la misma dirección
%    ✔  Los dos motores Y se mueven perfectamente sincronizados
%         (el carro no se tuerce ni los motores luchan entre sí)
%    ✔  Las distancias recorridas coinciden con los factores de
%         calibración guardados en config/robot_config.mat
%
%  ANTES DE CORRER ESTE SCRIPT:
%    1. Haber ejecutado step1_calibrate_robot.m (genera robot_config.mat)
%    2. Asegurarse de que el robot tiene recorrido libre en todas
%       las direcciones a probar
%    3. OPCIONAL: colocar una regla junto a cada eje para confirmar
%       que el desplazamiento es el esperado
%% ================================================================

%% ── Parámetros editables ────────────────────────────────────────
JOG_DIST_MM    = 30;     % Distancia de cada jog en el menú  [mm]
ACK_TIMEOUT_S  = 30;     % Tiempo máximo esperando "OK" del Arduino [s]

%% ── Carpetas ────────────────────────────────────────────────────
root_folder   = fileparts(mfilename('fullpath'));
config_folder = fullfile(root_folder, 'config');
config_file   = fullfile(config_folder, 'robot_config.mat');

%% ── 1. Cargar calibración ───────────────────────────────────────
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   Robot XY — Test de movimiento  (Step 2)        ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

if ~isfile(config_file)
    error(['No se encontró el archivo de calibración:\n  %s\n' ...
           'Ejecuta primero  step1_calibrate_robot.m\n'], config_file);
end

load(config_file, 'arduino_port', 'mm_per_cmd_x', 'mm_per_cmd_y', ...
                  'cmd_per_mm_x', 'cmd_per_mm_y', 'calibration_date');

fprintf('Calibración cargada: %s\n', calibration_date);
fprintf('  Eje X : %.5f mm/cmd  (%.5f cmd/mm)\n', mm_per_cmd_x, cmd_per_mm_x);
fprintf('  Eje Y : %.5f mm/cmd  (%.5f cmd/mm)\n', mm_per_cmd_y, cmd_per_mm_y);
fprintf('\n');

%% ── 2. Conexión al Arduino ──────────────────────────────────────
fprintf('Puerto guardado en calibración: %s\n', arduino_port);
use_saved = input('¿Usar ese puerto? [S/n]: ', 's');

if strcmpi(strtrim(use_saved), 'n')
    ports = serialportlist("available");
    if isempty(ports)
        error('No se encontraron puertos serie disponibles.');
    end
    fprintf('\nPuertos disponibles:\n');
    for k = 1:numel(ports)
        fprintf('  [%d]  %s\n', k, ports(k));
    end
    sel = input('Selecciona el número de puerto: ');
    arduino_port = ports(sel);
end

fprintf('\nConectando a %s...\n', arduino_port);
s = serialport(arduino_port, 9600);
configureTerminator(s, "LF");
s.Timeout = ACK_TIMEOUT_S;
pause(2);   % Esperar reset automático del Arduino
fprintf('✔ Conexión establecida.\n\n');

%% ── 3. Menú de jog ─────────────────────────────────────────────
% Mantener un registro de posición para poder volver al origen
pos_x = 0;   % desplazamiento acumulado desde el inicio  [mm]
pos_y = 0;

fprintf('══════════════════════════════════════════════════\n');
fprintf('  MENÚ DE JOG  (jog = %.0f mm nominales por acción)\n', JOG_DIST_MM);
fprintf('══════════════════════════════════════════════════\n');
fprintf('  [1]  X+   mover carro en X positivo\n');
fprintf('  [2]  X-   mover carro en X negativo\n');
fprintf('  [3]  Y+   mover rieles en Y positivo\n');
fprintf('  [4]  Y-   mover rieles en Y negativo\n');
fprintf('  [5]  TEST SINCRONISMO Y — mueve 5 mm Y+ y 5 mm Y-\n');
fprintf('       (observar si el puente cruza recto)\n');
fprintf('  [6]  GOTO ORIGIN — volver a posición (0,0)\n');
fprintf('  [0]  SALIR\n');
fprintf('══════════════════════════════════════════════════\n\n');

running = true;
while running

    fprintf('  Posición actual: X=%.1f mm  Y=%.1f mm\n', pos_x, pos_y);
    opt = input('  Opción: ');

    switch opt

        case 1   % X+
            n = cmd_for_mm('x', JOG_DIST_MM, cmd_per_mm_x);
            fprintf('  → X+ %d comandos (≈ %.1f mm)...\n', n, JOG_DIST_MM);
            move_axis(s, 'X+', n);
            pos_x = pos_x + JOG_DIST_MM;
            fprintf('  ✔ Completado.\n\n');

        case 2   % X-
            n = cmd_for_mm('x', JOG_DIST_MM, cmd_per_mm_x);
            fprintf('  → X- %d comandos (≈ %.1f mm)...\n', n, JOG_DIST_MM);
            move_axis(s, 'X-', n);
            pos_x = pos_x - JOG_DIST_MM;
            fprintf('  ✔ Completado.\n\n');

        case 3   % Y+
            n = cmd_for_mm('y', JOG_DIST_MM, cmd_per_mm_y);
            fprintf('  → Y+ %d comandos (≈ %.1f mm)...\n', n, JOG_DIST_MM);
            move_axis(s, 'Y+', n);
            pos_y = pos_y + JOG_DIST_MM;
            fprintf('  ✔ Completado.\n\n');

        case 4   % Y-
            n = cmd_for_mm('y', JOG_DIST_MM, cmd_per_mm_y);
            fprintf('  → Y- %d comandos (≈ %.1f mm)...\n', n, JOG_DIST_MM);
            move_axis(s, 'Y-', n);
            pos_y = pos_y - JOG_DIST_MM;
            fprintf('  ✔ Completado.\n\n');

        case 5   % TEST SINCRONISMO Y
            fprintf('\n  ── Test de sincronismo Y ──────────────────\n');
            fprintf('  OBSERVA el puente cruzado mientras se mueven los rieles:\n');
            fprintf('    • Si se mueve recto        → Y1/Y2 sincronizados ✔\n');
            fprintf('    • Si se tuerce o hay ruido → cambiar Y2_DIR_INVERTED\n');
            fprintf('                                  en robot_arduino.ino\n\n');
            input('  Pulsa ENTER para comenzar el test: ', 's');

            sync_mm = 5;
            n = cmd_for_mm('y', sync_mm, cmd_per_mm_y);
            fprintf('  Moviendo Y+ %d mm...\n', sync_mm);
            move_axis(s, 'Y+', n);
            pause(0.5);
            fprintf('  Moviendo Y- %d mm (vuelta al origen)...\n', sync_mm);
            move_axis(s, 'Y-', n);

            fprintf('\n  ¿El puente se movió recto en ambas direcciones?\n');
            resp = input('  [S = sí, N = no]: ', 's');
            if strcmpi(strtrim(resp), 's')
                fprintf('  ✔ Sincronismo Y correcto.\n\n');
            else
                fprintf(['  ⚠ Los motores Y no están sincronizados.\n' ...
                         '  Solución:\n' ...
                         '    1. Abrir  arduino/robot_arduino.ino\n' ...
                         '    2. Cambiar  #define Y2_DIR_INVERTED  false\n' ...
                         '                        →          true  (o viceversa)\n' ...
                         '    3. Recargar el firmware en el Arduino\n' ...
                         '    4. Reiniciar este script\n\n']);
            end

        case 6   % GOTO ORIGIN
            fprintf('\n  ── Volviendo al origen (0,0) ──────────────\n');
            fprintf('  Posición actual: X=%.1f mm  Y=%.1f mm\n', pos_x, pos_y);

            if pos_x ~= 0
                dir_x = ternary(pos_x > 0, 'X-', 'X+');
                n     = cmd_for_mm('x', abs(pos_x), cmd_per_mm_x);
                fprintf('  → %s %d comandos...\n', dir_x, n);
                move_axis(s, dir_x, n);
                pos_x = 0;
            end
            if pos_y ~= 0
                dir_y = ternary(pos_y > 0, 'Y-', 'Y+');
                n     = cmd_for_mm('y', abs(pos_y), cmd_per_mm_y);
                fprintf('  → %s %d comandos...\n', dir_y, n);
                move_axis(s, dir_y, n);
                pos_y = 0;
            end
            fprintf('  ✔ En origen. Posición: X=0  Y=0\n\n');

        case 0
            running = false;

        otherwise
            fprintf('  Opción no reconocida.\n\n');
    end
end

%% ── 4. Informe final ────────────────────────────────────────────
fprintf('\n══════════════════════════════════════════════════\n');
fprintf('  RESUMEN DEL TEST\n');
fprintf('══════════════════════════════════════════════════\n');
fprintf('  Posición final del carro: X=%.1f mm  Y=%.1f mm\n', pos_x, pos_y);
if abs(pos_x) > 0.5 || abs(pos_y) > 0.5
    fprintf('  ⚠ El carro NO está en el origen.\n');
    fprintf('     Muévelo manualmente antes de desconectar.\n');
else
    fprintf('  ✔ El carro está en el origen.\n');
end
fprintf('\n  Lista de comprobaciones antes del Step 3:\n');
fprintf('  [ ]  X+ mueve el carro en la dirección deseada\n');
fprintf('  [ ]  X- vuelve al punto de partida\n');
fprintf('  [ ]  Y+ mueve ambos rieles juntos, en línea recta\n');
fprintf('  [ ]  Y- vuelve al punto de partida, sin desviación\n');
fprintf('  [ ]  Sincronismo Y confirmado (opción 5)\n');
fprintf('\nConexión serie cerrada.\n');
fprintf('Próximo paso:  step3_homing.m\n');

clear s;

%% ================================================================
%  FUNCIONES LOCALES
%% ================================================================

% Calcula cuántos comandos de 1 mm son necesarios para cubrir
% una distancia real en mm usando el factor de calibración.
function n = cmd_for_mm(~, dist_mm, cmd_per_mm)
    n = max(1, round(dist_mm * cmd_per_mm));
end

% Operador ternario simple
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

% Envía N comandos de 1 mm al eje indicado; espera ACK por cada uno.
function move_axis(ser, cmd, n_steps)
    for i = 1:n_steps
        writeline(ser, cmd);
        ack = readline(ser);
        if ~strcmp(strtrim(ack), 'OK')
            warning('Respuesta inesperada en paso %d/%d: "%s"', ...
                    i, n_steps, strtrim(ack));
        end
        if mod(i, 10) == 0
            fprintf('    %d / %d\n', i, n_steps);
        end
    end
end
