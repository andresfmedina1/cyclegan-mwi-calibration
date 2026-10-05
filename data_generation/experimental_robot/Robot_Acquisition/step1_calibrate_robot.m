clc; clear; close all;

%% ================================================================
%  STEP 1 — Calibración del robot XY
%  Robot de adquisición automática de targets — POLITO
%  ----------------------------------------------------------------
%  Determina cuántos milímetros se mueve REALMENTE el robot por
%  cada comando enviado al Arduino.
%
%  Resultado: guarda  config/robot_config.mat  con los factores de
%  calibración que usarán todos los scripts futuros del robot.
%
%  ANTES DE CORRER ESTE SCRIPT:
%    1. Abrir  arduino/robot_arduino.ino  en el Arduino IDE
%    2. Cargar el firmware en el Arduino UNO
%    3. Conectar los drivers M422 y los motores según el diagrama
%    4. Comprobar que los carros se mueven libremente (sin topes)
%    5. Tener a mano un calibrador o regla de precisión (>= 60 mm)
%% ================================================================

%% ── Parámetros editables ────────────────────────────────────────
CAL_DIST_CMD          = 50;      % Nº de comandos para calibrar (= mm nominales)
ACK_TIMEOUT_S         = 30;      % Tiempo máximo esperando "OK" del Arduino [s]
DWELL_S               = 1.0;     % Pausa tras llegada a posición (para adquisición) [s]
OFFSET_X_MM           = 0.0;     % Offset carro→punta del target en X [mm]
                                  %   (dejar en 0 hasta tener el diseño de la sujeción)
OFFSET_Y_MM           = 0.0;     % Offset carro→punta del target en Y [mm]
STEPS_PER_MM_FIRMWARE = 6400.0;  % Valor actual de STEPS_PER_MM en robot_arduino.ino
                                  %   Motor T6×1: (200 × 32 microsteps) / 1 mm = 6400

%% ── Carpetas ────────────────────────────────────────────────────
root_folder   = fileparts(mfilename('fullpath'));
config_folder = fullfile(root_folder, 'config');
if ~exist(config_folder, 'dir'), mkdir(config_folder); end
config_file = fullfile(config_folder, 'robot_config.mat');

%% ── 1. Selección del puerto serie ──────────────────────────────
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║   Robot XY — Calibración paso/mm  (Step 1)      ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

ports = serialportlist("available");
if isempty(ports)
    error(['No se encontraron puertos serie disponibles.\n' ...
           'Asegúrate de que el Arduino está conectado por USB.']);
end

fprintf('Puertos serie disponibles:\n');
for k = 1:numel(ports)
    fprintf('  [%d]  %s\n', k, ports(k));
end
sel = input('\nSelecciona el número del puerto del Arduino: ');
if isnan(sel) || sel < 1 || sel > numel(ports)
    error('Selección inválida.');
end
arduino_port = ports(sel);
fprintf('\n✔ Puerto seleccionado: %s\n\n', arduino_port);

%% ── 2. Conexión al Arduino ──────────────────────────────────────
fprintf('Conectando...\n');
s = serialport(arduino_port, 9600);
configureTerminator(s, "LF");
s.Timeout = ACK_TIMEOUT_S;
pause(2);   % Esperar al reset automático del Arduino tras abrir puerto
fprintf('✔ Conexión establecida.\n\n');

%% ── 3. Calibración eje X ────────────────────────────────────────
fprintf('══════════════════════════════════════════════════\n');
fprintf('  CALIBRACIÓN  EJE X\n');
fprintf('══════════════════════════════════════════════════\n');
fprintf('  Pasos:\n');
fprintf('  1. Coloca el carro X en un punto de referencia cómodo.\n');
fprintf('  2. Sitúa un calibrador con el "0" exactamente en esa posición.\n');
fprintf('  3. Pulsa ENTER: el robot moverá %d mm nominales en X+.\n', CAL_DIST_CMD);
fprintf('  4. Mide la distancia real con el calibrador.\n\n');
input('  → Pulsa ENTER cuando el calibrador esté en posición: ', 's');

fprintf('\n  Moviendo %d comandos en X+  (espera ~%.0f s)...\n', ...
        CAL_DIST_CMD, CAL_DIST_CMD * 0.25);
move_axis(s, 'X+', CAL_DIST_CMD);
fprintf('  ✔ Movimiento X completado.\n\n');

measured_x = input('  Distancia REAL medida en X (mm): ');
if isnan(measured_x) || measured_x <= 0
    error('La distancia medida debe ser un número positivo.');
end

mm_per_cmd_x = measured_x / CAL_DIST_CMD;   % mm reales por comando
cmd_per_mm_x = CAL_DIST_CMD / measured_x;   % comandos necesarios por mm real
error_pct_x  = abs(mm_per_cmd_x - 1.0) * 100;
steps_sugg_x = STEPS_PER_MM_FIRMWARE / mm_per_cmd_x;

fprintf('\n  ── Resultado X ──────────────────────────\n');
fprintf('  Comandos enviados  : %d\n',         CAL_DIST_CMD);
fprintf('  Distancia nominal  : %.1f mm\n',    CAL_DIST_CMD);
fprintf('  Distancia medida   : %.3f mm\n',    measured_x);
fprintf('  mm / comando       : %.5f\n',       mm_per_cmd_x);
fprintf('  Comandos / mm real : %.5f\n',       cmd_per_mm_x);
fprintf('  Error              : %.2f%%\n',     error_pct_x);
if error_pct_x > 1.0
    fprintf('  ⚠ Error > 1%%. STEPS_PER_MM sugerido: %.4f\n', steps_sugg_x);
else
    fprintf('  ✔ Calibración X correcta (error < 1%%).\n');
end

fprintf('\n  Volviendo al origen X...\n');
move_axis(s, 'X-', CAL_DIST_CMD);
input('  ¿El carro volvió al punto de partida? Pulsa ENTER: ', 's');

%% ── 4. Calibración eje Y (dos rieles en paralelo) ───────────────
fprintf('\n══════════════════════════════════════════════════\n');
fprintf('  CALIBRACIÓN  EJE Y  (Y1 + Y2 sincrónicos)\n');
fprintf('══════════════════════════════════════════════════\n');
fprintf('  NOTA: al enviar "Y+" ambos motores (Y1 y Y2) se\n');
fprintf('  mueven simultáneamente desde el Arduino.\n');
fprintf('  Hay que medir los DOS extremos del carro cruzado\n');
fprintf('  para detectar diferencias mecánicas entre rieles.\n\n');
fprintf('  Pasos:\n');
fprintf('  1. Coloca el carro Y en un punto de referencia.\n');
fprintf('  2. Sitúa un calibrador en el extremo IZQUIERDO (riel Y1)\n');
fprintf('     y anota la referencia inicial.\n');
fprintf('  3. Haz lo mismo en el extremo DERECHO (riel Y2).\n');
fprintf('  4. Pulsa ENTER: el robot moverá %d mm nominales en Y+.\n', CAL_DIST_CMD);
fprintf('  5. Mide la distancia real en ambos extremos.\n\n');
input('  → Pulsa ENTER cuando ambos calibradores estén en posición: ', 's');

fprintf('\n  Moviendo %d comandos en Y+  (espera ~%.0f s)...\n', ...
        CAL_DIST_CMD, CAL_DIST_CMD * 0.25);
move_axis(s, 'Y+', CAL_DIST_CMD);
fprintf('  ✔ Movimiento Y completado.\n\n');

% ── Medición en ambos extremos ──────────────────────────────────
measured_y_left = input('  Distancia REAL medida en extremo IZQUIERDO — riel Y1 (mm): ');
if isnan(measured_y_left) || measured_y_left <= 0
    error('La distancia medida debe ser un número positivo.');
end

measured_y_right = input('  Distancia REAL medida en extremo DERECHO  — riel Y2 (mm): ');
if isnan(measured_y_right) || measured_y_right <= 0
    error('La distancia medida debe ser un número positivo.');
end

% ── Verificación de sesgo entre rieles ──────────────────────────
skew_mm   = abs(measured_y_left - measured_y_right);
measured_y = (measured_y_left + measured_y_right) / 2;   % promedio para calibración

fprintf('\n  ── Verificación de sincronismo Y1/Y2 ────\n');
fprintf('  Extremo izquierdo (Y1) : %.3f mm\n', measured_y_left);
fprintf('  Extremo derecho  (Y2) : %.3f mm\n', measured_y_right);
fprintf('  Diferencia (sesgo)    : %.3f mm\n', skew_mm);

if skew_mm > 1.0
    fprintf(['  ✖ SESGO CRITICO (> 1 mm).\n' ...
             '    El carro está torcido. Causas posibles:\n' ...
             '      a) Husillos con paso diferente → revisar hardware\n' ...
             '      b) Y2_DIR_INVERTED incorrecto → cambiar en robot_arduino.ino\n' ...
             '      c) Un motor pierde pasos      → reducir PULSE_US o velocidad\n' ...
             '    ACCIÓN: NO continuar hasta resolver el sesgo.\n\n']);
    clear s;
    error('Sesgo entre rieles Y excesivo (%.2f mm). Corrige el problema mecánico y repite.', skew_mm);
elseif skew_mm > 0.5
    fprintf(['  ⚠ Sesgo moderado (0.5–1 mm).\n' ...
             '    Aceptable para pruebas, pero revisar la mecánica antes\n' ...
             '    de realizar adquisiciones definitivas.\n']);
else
    fprintf('  ✔ Rieles Y sincronizados (sesgo < 0.5 mm).\n');
end

fprintf('  Distancia promedio usada para calibración: %.3f mm\n', measured_y);

% ── Cálculo de calibración ──────────────────────────────────────
mm_per_cmd_y = measured_y / CAL_DIST_CMD;
cmd_per_mm_y = CAL_DIST_CMD / measured_y;
error_pct_y  = abs(mm_per_cmd_y - 1.0) * 100;
steps_sugg_y = STEPS_PER_MM_FIRMWARE / mm_per_cmd_y;

fprintf('\n  ── Resultado Y ──────────────────────────\n');
fprintf('  Comandos enviados  : %d\n',         CAL_DIST_CMD);
fprintf('  Distancia nominal  : %.1f mm\n',    CAL_DIST_CMD);
fprintf('  Distancia promedio : %.3f mm\n',    measured_y);
fprintf('  mm / comando       : %.5f\n',       mm_per_cmd_y);
fprintf('  Comandos / mm real : %.5f\n',       cmd_per_mm_y);
fprintf('  Error              : %.2f%%\n',     error_pct_y);
if error_pct_y > 1.0
    fprintf('  ⚠ Error > 1%%. STEPS_PER_MM sugerido: %.4f\n', steps_sugg_y);
else
    fprintf('  ✔ Calibración Y correcta (error < 1%%).\n');
end

fprintf('\n  Volviendo al origen Y...\n');
move_axis(s, 'Y-', CAL_DIST_CMD);
fprintf('  Verifica que ambos extremos del carro volvieron al punto de partida.\n');
input('  ¿El carro volvió recto al punto de partida? Pulsa ENTER: ', 's');

%% ── 5. Guardar configuración ────────────────────────────────────
calibration_date = datestr(now, 'yyyy-mm-dd HH:MM:SS');

save(config_file, ...
    'arduino_port',      ...   % puerto serie del Arduino
    'mm_per_cmd_x',      ...   % mm reales por comando  (eje X)
    'mm_per_cmd_y',      ...   % mm reales por comando  (eje Y, promedio Y1+Y2)
    'cmd_per_mm_x',      ...   % comandos por mm real   (eje X)
    'cmd_per_mm_y',      ...   % comandos por mm real   (eje Y, promedio Y1+Y2)
    'measured_y_left',   ...   % distancia medida riel Y1 (izquierdo) [mm]
    'measured_y_right',  ...   % distancia medida riel Y2 (derecho)   [mm]
    'skew_mm',           ...   % diferencia Y1−Y2 en la calibración   [mm]
    'DWELL_S',           ...   % pausa de estabilización para adquisición [s]
    'OFFSET_X_MM',       ...   % offset mecánico carro→punta en X [mm]
    'OFFSET_Y_MM',       ...   % offset mecánico carro→punta en Y [mm]
    'steps_sugg_x',      ...   % STEPS_PER_MM sugerido para firmware (eje X)
    'steps_sugg_y',      ...   % STEPS_PER_MM sugerido para firmware (eje Y)
    'calibration_date');

fprintf('\n╔══════════════════════════════════════════════════╗\n');
fprintf('║   CALIBRACIÓN COMPLETADA  ✔                      ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n');
fprintf('\nArchivo guardado:\n  %s\n\n', config_file);
fprintf('Resumen:\n');
fprintf('  Eje X :  %.5f mm/cmd  —  error %.2f%%\n', mm_per_cmd_x, error_pct_x);
fprintf('  Eje Y :  %.5f mm/cmd  —  error %.2f%%\n', mm_per_cmd_y, error_pct_y);
fprintf('  Sesgo Y1/Y2 :  %.3f mm  ', skew_mm);
if skew_mm <= 0.5
    fprintf('(✔ correcto)\n');
else
    fprintf('(⚠ revisar mecánica)\n');
end

if error_pct_x > 1.0 || error_pct_y > 1.0
    fprintf(['\n⚠  El error es mayor al 1%%.\n' ...
             '   Dos opciones para corregirlo:\n\n' ...
             '   OPCIÓN A (recomendada) — Actualizar el firmware Arduino:\n' ...
             '     1. Abrir  arduino/robot_arduino.ino\n' ...
             '     2. Cambiar la línea  #define STEPS_PER_MM  por:\n']);
    if error_pct_x > 1.0
        fprintf('        Eje X → #define STEPS_PER_MM  %.4f\n', steps_sugg_x);
    end
    if error_pct_y > 1.0
        fprintf('        Eje Y → #define STEPS_PER_MM  %.4f\n', steps_sugg_y);
    end
    fprintf(['     3. Volver a cargar el .ino en el Arduino\n' ...
             '     4. Volver a correr este script para verificar\n\n' ...
             '   OPCIÓN B — Sin recargar el firmware:\n' ...
             '     Los factores cmd_per_mm_x/y guardados en robot_config.mat\n' ...
             '     compensarán el error automáticamente en los scripts futuros.\n']);
else
    fprintf('\n✔ La calibración es precisa. No es necesario cambiar el firmware.\n');
end

%% ── 6. Cerrar conexión ──────────────────────────────────────────
clear s;
fprintf('\nConexión serie cerrada.\n');
fprintf('Próximo paso:  step2_test_movement.m  (pendiente de crear)\n');

%% ================================================================
%  FUNCIÓN LOCAL — Enviar N comandos a un eje, esperar ACK por cada uno
%  El Arduino responde "OK\n" tras completar cada movimiento de 1 mm.
%% ================================================================
function move_axis(ser, cmd, n_steps)
    for i = 1:n_steps
        writeline(ser, cmd);
        ack = readline(ser);              % bloquea hasta recibir la respuesta
        if ~strcmp(strtrim(ack), 'OK')
            warning('Respuesta inesperada del Arduino en paso %d: "%s"', i, strtrim(ack));
        end
        % Mostrar progreso cada 10 pasos
        if mod(i, 10) == 0
            fprintf('    %d / %d comandos\n', i, n_steps);
        end
    end
end
