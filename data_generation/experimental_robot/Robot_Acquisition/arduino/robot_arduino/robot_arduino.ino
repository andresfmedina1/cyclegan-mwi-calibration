/* ═════════════════════════════════════════════════════════════════
   robot_arduino.ino
   Firmware para robot XY de adquisición automática de targets
   POLITO — Microwave Brain Imaging
   ─────────────────────────────────────────────────────────────────
   Hardware : Arduino UNO + 3 × driver M422 microstep

   CONFIGURACIÓN MECÁNICA (3 motores):
     Motor X   — eje cruzado entre los dos paralelos, lleva el target
     Motor Y1  — eje paralelo izquierdo  ┐ deben moverse
     Motor Y2  — eje paralelo derecho    ┘ perfectamente sincronizados

   SINCRONIZACIÓN Y1/Y2:
     En moveY(), ambos motores reciben su pulso en el MISMO ciclo
     del bucle → sin deriva de posición entre los dos ejes paralelos.

   PROTOCOLO (USB a 9600 baud, terminado en '\n'):
     MATLAB envía  →  "X+\n" | "X-\n" | "Y+\n" | "Y-\n"
     Arduino ejecuta 1 mm y responde  →  "OK\n"
     MATLAB no envía el siguiente comando hasta recibir "OK".

   ── CABLEADO M422 (señal single-ended) ─────────────────────────
     M422     │ Arduino UNO
     ─────────┼──────────────────────────────────────────────────
     PUL+     │ Pin de step  (ver mapa de pines abajo)
     PUL-     │ GND
     DIR+     │ Pin de dirección
     DIR-     │ GND
     ENA+     │ No conectar  (driver habilitado por defecto)
     ENA-     │ GND
     VCC      │ Fuente externa 9–30 V  (NO el 5 V del Arduino)
     GND      │ GND de la fuente externa

   ── CÁLCULO DE STEPS_PER_MM ────────────────────────────────────
     Fórmula: (pasos/rev × microstepping) / paso_husillo_mm
     Con M422 en 32 microsteps y husillo 12 mm:
       (200 × 32) / 12 = 533.33  → valor por defecto aquí
     Switches del M422 para 32 microsteps: S1=OFF S2=OFF S3=OFF
     Verificar con step1_calibrate_robot.m en MATLAB.
   ═════════════════════════════════════════════════════════════════ */

// ── MAPA DE PINES ─────────────────────────────────────────────
const int STEP_X  = 5,  DIR_X  = 4;   // Motor X  — eje cruzado (target)
const int STEP_Y1 = 3,  DIR_Y1 = 2;   // Motor Y1 — paralelo izquierdo
const int STEP_Y2 = 7,  DIR_Y2 = 6;   // Motor Y2 — paralelo derecho

// ── DIRECCIÓN DE Y2 ───────────────────────────────────────────
// Los dos motores paralelos suelen estar montados en espejo:
// para que el carro se mueva recto, uno de los dos necesita
// la señal DIR invertida respecto al otro.
//
// Probar primero con FALSE. Si al enviar "Y+" el carro se tuerce
// o los motores luchan entre sí, cambiar a TRUE y recargar.
#define Y2_DIR_INVERTED  false

// ── CALIBRACIÓN ───────────────────────────────────────────────
// 32 microsteps (S1=OFF S2=OFF S3=OFF en M422) + husillo T6×1 (1 mm/vuelta)
// Fórmula: (200 pasos/rev × 32 microsteps) / 1 mm = 6400 pasos/mm
// Verificar con step1_calibrate_robot.m; ajustar si es necesario
#define STEPS_PER_MM   6400.0

// Velocidad: µs por semiciclo de pulso
//   Velocidad = 1 / (STEPS_PER_MM × 2 × PULSE_US×1e-6)  [mm/s]
//   200 µs →  0.4 mm/s  (muy lento, para prueba inicial)
//    50 µs →  1.6 mm/s  (buen punto de partida)
//    25 µs →  3.1 mm/s  (rápido, verificar que no pierda pasos)
//    15 µs →  5.2 mm/s  (máximo recomendado para motor 28 mm)
// Si el motor hace ruido raro o no avanza: aumentar este valor.
#define PULSE_US        25

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  pinMode(STEP_X,  OUTPUT);  pinMode(DIR_X,  OUTPUT);
  pinMode(STEP_Y1, OUTPUT);  pinMode(DIR_Y1, OUTPUT);
  pinMode(STEP_Y2, OUTPUT);  pinMode(DIR_Y2, OUTPUT);
  digitalWrite(DIR_X,  LOW);
  digitalWrite(DIR_Y1, LOW);
  digitalWrite(DIR_Y2, LOW);
}

void loop() {
  static char line[16];
  static byte idx = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line[idx] = '\0';
      idx = 0;
      handleCmd(line);
    } else if (idx < sizeof(line) - 1) {
      line[idx++] = c;
    }
  }
}

/* ── Dispatcher de comandos ──────────────────────────────────── */
void handleCmd(const char *cmd) {
  if      (!strcmp(cmd, "X+")) { digitalWrite(DIR_X, HIGH);  moveX(); }
  else if (!strcmp(cmd, "X-")) { digitalWrite(DIR_X, LOW);   moveX(); }
  else if (!strcmp(cmd, "Y+")) { setDirY(true);   moveY(); }
  else if (!strcmp(cmd, "Y-")) { setDirY(false);  moveY(); }
  else    { return; }   // comando desconocido → ignorar

  Serial.println("OK");   // confirma al PC que el movimiento terminó
}

/* ── Establece la dirección de Y1 e Y2 ──────────────────────── */
void setDirY(bool positive) {
  bool dir2 = Y2_DIR_INVERTED ? !positive : positive;
  digitalWrite(DIR_Y1, positive ? HIGH : LOW);
  digitalWrite(DIR_Y2, dir2     ? HIGH : LOW);
}

/* ── Mueve el eje X exactamente 1 mm ────────────────────────── */
void moveX() {
  long steps = (long)round(STEPS_PER_MM);
  for (long i = 0; i < steps; i++) {
    digitalWrite(STEP_X, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(STEP_X, LOW);
    delayMicroseconds(PULSE_US);
  }
}

/* ── Mueve Y1 e Y2 sincronizados: 1 mm en el mismo bucle ────── */
void moveY() {
  long steps = (long)round(STEPS_PER_MM);
  for (long i = 0; i < steps; i++) {
    // Ambos motores reciben el pulso en el mismo ciclo → sincronización perfecta
    digitalWrite(STEP_Y1, HIGH);
    digitalWrite(STEP_Y2, HIGH);
    delayMicroseconds(PULSE_US);
    digitalWrite(STEP_Y1, LOW);
    digitalWrite(STEP_Y2, LOW);
    delayMicroseconds(PULSE_US);
  }
}
