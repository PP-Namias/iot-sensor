const int buzzerPin = 9;  // Connect buzzer to pin 9

void setup() {
  pinMode(buzzerPin, OUTPUT);
  Serial.begin(9600);     // Start serial communication at 9600 baud
  digitalWrite(buzzerPin, LOW);  // Ensure buzzer is off at startup
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == '1') {
      // Turn on buzzer
      //tone(buzzerPin, 2000);  // 2000 Hz tone
      digitalWrite(buzzerPin, HIGH);
    } 
    else if (command == '0') {
      // Turn off buzzer
      //noTone(buzzerPin);
      digitalWrite(buzzerPin, LOW);
    }
  }
}