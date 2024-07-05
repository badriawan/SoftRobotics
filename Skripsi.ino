int pwm_pin = 11;
float data = 0;

void setup() {
  Serial.begin(9600);
  pinMode(pwm_pin, OUTPUT);
  pinMode(relay_pin, OUTPUT);
}

void loop() {
  //if (Serial.available())
  
      data = Serial.parseFloat();
      Serial.println("Hello from Arduino!");
      analogWrite(pwm_pin, (int) data);

}
