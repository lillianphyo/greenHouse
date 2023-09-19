#include <DHT.h>
#include <Servo.h>
#include <LiquidCrystal_I2C.h>
#include<stdio.h>
#define DIMENSION 2
#define LAYER1 16
#define LAYER2 8
#define OUT_LAYER 8
double layer1_weights[LAYER1][DIMENSION] = {{ 2.95845826e+000,  3.70581235e+000},
       { 2.62680294e-151,  1.07164200e-136},
       { 1.68454723e+000,  1.48084828e-001},
       {-3.71697493e-001,  3.99229036e-001},
       {-4.13991150e+000,  1.44800993e+000},
       { 8.53281870e-001,  2.14375537e+000},
       { 4.39498957e-133, -4.57236513e-098},
       { 2.49949929e+000, -5.34739780e-001},
       { 2.05022038e+000,  4.56226909e-001},
       { 4.44919735e-165,  5.09892594e-175},
       {-1.02592115e+000,  5.62606743e-001},
       {-8.84529456e-106,  1.16763444e-137},
       { 1.16865229e+000,  2.49084092e+000},
       {-2.48378582e+000,  2.61039197e+000},
       {-1.12685780e+000,  5.92053689e-001},
       {-1.52749588e+000,  2.11379273e+000}};
double layer1_bias[] = {-0.40701188, -0.15120029, -0.28479596, -0.61686449, -0.04499721,
        -0.82102903,  0.34871346, -0.22889616, -0.49628442,  0.44222855,
        -0.98829103, -4.30817704,  0.88600191, -0.34990473, -0.26829206,
         0.0134266 };

double layer2_weights[LAYER2][LAYER1] = {{-5.41265446e-001,  5.66518419e-124,  1.09741561e+000,
        -1.45759796e-001,  2.50662851e+000, -1.85629459e+000,
         6.40856407e-177, -1.87732991e-001},
       { 3.40952564e+000, -1.25183800e-103,  1.20222100e+000,
        -8.94811723e-003,  2.71523692e+000,  8.54769739e-001,
         3.16043931e-145,  1.98162405e+000},
       {-4.34539424e+000, -5.31111317e-179,  1.71026036e+000,
        -2.34782663e-001, -5.96463650e+000, -5.53329850e-001,
         1.61220283e-111,  1.33463399e+000},
       { 4.05294487e+000,  2.19274267e-107,  1.08968116e+000,
        -3.66372688e-001,  3.90360829e+000,  6.66827908e-001,
         1.91819701e-107,  1.89632659e+000},
       { 4.58219508e+000,  2.09822413e-143,  8.13524206e-001,
        -2.85093029e-001,  2.44821652e+000, -1.40172824e+000,
        -5.35005832e-111,  1.02238004e+000},
       { 3.54417762e+000,  6.62536393e-154,  1.30494027e+000,
        -5.01975766e-002,  3.28224757e+000,  9.18389620e-001,
        -3.23359536e-105,  1.57691897e+000},
       {-4.15116210e+000, -8.30264048e-123,  1.74346549e-001,
         1.11478502e-180,  3.67993926e+000, -1.68561964e+000,
         2.38634266e-117, -2.54672856e+000},
       {-5.49907549e+000, -4.46521254e-180, -6.18023749e-001,
         1.29004049e-139, -5.95615299e+000,  2.48820761e+000,
         8.47591201e-151, -9.17654212e-002}};
double layer2_bias[] = {0.55290317, -0.4743239 , -0.23624257,  0.59785689, -0.40770618,
        -0.07411662, -0.10994495, -0.41092096};

double output_layer_weights[OUT_LAYER] = {-2.4994282 ,2.36415845,-4.71240845,2.47978599,-5.79256752,2.78942908,-3.50807105,-5.54082873};
double output_layer_bias[] = {0.38697837};
// create arrays for storing the values of outputs from each layer
double out_layer_1[LAYER1];
double out_layer_2[LAYER2];

/*
Define the activation_function activation function that returns max(0, x).
*/
void activation_function(double out_layer[], int n)
{
    int i = 0;
    while(i<n)
      {  
        out_layer[i] = out_layer[i] > 0.0 ? out_layer[i] : 0.0;
        i++;
      }
}

/*
Define the forward pass function that takes temperature and humidity as the input
for neural network and returns the result from the output layer
*/

double predict(double humidity, double temperature)
{    
    double inputs[] = {temperature, humidity};
    double result = 0.0;

    int i, j;
  //Multiplication of layer1 weights and input matrix
    i = 0;
    while(i<LAYER1){
        out_layer_1[i] = 0.0;
        j = 0;
        while(j<DIMENSION)
          {  
            out_layer_1[i] += (layer1_weights[i][j] * inputs[j]);
            j++;
          }
        i++;
    }

    //adding layer1 bias
    i = 0;
    while(i<LAYER1)
    {
        out_layer_1[i] += layer1_bias[i];
        i++;
    }
    activation_function(out_layer_1, LAYER1);
    
    //Multiplication of layer2 weights and previous output layer
    i = 0;
    while(i<LAYER2)
    {
        out_layer_2[i] = 0.0;
        j = 0;
        while(j<LAYER1)
          {  
            out_layer_2[i] += (layer2_weights[i][j] * out_layer_1[j]);
            j++;
          }
        i++;
    }
    //adding bias of layer2
    i = 0;
    while(i<LAYER2)
    {
        out_layer_2[i] += layer2_bias[i];
        i++;
    }
    activation_function(out_layer_2, LAYER2);

    i = 0; 
    while(i<OUT_LAYER)
    {
        result += (out_layer_2[i] * output_layer_weights[i]);
        i++;
    }
    //adding bias of output layer
    result += output_layer_bias[0];

    result = result > 0.0 ? result : 0.0;

    return result;
}

LiquidCrystal_I2C lcd(0x27,16,2);  // set the LCD address to 0x27 for a 16 chars and 2 line display

#define DHTPIN1 7 
#define DHTPIN2 5
#define DHTPIN3 6
#define DHTPIN4 4
const float GAMMA = 0.7;
const float RL10 = 50;

void read_temp_humid();
int water_level_in_all_motors(int [], int [], int);

DHT dht[] = {{DHTPIN1, DHT22},{DHTPIN2, DHT22},{DHTPIN3, DHT22},{DHTPIN4, DHT22},};

float humidity[4];
float temperature[4];
int water_percent[4];
int randomNumber;

Servo servo1; 
Servo servo2;
Servo servo3;
Servo servo4;

void setup()
{
  lcd.init(); // initialize the lcd
  lcd.backlight();

  Serial.begin(9600);
 
  dht[0].begin();
  dht[1].begin();
  dht[2].begin();
  dht[3].begin();

    

  servo1.attach(10);
  servo2.attach(9);
  servo3.attach(11);
  servo4.attach(8);

  lcd.clear();

  lcd.setCursor(0,0);
  lcd.print("  Submitted by  ");
  lcd.setCursor(0,1);
  lcd.print(" 211110(33_63) ");

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print(" Lets Begin the");
  lcd.setCursor(0,1);
  lcd.print("   SIMULATION   ");
}
void read_temp_humid()
{
    double min_humidity = 8.132132, max_humidity = 51.867868;
    double min_temperature = 20.000000, max_temperature = 100.000000;

  for (int i = 0; i < 4; i++) 
  {
    temperature[i] = dht[i].readTemperature();
    humidity[i] = dht[i].readHumidity();
    humidity[i]  = (humidity[i] - min_humidity)/(max_humidity - min_humidity);
    temperature[i] = (temperature[i] - min_temperature)/(max_temperature- min_temperature);
  }

}

void  water_level_in_all_motors(float temp[], float humid[])
{
      for(int i=0;i<4;i++)
      {
        double h = humid[i];
        double t = temp[i];
        double w = predict(h, t);
        water_percent[i] = w;
      }
}

void convert_water_percentage_to_angle(int pos[])
{ 
  for (int i = 0; i < 4; i++) 
  {
    pos[i] = (pos[i]*180)/100;
  } 

  servo1.write(pos[0]);          
  servo2.write(pos[1]);    
  servo3.write(pos[2]);
  servo4.write(pos[3]);                            
}

void loop()
{
    int analogValue = analogRead(A0);
    float voltage = analogValue / 1024. * 5;
    float resistance = 2000 * voltage / (1 - voltage / 5);
    float lux = pow(RL10 * 1e3 * pow(10, GAMMA) / resistance, (1 / GAMMA));
  
   if (lux >= 50 ) 
   {
      read_temp_humid();
      water_level_in_all_motors(temperature, humidity);

      for (int i = 0; i < 4; i++) 
      {
          Serial.print(F("Temperature_"));
          Serial.print(i+1);
          Serial.print("  :  ");
          Serial.println(temperature[i]);
          Serial.print(F("Humidity_"));
          Serial.print(i+1);
          Serial.print("  :  ");
          Serial.println(humidity[i]);
          Serial.print(F("Water Percentage_"));
          Serial.print(i+1);
          Serial.print("  :  ");
          Serial.print(water_percent[i]);
          Serial.print("%");
          Serial.println();
          // delay(1000);
      }
      lcd.clear();
      lcd.setCursor(0, 0);

      lcd.print(" 1. ");
      lcd.print(water_percent[0]);
      lcd.print("%  2. ");
      lcd.print(water_percent[1]);
      lcd.print("%");
      
      lcd.setCursor(0, 1);

      lcd.print(" 3. ");
      lcd.print(water_percent[2]);
      lcd.print("%  4. ");
      lcd.print(water_percent[3]);
      lcd.print("%");


      Serial.print("------------------------------\n");
      convert_water_percentage_to_angle(water_percent);
      
      // delay(2000);
  } 

  else 
  {
    // set servo motor to zero angle in all places.
    convert_water_percentage_to_angle(0);

    Serial.println("Night Time : The system is off");
    servo1.write(0);          
    servo2.write(0);          
    servo3.write(0);          
    servo4.write(0);          
      lcd.setCursor(0, 0);

      lcd.print(" 1. 0%");
      lcd.print("   2. 0%  ");
        lcd.setCursor(0, 1);

      lcd.print(" 3. 0%");
      lcd.print("   4. 0%  ");
    //print water percent on lcd
  }
  delay(1000);
}







      











