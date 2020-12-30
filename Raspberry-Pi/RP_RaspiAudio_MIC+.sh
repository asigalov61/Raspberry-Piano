#!/bin/bash

# Raspbery Piano[TM] Bash Script 1.0 for Raspbery Pi
# Project Los Angeles
# Tegridy Code 2020

cd /home/pi/Raspberry_Piano

#init led & button
echo 25 >/sys/class/gpio/unexport
echo 25 >/sys/class/gpio/export
echo out >/sys/class/gpio/gpio25/direction
echo 23 >/sys/class/gpio/export
echo in >/sys/class/gpio/gpio23/direction

#Set volume
#amixer set Micro 50%
amixer set Master 80%
sudo alsactl store
sleep 3

##Chime prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
sleep 1

##Welcome prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Hello_Listener_Welcome_Back.wav
sleep 1

echo "--------Main Outer Cycle-----------"
while [ 1 ]
do

#led ON
echo 1 >/sys/class/gpio/gpio25/value
sleep 1

##Chime prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
sleep 1

##Lets Make Music prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Lets_Make_Music.wav
sleep 1

##Play Back Last prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Play_Back_Last.wav
sleep 1

echo "---------Inner Main cycle------"
while [ 1 ]
do

#waiting button pressed
while [ `cat /sys/class/gpio/gpio23/value` = 1 ]; do
set i = 1
done
echo  "------Please press the on Yellow button to listen to your composition"

#led ON
echo 1 >/sys/class/gpio/gpio25/value
sleep 1

#Set volume
amixer set Micro 50%
amixer set Master 96%
sudo alsactl store
sleep 1

#play last record

#root issues/run as root by executing this hack
#sudo sed -i 's/geteuid/getppid/' /usr/bin/vlc
#sudo /usr/bin/vlc -I "dummy" --play-and-exit ./output.mid
aplay /home/pi/Raspberry_Piano/output.wav

sleep 5

#Set volume
amixer set Micro 50%
amixer set Master 80%
sudo alsactl store
sleep 1

echo 0 >/sys/class/gpio/gpio25/value
sleep 1
echo 1 >/sys/class/gpio/gpio25/value
sleep 1

##Chime prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
sleep 1

##Generate Long Press prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Generate_Long_Press.wav
sleep 1

#waiting button pressed for 5 secons (delayed press)
while [ `cat /sys/class/gpio/gpio23/value` = 1 ]; do
sleep 5
while [ `cat /sys/class/gpio/gpio23/value` = 1 ]; do
sleep 5
set i = 1
done
set i = 1
done
echo  "------Please press the on Yellow button to listen to your composition"

#led OFF
echo 0 >/sys/class/gpio/gpio25/value
sleep 1

##Chime prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
sleep 1

#Remove last composition
rm /home/pi/Raspberry_Piano/output.mid
sleep 1

echo 1 >/sys/class/gpio/gpio25/value
sleep 1
echo 0 >/sys/class/gpio/gpio25/value
sleep 1

##Nice Make Music prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Nice_Make_Music.wav
sleep 1

#led BLINK
echo 1 >/sys/class/gpio/gpio25/value
sleep 1
echo 0 >/sys/class/gpio/gpio25/value
sleep 1
echo 1 >/sys/class/gpio/gpio25/value
sleep 1
echo 0 >/sys/class/gpio/gpio25/value
sleep 1

#Generate
cd /home/pi/Raspberry_Piano/
python3 ./RP_Generator.py
sleep 120

echo 1 >/sys/class/gpio/gpio25/value
sleep 1
echo 0 >/sys/class/gpio/gpio25/value
sleep 1
echo 1 >/sys/class/gpio/gpio25/value
sleep 1
echo 0 >/sys/class/gpio/gpio25/value
sleep 1

##Chime prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
sleep 1

##Composition Ready prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Composition_Ready.wav
sleep 1

##Play New prompt
aplay /home/pi/Raspberry_Piano/Voice_Prompts/Play_New.wav
sleep 1

echo 1 >/sys/class/gpio/gpio25/value
sleep 1

#waiting button pressed
#while [ `cat /sys/class/gpio/gpio23/value` = 1 ]; do
#set i = 1
#done

#echo 1 >/sys/class/gpio/gpio25/value
#sleep 1
#echo 0 >/sys/class/gpio/gpio25/value
#sleep 1

##Chime prompt
#aplay /home/pi/Raspberry_Piano/Voice_Prompts/Computer_Magic-Microsift-1901299923.wav
#sleep 1

echo "------Playing the output MIDI file..."

#echo 1 >/sys/class/gpio/gpio25/value
#sleep 1

#play record
#timidity /home/pi/Raspberry_Piano/output.mid
#sleep 1

#echo 0 >/sys/class/gpio/gpio25/value
#sleep 1

done
echo "------------------------------------------------------------------------"
done
exit

done
echo 25 >/sys/class/gpio/unexport