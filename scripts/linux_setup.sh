#!/bin/bash 

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


if [ -d venv/ ]
then
echo "Found a virtual environment" 
source venv/bin/activate
else 
echo "Creating a virtual environment"
#Simple dependency checker that will apt-get stuff if something is missing
# sudo apt-get install python3-venv python3-pip
SYSTEM_DEPENDENCIES="python3-venv python3-pip zip libhdf5-dev"

for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then

  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."

  #If this is uncommented then only packages that are missing will get prompted..
  #sudo apt-get --yes install $REQUIRED_PKG

  #if this is uncommented then if one package is missing then all missing packages are immediately installed..
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done
#------------------------------------------------------------------------------
python3 -m venv venv
source venv/bin/activate
fi 


#git clone https://github.com/deepseek-ai/DeepSeek-VL2
#cd DeepSeek-VL2
#python3 -m venv venv
#source venv/bin/activate


#Make sure pip is up to date
python3 -m pip install --upgrade pip

python3 -m pip install -e .
python3 -m pip install -e .[gradio]

python3 -m pip install joblib wheel
python3 -m pip install flash-attn --no-build-isolation
python3 -m pip install xformers
python3 -m pip install --upgrade gradio

#You can now run using :
#CUDA_VISIBLE_DEVICES=2 python3 web_demo.py --model_name "deepseek-ai/deepseek-vl2-tiny"  --port 8080

echo "From now on you can run the web demo using: "
DEMO_DIR=`pwd`
echo "cd $DEMO_DIR"
echo "source venv/bin/activate"
echo "python3 web_demo.py --model_name \"deepseek-ai/deepseek-vl2-tiny\" --port 8080"


exit 0

