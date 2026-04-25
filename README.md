# cicids_2017_intrusion
ML-based intrusion detection system using cicids-2017 dataset.

This project is based on the paper "CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection" 

Kurniabudi, D. Stiawan, Darmawijoyo, M. Y. Bin Idris, A. M. Bamhdi and R. Budiarto, "CICIDS-2017 Dataset Feature Analysis With Information Gain for Anomaly Detection," in IEEE Access, vol. 8, pp. 132911-132921, 2020, doi: 10.1109/ACCESS.2020.3009843.

#overview
To detect intrusion,we train and test the network traffic data with two classifiers ,random forest and support vector machine and evaluate the results.

#pipeline
1.load the dataset
2.preprocess the data
3.feature selection
4.splitting of data for train/test
5.Training the classifiers
6.evaluate the results


#requirements
python 3.8+ , requirements.txt file 

#setup
1.clone the repositorary
git clone https://github.com/anusham2512/cicids_2017_intrusion

2. create virtual enviroment
   python -m venv venv
   source venv/bin/activate

3.install dependencies
  pip install -r requirements.txt

4.dataset availability
  download from https://www.unb.ca/cic/datasets/ids-2017.html which is offical source.


#execution
python simulation.py 

