# Change-Detection
OSCD DATASET
Le reti utilizzate si trovano nella cartella "data" e sono contenute nei file "FCSNN.py" e "MyUNet.py".
Le prestazioni ottenute sono visualizzabili sotto formato .csv (visualizzando per ogni epoca numero di epoca, loss sul train, loss sul test ecc.). 
La rete siamese da me creata si assesta su un 50.5% di f1-score dopo un sufficiente numero di epoche.
La Unet ottiene fino al 59.5% di f1-score. 
Questi risultati ci sono con inizializzazione dei pesi come nel paper, cambiando quest'ultima si hanno risultati sull'f1-score molto inferiori (dovuti ad una precision molto bassa); una delle task che voglio continuare ad esplorare è proprio l'inizializzazione dei pesi per fare in modo che sia bilanciata ma che non risulti in una precision molto bassa. 
Nella cartella data è possibile trovare tutto il materiale utilizzato (tranne il dataset stesso e i salvataggi del modello durante le epoche). 
Sono inoltre presenti all'interno della cartella data alcuni salvataggi delle prediction effettuate dalle reti sul test-set e le reti originali implementate dal paper.

Altre task che voglio esplorare sono separare opportunamente train,test e validation in modo che ognuno di questi sia "significativo" quanto gli altri.
