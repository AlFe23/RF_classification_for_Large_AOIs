# RF4LargeAOIs

# Indice

1. [Introduzione](#1-introduzione)
2. [Generazione dei dati di input [GEE script]](#2-input-data-generation-gee-script)
3. [Struttura della directory per la lettura dei file di input](#3-directory-structure-for-input-file-reading)
4. [Pre-elaborazione di normalizzazione](#4-auto-normalization-preprocessing)
    - [4.1 Utilizzo (Auto-Normalization)](#41-utilizzo-auto-normalization)
    - [4.2 Dipendenze (Auto-Normalization)](#42-dependencies-auto-normalization)
    - [4.3 Input e Formato (Auto-Normalization)](#43-input-e-formato-auto-normalization)
    - [4.4 Z-score Normalization - info](#44-z-score-normalization-info)
5. [Modulo-1](#5-modulo-1)
    - [5.1 Dipendenze (Modulo-1)](#51-dipendenze-modulo-1)
    - [5.2 Utilizzo (Modulo-1)](#52-utilizzo-modulo-1)
    - [5.3 Algoritmo (Modulo-1)](#53-algoritmo-modulo-1)
6. [Modulo-2](#6-modulo-2)
    - [6.1 Dipendenze (Modulo-2)](#61-dipendenze-modulo-2)
    - [6.2 Utilizzo (Modulo-2)](#62-utilizzo-modulo-2)
    - [6.3 Algoritmo (Modulo-2)](#63-algoritmo-modulo-2)
7. [Modulo-3](#7-modulo-3)
    - [7.1 Utilizzo (Modulo-3)](#71-utilizzo-modulo-3)
    - [7.2 Algoritmo (Modulo-3)](#72-algoritmo-modulo-3)
    - [7.3 Limitazioni e considerazioni](#73-limitazioni-e-considerazioni)
8. [Modulo-4](#8-modulo-4)
    - [8.1 Descrizione](#81-descrizione)
    - [8.2 Dipendenze](#82-dipendenze)
    - [8.3 Utilizzo](#83-utilizzo)
    - [8.4 Algoritmo](#84-algoritmo)
	
# 1. Introduzione

Questa repository contiene degli script per l'elaborazione di immagini multitemporali su aree di estensione indefinita, al fine di produrre mappe di classificazione di copertura del suolo di intere regioni coperte da un numero non specificato di immagini satellitari. 
I dati in input includono immagini GeoTIFF organizzate in sottocartelle organizzate per tile e un file CSV contenente punti di classe nota. 
Gli script sono organizzati in quattro moduli più un modulo di preprocessing per la normalizzazione. 
Il Modulo 1 esegue il pre-processing dei dati in input per generare una matrice di feature adatta per l'addestramento del modello random-forest. 
Il Modulo 2 addestra il modello random-forest, lo salva e ne valuta le prestazioni su un set di test. 
Il Modulo 3 consente agli utenti di testare l'accuratezza di un modello pre-addestrato su un diverso dataset di punti di classe nota. 
Il Modulo 4 classifica immagini multitemporali utilizzando un modello random-forest pre-addestrato e genera mappe di probabilità e classificazione in formato GeoTIFF. 
Viene inoltre fornito un modulo di preprocessing per la normalizzazione di tipo Z-Score di tutte le immagini di input che salva le immagini normalizzate con una organizzazione di directory identica a quella originale.
Gli script sono scritti in Python e utilizzano librerie come pandas, rasterio e scikit-learn.
Si prega di notare che il codice presente nel repository è attualmente in fase di sviluppo e non è ancora stato rilasciato come versione definitiva.

# 2. Generazione dei dati di input [GEE script]

Nella attuale fase prototipale la catena di elaborazione atta a produrre mosaici multitemporali di immagini Sentinel-2 L2A e Sentinel-1 GRD è implementata in linguaggio javascript sull’editor di Google Earth Engine. 
Tale scelta si deve alla rapidità con cui è possibile accedere ai dati di interesse ed esportarli.  
L’output di questa catena di elaborazione sono i mosaici Geotiff multitemporali che possano essere usati come input al classificatore random-forest, sia in addestramento che predizione.
**Il report che descrive la procedura e il codice javascript è scaricabile nella sezione Download di questa repository.

I mosaici ottenuti dalla procedura descritta in questo documento sono utilizzati come input per la classificazione di aree di estensione indefinita tramite gli algoritmi in linguaggio python implementati nei moduli descritto di seguito. 
I mosaici esportati su Google Drive sono automaticamente divisi in sub-tile (non corrispondenti alle tile originali Sentinel-2) e sono in formato Geotiff Uint16 con con i canali corrispondenti alle bande come da tabella 1.

Struttura mosaici GeoTIFF Sentinel-1/2:

| canale | Banda S2/S1 |
|--------|-------------|
| Sentinel-2 L2A |        |
| 1      | B2 (10m native) |
| 2      | B3 (10m native) |
| 3      | B4 (10m native) |
| 4      | B8 (10m native) |
| 5      | B5 10m (resampled(bicubic) 20m ->10m)  |
| 6      | B6 10m (resampled(bicubic) 20m ->10m) |
| 7      | B7 10m (resampled(bicubic) 20m ->10m) |
| 8      | B8A 10m (resampled(bicubic) 20m ->10m) |
| 9      | B11 10m (resampled(bicubic) 20m ->10m) |
| 10     | B12 10m (resampled(bicubic) 20m ->10m) |
| 11     | SCL 10m (resampled (nearest neighbor)20m ->10m)  |
| Sentinel-1 GRD |        |
| 12     | VH (10m coregistered with S2) |
| 13     | VV (10m coregistered with S2) |


# 3. Struttura della directory per la lettura dei file di input

1- Serie di immagini multitemporali. Questa versione del programma accetta in input immagini formato Geotiff organizzate come segue: cartella ‘input_images’ contenente tante sottocartelle quante sono le tiles (es. subfolders: Tile_A,…, Tile_N). All’ interno di queste sottocartelle sono presenti le diverse acquisizioni multi-temporali della stessa tile:

```
  Sentinel-2 Folder
  ├── subdirectory_1
  │   ├── image_1.tif
  │   ├── image_2.tif
  │   └── ...
  ├── subdirectory_2
  │   ├── image_1.tif
  │   ├── image_2.tif
  │   └── ...
  └── ...
```
  
2- primo file CSV contenente le coordinate dei campioni di classe nota e le classi di ogni campione del training dataset, organizzati nelle sezioni: ID, X(LAT), Y(LON), class:

```
  X,Y,class
  x1,y1,label1
  x2,y2,label2
  ...
  ```


# 4. Pre-elaborazione di normalizzazione #
Il codice Python Autonormalization.py viene utilizzato per normalizzare bande specifiche nelle immagini satellitari Sentinel-2 e Sentinel-1 e salvare le immagini normalizzate come nuovi file GeoTIFF. Attualmente lo script effettua la normalizzazione di tipo z-score sugli input organizzati come da convenzione e salva gli output con la stessa struttura di directory. Utilizza la libreria `rasterio` per la lettura e scrittura dei dati raster e la libreria `numpy` per le operazioni numeriche.
## 4.1 Utilizzo (Auto-Normalization)
1. Imposta la variabile `image_folder_path` al percorso della cartella contenente le immagini satellitari Sentinel-2 che devono essere normalizzate.
2. Imposta la variabile `output_folder_path` al percorso desiderato in cui verranno salvate le immagini normalizzate. Lo script creerà la cartella di output se non esiste già.
3. Specifica le bande che devono essere normalizzate modificando la lista `specified_bands`. La lista contiene gli indici delle bande (a partire da 0) da normalizzare.
4. Esegui lo script. Esso ciclerà su ogni sottodirectory nella cartella principale specificata da `image_folder_path` e processerà le immagini in ogni sottodirectory.
5. Lo script apre ogni immagine GeoTIFF, legge i dati dell'immagine, normalizza le bande specificate e sostituisce l'undicesima banda dell'immagine originale (SCL di Sentinel-2 L2A) mantenendo quella originale. Il processo di normalizzazione z-score calcola la media e la deviazione standard dei dati della banda, normalizza i dati sottraendo la media e dividendo per la deviazione standard, e scala i valori tra 0 e 65535.
6. Lo script copia i metadati dall'immagine originale, aggiorna i campi dei metadati necessari (ad esempio, tipo di dato, numero di bande, compressione) e aggiorna i nomi delle bande (descrizioni) nei metadati.
7. Viene creato un nuovo file GeoTIFF con i dati dell'immagine normalizzata e i metadati aggiornati. L'immagine normalizzata viene salvata nella `output_folder_path` con la stessa struttura di directory delle immagini di input. Le immagini normalizzate vengono denominate con il suffisso "_normalized" aggiunto ai nomi dei file originali.

## 4.2 Dependencies (Auto-Normalization)

Il codice richiede le seguenti dipendenze:

- Moduli: 
  - `os`
  - `glob`
- Librerie: 
  - `rasterio`
  - `numpy`


## 4.3 Input e Formato (Auto-Normalization)
Il codice si aspetta che le immagini di input siano nel formato GeoTIFF. 
La variabile `image_folder_path` deve essere impostata al percorso della cartella contenente le immagini satellitari Sentinel-2 / Sentinel-1 per cui è richiesta la normalizzazione. Si assume che la struttura delle cartelle includa sottodirectory all'interno della cartella principale, con ciascuna sottodirectory contenente i singoli file immagine. Il codice elabora in modo ricorsivo tutte le sottodirectory all'interno della cartella principale specificata.
Le immagini di input devono avere la struttura specificata nella sezione ‘Input data generation’. Il codice utilizza la libreria `rasterio` per aprire e leggere le immagini GeoTIFF. 
Le immagini normalizzate in output vengono salvate anche nel formato GeoTIFF. Vengono create nella `output_folder_path`, che deve essere impostata al percorso desiderato in cui verranno salvate le immagini normalizzate. Lo script crea automaticamente la cartella di output se non esiste già, garantendo la conservazione della stessa struttura di directory delle immagini di input.
Durante il processo di salvataggio delle immagini normalizzate, il codice aggiorna i metadati di ciascuna immagine, inclusi il tipo di dato, il numero di bande e le impostazioni di compressione. Inoltre, i nomi delle bande (descrizioni) dalle immagini originali vengono preservati nei metadati delle immagini normalizzate. Ciò consente alle applicazioni software come QGIS di riconoscere e visualizzare correttamente i nomi delle bande quando si aprono le immagini normalizzate.
## 4.4 Z-score Normalization - info

**La normalizzazione Z-score** (Z-score normalization), nota anche come standardizzazione o z-normalization, è una tecnica utilizzata nel machin learning e in statistica per uniformare l'intervallo delle variabili indipendenti o delle fetures. Può essere particolarmente utile quando i dati presentano scale diverse, ma è necessario che si trovino su una scala simile affinché l'algoritmo ML sia più efficace.
Lo Z-score di un singolo campione viene calcolato sottraendo la media dell'insieme di dati dal campione individuale e poi dividendo questa differenza per la deviazione standard dell'insieme di dati. La formula per lo Z-score è la seguente:
Z = (X - μ) / σ
Dove:
- Z è lo Z-score,
- X è il valore del singolo campione,
- μ è la media dell'insieme di dati, 
- σ è la deviazione standard dell'insieme di dati.
Dopo la normalizzazione Z-score, la distribuzione risultante ha una media di 0 e una deviazione standard di 1.

# 5. Modulo-1 #
Modulo di lettura e costruzione del dataset di addestramento.
Questo modulo effettua il pre-processing degli input al fine di generare una struttura dati idonea all'addestramento del modello random-forest, che avviene nel modulo 2. 
Prende in input un file CSV del Training Set, una cartella contenente sottocartelle di immagini multitemporali, e una cartella di Output dove i dati elaborati verranno salvati.

## 5.1 Dipendenze (modulo-1)

- Python 3.x
- Tkinter
- pandas
- numpy
- rasterio

## 5.2 Utilizzo (modulo-1)

1. Avvia il programma eseguendo lo script `modulo_1.py` in Python.
2. Apparirà la finestra del programma.
3. Seleziona il file CSV del Training Set.
4. Seleziona la cartella Sentinel-2. Questa cartella dovrebbe contenere sottocartelle con acquisizioni multitemporali per ogni tile adiacente.
5. Seleziona la cartella di Output. Qui verranno salvati i dati elaborati.
6. Seleziona le bande desiderate spuntando le caselle accanto ai nomi delle bande corrispondenti. Puoi selezionare singole bande o usare la casella "Seleziona tutto" per selezionare tutte le bande. (nota: in fase di prediction bisogna specificare le bande da usare coerentemente con la scelta qui effettuata.
7. Clicca il pulsante "Esegui" per avviare il processo di pre-processamento.
8. Il programma genererà una matrice di caratteristiche (X) e un array di etichette di classe (y) basati sulle bande selezionate e sui dati in input. Questi verranno salvati come file CSV nella cartella di Output.
9. Un file di report (`Module1_report_esecuzione.txt`) verrà anche generato nella cartella di Output, fornendo informazioni sui file di input e output, il numero di caratteristiche in X, e il tempo di esecuzione.
10. Il programma mostrerà le forme di X e y nella console per verificare che abbiano le dimensioni corrette.

**Nota:**
- Il file CSV in input dovrebbe contenere colonne per le coordinate geografiche (X, Y) e le etichette di classe (class). Il formato atteso è il seguente:
  ```
  X,Y,class
  x1,y1,label1
  x2,y2,label2
  ...
  ```
## 5.3 Algoritmo (modulo-1)
Lo script prende in input la directory contenente tutte le sottocartelle di immagini e il file CSV e esegue i seguenti passaggi:

- Lo script carica un file CSV contenente coordinate geografiche (latitudine e longitudine) e la class label, in un dataframe di pandas.
- Estrae i valori di latitudine e longitudine dal dataframe e li memorizza in array numpy separati.
- Lo script itera su ogni sotto-directory nella directory input_images e controlla se contiene file .tif. Apre il primo file ed estrae l'estensione geografica (cioè i valori di latitudine e longitudine minimi e massimi).
- Filtra le righe del file CSV per includere solo quelle che rientrano nell'estensione geografica della sotto-directory corrente.
- Lo script itera su ogni file .tif nella sotto-directory corrente ed estrae i valori di pixel alle coordinate geografiche specificate dalle righe CSV filtrate. Questi valori di pixel vengono memorizzati in una lista.
- Sfruttando la maschera `SCL` lo script esclude tutti quei punti di training per cui in almeno un immagine è presente pixel nuvoloso in corrispondenza del punto di training.
- Dopo aver elaborato tutti i file .tif nella sotto-directory corrente, lo script concatena la lista di array di valori di pixel in un singolo array numpy e lo aggiunge a una lista di array di valori di pixel per tutte le sotto-directory.
- Una volta elaborate tutte le sotto-directory, lo script concatena la lista di array di valori di pixel in un singolo array numpy (X) che verrà utilizzato come matrice delle caratteristiche per addestrare un modello di apprendimento automatico.
- Lo script combina l'array 2D delle features con il vettore di labels corrispondenti (y) per creare un set di dati di addestramento.
- Infine, lo script verifica che X e y abbiano lo stesso numero di righe, indicando che tutte le coordinate geografiche nel file CSV sono state correttamente abbinate ai valori di pixel nei file .tif.
- X e Y vengono salvati in CSV.






# 6. Modulo-2 #

Questo modulo è progettato per addestrare un classificatore Random Forest (RF) utilizzando l’array delle features X e l'array delle labels di classe (y) generato dal Modulo-1. Prende in input i file CSV di X e y, e restituisce un RF addestrato insieme alle metriche di performance.

## 6.1 Dipendenze (Modulo-2)
- Python 3.x
- Tkinter
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

## 6.2 Utilizzo (Modulo-2)

1. Avvia il programma eseguendo lo script `modulo_2.py` in Python. Apparirà la finestra del programma.
3. Seleziona il file CSV delle features di input X. Questo file dovrebbe contenere la matrice delle features X generata dal Modulo-1.
4. Seleziona il file CSV delle classi di output y. Questo file dovrebbe contenere le labels di classe (y) corrispondenti ai campioni nelle features di input X.
5. Seleziona la cartella di Output. Qui verranno salvati il modello addestrato e il rapporto di esecuzione.
6. Scegli gli iperparametri desiderati per il classificatore di foresta casuale selezionando i valori dai menu a tendina.
7. Clicca sul pulsante "Run" per iniziare il processo di addestramento.
8. Il programma addestrerà un classificatore RF utilizzando gli iperparametri selezionati sui dati X e y forniti.
9. Il modello addestrato verrà salvato come `rf_model.joblib` nella cartella di Output.
10. Il programma valuterà le prestazioni del modello addestrato sul set di test e calcolerà diverse metriche di classificazione comuni, tra cui precisione, ricall, punteggio F1 e una matrice di confusione.
11. Le metriche di performance verranno stampate nella console.
12. Il programma genererà un rapporto di esecuzione (`Module2_report_esecuzione.txt`) nella cartella di Output, fornendo informazioni sul tempo di esecuzione, file di input e output, iperparametri del classificatore, rapporto di classificazione e matrice di confusione.

**Nota:** I file CSV di X e y dovrebbero essere generati utilizzando il Modulo-1 prima di eseguire il Modulo-2. Assicurati che i file X e y siano correttamente pre-elaborati e rispettino i requisiti di input del Modulo-2.


## 6.3 Algoritmo (modulo-2):

- In primo luogo, vengono specificati i percorsi dei file per il dataset, e la matrice delle caratteristiche X e l'array dei labels di classe vengono letti dai file CSV ( X e y) in Pandas DataFrames. Questi DataFrames vengono quindi convertiti in array numpy.
- Successivamente, il dataset viene suddiviso in set di addestramento e di test utilizzando la funzione train_test_split di Scikit-learn. La suddivisione è fatta in modo tale che il 20% dei dati sia utilizzato per il test.
- Viene quindi addestrato un classificatore random-forest sul set di addestramento utilizzando la classe RandomForestClassifier di Scikit-learn. Vengono utilizzati gli iperparametri predefiniti per il classificatore.
- Il modello addestrato viene quindi salvato utilizzando la funzione joblib.dump della libreria joblib.
- Le prestazioni del modello addestrato vengono valutate sul set di test calcolando diverse metriche comuni di classificazione, tra cui l'accuratezza, precision e recall, ed F1-score. 
- Infine, le metriche calcolate vengono stampate sulla console, insieme a un rapporto delle metriche di accuratezza calcolate.


# 7. Modulo-3

Questo modulo è utilizzato per valutare l'accuratezza di un modello RF pre-addestrato su un diverso insieme di dati di coordinate di classi note. Il modulo prende in input due file CSV per X e y del dataset da testare.

## 7.1 Utilizzo

1. Assicurati di avere un modello pre-addestrato RF (`rf_model.joblib`) disponibile nel percorso della cartella principale, oltre che i dataset da testare i formato csv e nominati `X´ e ´y´.
2. Avvia il programma eseguendo lo script `modulo_3.py` in Python.
3. Il programma caricherà il modello pre-addestrato dal percorso della cartella principale.
4. Il programma leggerà quindi i file CSV di X e y dai percorsi specificati.
5. La matrice delle features (X) e l'array delle labels di classe (y) verranno estratti dai file CSV.
6. Il programma dividerà l'insieme di dati in insiemi di addestramento e di test.
7. Il modello addestrato verrà utilizzato per fare predizioni sui dati di test.
8. Il programma calcolerà diverse metriche di classificazione comuni, tra cui precisione, richiamo e punteggio F1.
9. Le metriche calcolate verranno stampate nella console.
10. Verrà stampato anche un rapporto di classificazione, fornendo informazioni dettagliate sulle prestazioni del modello per ogni classe.
11. La valutazione del modello sul dataset di test aiuterà a valutare la sua accuratezza e prestazioni nel prevedere le labels di classe su dati nuovi e non visti.

## 7.2 Algoritmo
Questo modulo permette di testare l’accuratezza di un modello addestrato su un altro dataset di coordinate di classe nota, 
diverso da quello usato per l’addestramento e validazione. Quindi gli input sono due file CSV per X e Y del dataset da testare. 
Si tenga a mente che mentre la struttura di Y sarà di n righe per n pixel e 2 colonne (id pixel, classe), per quanto riguarda 
l’input X, per applicare il metodo evaluate()  che fa predizione sul test dataset, è necessario che quest’ultimo contenga le 
riflettanze da un numero di bande pari a quelle usate per l’addestramento. Ad esempio, se il dataset usato per l’addestramento 
conteneva 4 acquisizioni multitemporali da 4 bande ognuna, la struttura di X sarà uguale a n x 16, dove ‘n’ è il numero di pixel 
di classe nota e 16 è il numero di valori di riflettanza, quindi il test dataset dovrà avere la stessa struttura. Si nota che il 
test dataset può essere preprocessato con il modulo di normalizzazione e con il Modulo 1 allo stesso modo di quanto fatto per il dataset di addestramento e 
validazione. Questo modulo fornisce la capacità agli utenti di valutare l’accuratezza di un modello pre-addestrato al loro caso 
d´uso.


## 7.3 Limitazioni e considerazioni

Quando si applica un modello RF pre-addestrato a un diverso insieme di immagini o a un dataset di test con valori di riflettanza derivati da immagini o punti diversi da quelli utilizzati per l'addestramento, ci sono diverse limitazioni da considerare:

- Se il nuovo insieme di dati consiste in immagini con caratteristiche diverse, come diverse condizioni di illuminazione, risoluzione spaziale o stagioni di acquisizione, le prestazioni del modello addestrato possono diminuire, possibilmente in modo significativo.
- Se il modello RF è addestrato sulla copertura del suolo di una regione specifica (ad es. Italia), potrebbe non essere in grado di generalizzare la previsione degli stessi tipi di copertura del suolo in regioni diverse aventi caratteristiche diverse.
- Se durante l'addestramento si è verificato l'overfitting, il modello avrà ulteriori difficoltà a generalizzare le classi in altre regioni.

# 8. Modulo-4 

## 8.1 Descrizione

Il Modulo-4 è progettato per eseguire la classificazione di un insieme di immagini multi-temporali utilizzando un modello random forest pre-addestrato. Accetta immagini GeoTIFF come input. Le immagini multi-temporali organizzate all'interno di sotto-cartelle che rappresentano delle tile specifiche. Queste sotto-cartelle sono contenute all'interno di una cartella principale, e il percorso di input di questa cartella principale deve essere specificato. Per ogni sotto-cartella, il modulo genera mappe di classificazione e mappe di probabilità di classe, che vengono salvate come file GeoTIFF.

La mappa di classificazione è un'immagine a 8 bit con un singolo canale, in cui ogni valore di pixel corrisponde alla classe più probabile. Le mappe di probabilità di classe sono invece immagini multi-canale, con il numero di canali uguale al numero di classi. I valori dei pixel in ogni canale rappresentano la probabilità di appartenenza alla classe corrispondente a quel canale.

## 8.2 Dipendenze

Il Modulo-4 richiede l'installazione delle seguenti dipendenze:
- `os`
- `joblib`
- `pandas`
- `numpy`
- `rasterio`
- `sklearn`
- `glob`
- `tqdm`
- `gc`
- `time`

## 8.3 Utilizzo

L'interfaccia grafica del Modulo-4 consente di eseguire la classificazione di immagini multi-temporali utilizzando un modello pre-addestrato. Di seguito sono riportati i passaggi per utilizzare l'interfaccia grafica:

1. Selezionare la cartella che contiene le immagini da classificare.
2. Slezionare il file del modello pre-addestrato da utilizzare per la classificazione.
3. Selezionare la cartelladi output in cui verranno salvate le mappe di classificazione e di probabilità.
4. Specificare il valore di "batch_size": scegliere un valore di "batch_size" dal menu a tendina. Questo valore determina la dimensione dei batch di dati che vengono elaborati contemporaneamente durante la classificazione. È possibile selezionare un valore predefinito o specificare un valore personalizzato.
5. Specificare le bande desiderate: nel campo "Desired bands" è possibile specificare le bande desiderate per la classificazione. Le bande devono essere separate da virgole (ad esempio: 0,1,2). Per impostazione predefinita, vengono selezionate tutte le bande Sentinel-2 escludendo 'SCL', oltre a VV e VH da Sentinel-1. (deve essere mantenuta la coerenza con le features usate per addestrare il modello nel modulo-2)
6. Specificare la soglia di probabilità (binary mask): nel campo "Probability Threshold" è possibile specificare la soglia di probabilità per generare una maschera binaria. I pixel con una probabilità massima di classe superiore alla soglia verranno considerati per la generazione della maschera, mentre i pixel con una probabilità inferiore verranno scartati.
7. Fare clic sul pulsante "Run" per avviare la classificazione. Verrà visualizzato il tempo di esecuzione e le mappe di classificazione e di probabilità saranno salvate nella cartella di output specificata.



## 8.4 Algoritmo
I principali passaggi dell'algoritmo sono descritti di seguito:

1. Si definiscono i percorsi di input per la cartella principale che contiene tutte le sotto-cartelle, le posizioni delle immagini di input e il file del modello pre-addestrato.
2. Si carica il modello pre-addestrato utilizzando la funzione `joblib.load()` dalla libreria `sklearn`.
3. Si itera su ogni sotto-cartella all'interno della cartella principale utilizzando un ciclo. Ogni sotto-cartella contiene acquisizioni multi-temporali della stessa tile. Per ogni sotto-cartella, il programma crea una lista di percorsi delle immagini all'interno della sotto-cartella.
4. Si legge ogni immagine di input utilizzando la funzione `rasterio.open()` e si ridimensiona in una matrice 2D, in cui ogni riga corrisponde a un pixel e ogni colonna corrisponde a una banda spettrale.
5. Si concatenano le matrici 2D di tutte le immagini di input all'interno della sotto-cartella in una singola matrice 2D, aggiungendo n colonne per ogni immagine con n bande aggiuntive. A questo punto, l'array 2D ha una dimensione di n_pixel x (m_immagini x k_bande), in cui n_pixel è il numero di pixel, m_immagini è il numero di immagini di input e k_bande è il numero di bande spettrali.
6. Si effettuano le predizioni delle probabilità di classe per i dati di input in batch paralleli utilizzando il modello pre-addestrato. Viene inizializzata una matrice vuota `y_predict_proba` con dimensioni (numero di pixel, numero di classi) per memorizzare le probabilità di classe previste. Il ciclo itera sui dati di input in batch paralleli di dimensione `batch_size` utilizzando la funzione `joblib.Parallel()`, con il numero di processi impostato su `n_cores`. Ogni batch di dati viene elaborato utilizzando una funzione di supporto `process_batch()` che applica il modello pre-addestrato ai dati di input e restituisce le probabilità di classe previste. I risultati vengono concatenati in una matrice `y_batch`, che viene assegnata alla porzione corrispondente di `y_predict_proba`. Infine, le probabilità di classe previste per tutti i dati di input vengono restituite come matrice `y_predict_proba`.
7. Si convertono le probabilità di classe per ogni pixel in label di classe, trovando l'indice della probabilità più alta e mappandolo alla corrispondente classe utilizzando `clf.classes_`. L'output è un vettore contenente la classe più probabile per ogni pixel. Questo vettore viene quindi ridimensionato in una matrice 2D con le stesse dimensioni delle immagini di input.
8. Si ridimensiona l'array 2D delle probabilità di classe previste (`y_predict_proba`) in una matrice 3D con dimensioni (n_classi, righe, colonne), in cui righe e colonne rappresentano il numero di righe e colonne della tile processata, mentre n_classi è il numero di classi nel modello pre-addestrato. Ogni colonna dell'array 2D `y_predict_proba`, che rappresenta la probabilità di appartenenza ad una specifica classe per ogni pixel, viene ridimensionata nella forma dell'immagine di input. Questo viene fatto tramite un ciclo `for` che viene ripetuto per ogni colonna (o classe) e l'array ridimensionato viene aggiunto come nuovo canale all'array 3D `y_predict_proba_reshaped`.
9. Si recuperano i metadati dell'immagine di input utilizzando `rasterio.open()` e si aggiornano per riflettere le mappe di probabilità e di classificazione (numero di bande, tipo di dati e valore nodata). La variabile `profile_proba` contiene i metadati aggiornati per la mappa di probabilità, mentre `profile_class` contiene i metadati aggiornati per la mappa di classe.
10. Si generano i nomi dei file di output per le mappe di probabilità e di classe predette utilizzando la funzione `os.path.join()` e si scrivono su file GeoTIFF separati utilizzando la funzione `rasterio.open()`. La mappa di probabilità viene scritta come un file GeoTIFF multibanda, in cui ogni banda corrisponde a una classe. La mappa di classe viene scritta come un file GeoTIFF a singola banda, in cui il valore dei pixel rappresenta la classe predetta per quel pixel.
11. Si libera la memoria eliminando le variabili non necessarie e chiamando `gc.collect()`.



