# biz_stats_ml

engelsk versjon:

- [English](./README.md)

**Dataene som vises her er endret, og de reflekterer ikke faktisk industriell økonomisk utvikling, og heller ikke virkelige selskapsdata. Dette er kun for illustrasjonsformål.**

## Oversikt:

Hovedproblemet som løses, er unøyaktige/lavkvalitets svar levert av respondenter til en undersøkelse - ofte med svært materielle konsekvenser for det endelige resultatet. Dette tar normalt et team av statistikere et helt år å korrigere (noen ganger resulterer det i at man må kontakte respondentene på nytt) - Jeg skal løse denne oppgaven ved hjelp av maskinlæring og andre statistiske tiltak.

Resultater: En full produksjonskjøring (normalt fullført av et team på 5-7 personer over et helt år) fullført på 600,38 sekunder. Resultatene passerer flere logiske tester og sammenlignet med tidligere produksjoner viser de seg å være svært gunstige. R^2 når man sammenligner det dette programmet produserer mot det som faktisk ble publisert var omtrent 98% med en gjennomsnittlig absolutt feil på omtrent 5.000 NOK - noe som er lavt gitt egenskapene til våre data.

Føl deg fri til å klone repoet hvis du har passende tilgang. Jeg vil også demonstrere hva koden gjør her i denne ReadMe-filen:

## Visualiseringer:

Flere visualiseringer brukes til å analysere dataene på industrinivå. Plottene er interaktive, brukeren kan velge år, diagramtyper, fokusvariabler osv. Alle de vanlige interaktive verktøyene fra Plotly er også tilgjengelige. Noen visualiseringer er animerte, og hvis brukeren trykker på spill av, vil de se endringer over tid. Her er hvordan noen av resultatene ser ut (som naturligvis vil justere seg hvis noe annet ble valgt i rullegardinmenyene).

**Enkle plot:** 

<img width="990" alt="Line Plots" src="https://github.com/user-attachments/assets/18b5c77b-ee64-4d4a-b79d-7230be50b016"><br><br>




**Stolpediagrammer og Varmekart:**

<img width="457" alt="Bar Chart and Heat Map" src="https://github.com/user-attachments/assets/3a561afc-0d51-4666-b6a2-ea2860cb0200"><br><br>



**Kart (ett som er animert):**


<img width="422" alt="static map" src="https://github.com/user-attachments/assets/6eff24d7-ef9c-43ab-89d1-bf38d0bf87cc">
<img width="485" alt="image" src="https://github.com/user-attachments/assets/181b1f61-79f5-4ae8-87a8-59c7a31b97c5"><br><br>

**Histogram med kumulativ prosent:**


<img width="391" alt="histogram" src="https://github.com/user-attachments/assets/35454eef-b4c5-4ae3-9168-30265510f57d"><br><br>


**Koblede plot:**

<img width="394" alt="Linked Plots" src="https://github.com/user-attachments/assets/0f85e171-6b63-493d-9bf9-5fc61dd7c7b4"><br><br>

**Boblediagrammer:**

<img width="445" alt="Bubble Plot" src="https://github.com/user-attachments/assets/4e29d2d4-1055-48ce-9d4b-9ff72cdc1e3b"><br><br>

**Parallellkoordinatdiagram:**

<img width="845" alt="Parallel Cooridinates" src="https://github.com/user-attachments/assets/2158cb5e-7edf-4262-87a4-b1f5015adff0"><br><br>

**Geografisk diagram:**

<img width="851" alt="geographic" src="https://github.com/user-attachments/assets/b367bfe7-9cf0-4874-90ff-3a754b7939eb"><br><br>

**Animert stolpediagram:**


<img width="662" alt="animated bar chat" src="https://github.com/user-attachments/assets/962a2958-4929-4eb2-aa9e-1c79bd7d9d1f"><br><br>

**3D Plot:**

<img width="532" alt="3D" src="https://github.com/user-attachments/assets/b8ae22ca-7f17-421b-bde5-28f379e0edad"><br><br>


## Evaluering av maskinlæring

Dette programmet har som mål å løse problemet med lavkvalitets svar på økonomiske dataundersøkelser. Vi evaluerer kvaliteten ved å sammenligne svarene med skattetatens data og hvor mange av feltene som er utfylt. Svar av dårlig kvalitet blir imputert ved hjelp av en valgt maskinlæringsalgoritme som er trent på hele datasettet (utenom undersøkelsene av dårlig kvalitet).

**Viktige verktøy brukt:**

**Feature engineering:** Jeg samlet ekstra data ved å spørre ulike API-er og samarbeide med flere andre avdelinger innen SSB. Jeg brukte også verktøy som KNN-imputasjon for å fylle NaN-verdier og opprettet nye trendvariabler ved hjelp av lineær regresjon.

**GridSearch:** Dette ble brukt for hyperparametertuning. Dette kan slås av og på avhengig av brukerens behov.

**Andre viktige verktøy og parametere:**

**Scaler (object):** Scalers brukes til å normalisere eller standardisere numeriske funksjoner. Vanlige scalers inkluderer StandardScaler og RobustScaler. Normalisering hjelper til med å akselerere konvergensen av treningsalgoritmen ved å sikre at alle funksjoner bidrar like mye til læringsprosessen.

**epochs_number (int):** Antallet epoker bestemmer hvor mange ganger læringsalgoritmen skal jobbe gjennom hele treningsdatasettet. Flere epoker kan føre til bedre læring, men kan også føre til overtilpasning hvis for mange.

**batch_size (int):** Dette definerer antall prøver som vil bli propagert gjennom nettverket samtidig. Mindre batchstørrelser kan føre til mer pålitelige oppdateringer, men er mer krevende å beregne. Jeg valgte en middels størrelse basert på formen på dataene og hvor ofte visse funksjoner vises i df. Hastighet var også en vurdering.

**Early Stopping:** Jeg bruker tidlige stoppteknikker for å forhindre overtilpasning og forbedre treningstiden.

**Learning Curves:** Jeg har brukt læringskurver for å avgjøre om modellene er overtilpasset. Resultatene indikerer at dette ikke har skjedd.

#### Spesifikke parametere for nevrale nettverk:

**Alle parametere kan endres basert på resultater og noen ganger som følge av GridSearch (hyperparametertuning).**

**learning_rate (float):** I funksjonen er standard læringsrate satt til 0,001. Læringsraten styrer hvor mye modellens vekter justeres i forhold til tapets gradient. En læringsrate på 0,001 er et vanlig utgangspunkt da det lar modellen konvergere jevnt uten å overskyte den optimale løsningen.

**dropout_rate (float):** Standard dropout-rate er satt til 0,5. Dropout er en regulariseringsteknikk som brukes for å forhindre overtilpasning ved å tilfeldig sette en andel av inngangsenhetene til null ved hver oppdatering under trening. En dropout-rate på 0,5 betyr at halvparten av nevronene blir droppet, noe som er en standardverdi for å fremme robusthet i nettverket.

**neurons_layer1 (int):** Det første laget av det nevrale nettverket har 64 nevroner som standard. Å ha 64 nevroner lar modellen fange komplekse mønstre i dataene samtidig som det opprettholder en balanse mellom beregningseffektivitet og modellkapasitet.

**neurons_layer2 (int):** Det andre laget har 32 nevroner som standard. Dette mindre antallet nevroner i det påfølgende laget bidrar til å redusere modellkompleksiteten gradvis, noe som kan hjelpe med å fange hierarkiske mønstre i dataene.

**activation (str):** Aktiveringsfunksjonen som brukes i de skjulte lagene er relu (Rectified Linear Unit). ReLU-funksjonen er populær fordi den introduserer ikke-linearitet samtidig som den er beregningsmessig effektiv og motvirker problemet med forsvinnende gradienter som er vanlig i dypere nettverk.

**optimizer (str):** Optimalisereren som brukes er adam som standard. Adam (Adaptive Moment Estimation) er en optimaliseringsalgoritme med adaptiv læringsrate som har blitt mye brukt på grunn av sin effektivitet og effektivitet i trening av dype nevrale nettverk. Den kombinerer fordelene med to andre utvidelser av stokastisk gradientavstigning, nemlig AdaGrad og RMSProp, for å gi raskere konvergens.

**Ytterligere detaljer om modellbyggingsprosessen**

**Lagsammensetning:**

Det første tette laget med 64 nevroner bruker relu-aktivering, som er ideell for å fange komplekse ikke-lineære sammenhenger.
Et dropout-lag følger for å forhindre overtilpasning ved tilfeldig å droppe 50 % av nevronene under trening.
Det andre tette laget med 32 nevroner bruker også relu-aktivering, som hjelper til med å finjustere funksjonene som er trukket ut av det første laget.
Et annet dropout-lag legges til etter det andre tette laget for ytterligere regularisering.
Det siste utgangslaget har en enkelt nevron med en lineær aktiveringsfunksjon, som er passende for regresjonsoppgaver da det gir en kontinuerlig verdi.
Regularisering:

kernel_regularizer=tf.keras.regularizers.l2(0.01) brukes på de tette lagene. L2-regularisering bidrar til å forhindre overtilpasning ved å straffe store vekter, og dermed fremme mindre, mer generaliserbare vekter.

**Resultater:**

**XGBoost:**

**Jeg brukte visualiseringsteknikker for å se viktigheten av flere funksjoner.**

<img width="388" alt="XG1" src="https://github.com/user-attachments/assets/5dd15eb7-cd0a-41f9-b81d-4fa9ab640a11">
<img width="395" alt="xg2" src="https://github.com/user-attachments/assets/68290aba-e794-4f60-818c-470d13b78243">
<img width="368" alt="xg3" src="https://github.com/user-attachments/assets/2e10b292-b999-4b14-b465-8e1bb55bb114"><br><br>

**K-Nearest Neighbors:**

<img width="401" alt="NN" src="https://github.com/user-attachments/assets/088dc808-8968-44e7-b2c7-f7a0b47733c8"><br><br>

**Neural Network:**
<img width="446" alt="Neural Networks" src="https://github.com/user-attachments/assets/a34c86cd-704b-4afc-bb80-b9cd8844c085"><br><br>


## DASH APP:

Jeg laget også et dashbord ved hjelp av Dash for å visualisere det endelige produktet. Her er et raskt øyeblikksbilde (det er mer), men i hovedsak er det visualiseringene sett i notatboken, men i dashbordform hvor variabler kan velges og brukes til å oppdatere alle plott samtidig:


<img width="1695" alt="dash 1" src="https://github.com/user-attachments/assets/e70ddcc1-4724-498c-953e-41406d64da42"><br><br>


## Testing av resultatene:

Jeg utfører flere logiske tester og backtester programmets output mot faktiske publiseringer:

<img width="696" alt="Test Results" src="https://github.com/user-attachments/assets/4fe37337-f077-4d51-b59c-ce6d5b6f0648">

<img width="317" alt="Test Results 2" src="https://github.com/user-attachments/assets/a5bcd20d-9d11-44a9-942a-7503e19b5de5"><br><br>

**Basert på disse resultatene er det sannsynlig at jeg vil bruke K-NN nærmeste naboer for 2023-produksjonen. (Men skal teste på enkelt næringer først)**

## Veien videre:

Modeller kan alltid forbedres. Med flere ressurser, spesielt tid, kan det være verdt å undersøke flere andre muligheter, som:

- Å trene modeller for spesifikke industrier. Spesielt hvis disse industriene er særlig unike. For eksempel, for salg av bensin og diesel kan vi prøve å bruke ulike veiinfrastrukturfunksjoner (avstand til nærmeste bensinstasjoner, hvor ofte en vei brukes, etc.):
- Korttransaksjonsdata kan snart bli tilgjengelig, noe som åpner for muligheten for bedre feature engineering - spesielt for detaljhandelsindustrier.
- Det er en mulighet til å identifisere hvilken bransje et selskap kan tilhøre, og som et resultat, identifisere selskaper som for øyeblikket er tildelt feil bransje (nøkkelen som alt aggregeres etter). Nåværende klassifiseringsmodeller presterer dårlig, som vist nedenfor. Men disse bruker bare økonomiske data; jeg forventer at hvis vi bruker funksjoner som stillingstitler (antall ansatte under en gitt stillingstittel), vil modellene prestere bedre.

**Veinettverksdata:**

<img width="445" alt="Roads" src="https://github.com/user-attachments/assets/d30ca253-3720-4a19-bc79-4d125bb1f26b"><br><br>


**Klassifiseringsytelse (så langt)**

<img width="287" alt="Classification 1" src="https://github.com/user-attachments/assets/3a03cb33-0d9d-4148-a89d-8e4af063ee27">
<img width="314" alt="Classification 2" src="https://github.com/user-attachments/assets/e1800742-becd-45b3-b725-01519c6312dc">

