# StoguAtpazinimas
## Duomenų paruošimas
Duomenys imami iš geoportalo puslapio su maksimaliu priartinimu. Imama tiek ortho, tiek poligonų nuotrauka.

Norint sukurti modifikuotus duomenis stogu tolimesniam apdorojimui config faile reikia nurodyti: **data_creation=1**
Tada paleidus programą bus atidaroma nuotrauka iš geoportalo. Joje reikia užeiti ant norimo namo ir suvesti jo koordinates į terminalą.
Sukuriama modifikuota ortoho nuotrauka ir mask.

## Duomenų apdorojimas

*Mask'e paliekamas tik vienas poligonas - esantis arčiausiai centro.
*Išskaičiuojami taškai poligone. kurie atitinka poligono (namo) kampus.
*Pagal maską sukuriami taškai poligono viduje (atstumas iki krašto reguliuojamas config faile: **margin**.
*SAM2 algoritmui paduodame namo ortho nuotrauką.
*Apdirbame mūsu sukurtą modifikuotą mask'ą pagal SAM2 sugeneruotą Mask'ą. Stumiame keikvieną modifikuoto mask'o poligono sieną į vidų arba į išorę.
 Siekiame jį kuo labiau priartinti SAM2 mask'ui. (kriterijus IoU). Tokiu būdu išsaugome realią namo formą, sienų skaičių, kuris yra sužinomas iš 
 poligonų nuotraukos. 
*Apdirbtą galutinį mask'ą iškerpame ir išsaugome




