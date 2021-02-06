# Elektrotehnički fakultet, Univerzitet u Sarajevu
# Predmet: Optimizacija resursa, 2020/2021

Projekat obuhvata nekoliko algoritama za optimizaciju, tj. traženje minimuma funkcije jedne ili više promjenljivih:
1. Newton Rapsonov algoritam
Newton-Raphsonov algoritam predstavlja algoritam za traženje lokalnog ekstrema funkcije f(x) korištenjem rekurzivne relacije:
x2 = x1 − f'(x1)/f''(x1)
Uslov zaustavljanja algoritma je dostizanje maksimalnog broja iteracija ili kada bude ispunjeno |f(x2) − f(x1)| < epsilon, nakon čega se usvaja da je x2 rješenje tj. tačka lokalnog ekstrema.

2. Lokalno pretraživanje
Jednostavno lokalno pretraživanje je heuristički algoritam koji pretražuje okolinu tekućeg rješenja, a kao kandidat za novo tekuće rješenje postaju samo tačke koje su bolje od trenutnog rješenja, te se bira najbolja tačka. Pretraživanje se nastavlja dok se ne ispuni neki od uslova zaustavljanja.
U cilju poboljšanja performansi implementirano je ponavljano lokalno pretraživanje. Algoritam za početnu tačku izvrši prethodno opisano lokano pretraživanje, te se određeni broj puta izvrši preturbacija pronađenog rješenja kako bi se generisala nova tačka od koje će se ponoviti lokalno pretraživanje.
U obje verzije algoritma kao uslov zaustavljanja je korišteno dostizanje maksimalnog broja iteracija ili minimalne promjene vrijednosti funkcije u odnosu na prethodnu iteraciju.

3. Tabu pretraživanje 
Kako bi se izbjegao osnovni nedostatak lokalnog pretraživanja, a to je zapadanje u lokalne ekstreme, tabu algoritam omogućava prelazak u tačke iz okoline koje nisu bolje od trenutnog rješenja, ali će u nastavku voditi do globalnog optimuma. Koristi se tabu lista u kojoj se pohranjuju potencijalna rješenja koja su trenutno zabranjena, tj. tačke od kojih nije moguće nastaviti pretragu.
Uslov zaustavljanja je dostizanje maksimalnog broja iteracija. 

4. Simulirano hlađenje
Simulirano hlađenje predstavlja metaheuristički algoritam lokalnog pretraživanja, koji je često u stanju izbjeći zapadanje u lokalne ekstreme. Naziv algoritma dolazi od analogije sa procesom zagrijavanja i hlađenja čvrstih materijala, tako da se algoritam oslanja na temperaturu koja predstavlja parametar algoritma, a na osnovu kojeg se računaju vjerovatnoće prihvatanja svakog potencijalnog rješenja. 
Uslovi zaustavljanja su dostizanje maksimalnog broja iteracija ili kada vrijednost temperature, koja se kroz algoritam postepeno smanjuje, dostigne vrijednost 0.

5. Evolucione strategije
Kreirana je klasa koja u nekoliko metoda implementira ES(1+1) algoritam, koji se smatra najjednostavnijim iz skupine evolucionih algoritama.
Metoda mutate na samom početku kreira vektor kolonu Z sa 2 elementa dobijena koristeći funkciju random.gauss sa parametrom slučajnosti sigma, koji predstavlja atribut klase, te kreira novi potomak.
Metoda step vrši jednu iteraciju algoritma tako što poziva prethodno objašnjenu metodu, te provjerava da li funkcija ima bolju vrijednost u tački koja predstavlja potomka ili roditelja. Ako tačka potomka daje bolju vrijednost funkcije, onda se to smatra uspješnom mutacijom.
Metoda run izvršava algoritam onoliko puta koliki je maksimalan broj iteracija te pri svakoj iteraciji izračunava parametar uspješnosti generisanja novih jedinki i ažurira matricu standardnih devijacija na osnovu pravila 1/5.

6. Optimizacija rojem čestica
Algoritam optimizacije rojem čestica u svojoj osnovnoj varijanti definira skup čestica koje se kreću kroz problemski prostor kao materijalne čestice koje razmjenjuju informacije o najboljoj vrijednosti kriterija.
Algoritma na samom početku kreira početnu populaciju čestica i njihove brzine, a inicijaliziraju se slučajno odabranim vrijednostima iz opsega. U nastavku se za svaku iteraciju traži lokacija čestice u kojoj funkcija kriterija ima najbolju vrijednost, te kada se takva pronađe ona postaje novo tekuće rješenje, za koje se ažuriraju nova brzina i pozicija. Postupak se ponavlja sve dok se ne ispuni neki od uslova zaustavljanja algoritma, a to je dostizanje maksimalnog broja iteracija ili minimalna promjena vrijednosti funkcije.
