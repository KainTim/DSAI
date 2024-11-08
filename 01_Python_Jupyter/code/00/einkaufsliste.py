import os

class Product:
    def __init__(self, name:str, price:float, menge:int=1):
        self.name = name
        self.price = price
        self.menge = menge

    def __str__(self):
        return f"{self.name} ({self.price} €)"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Product):
            print(f"self.name: {self.name}, value.name: {value.name}, self.menge: {self.menge}, value.menge: {value.menge}, self.price: {self.price}, value.price: {value.price}")
            return self.name == value.name and self.price == value.price
        return False


def produkt_hinzufuegen(einkaufsliste:list[Product]):
    #Get User input
    name = input("Geben Sie den Namen des Produkts ein: ")
    preis = float(input("Geben Sie den Preis des Produkts ein: "))
    menge = int(input("Geben Sie die Menge des Produkts ein: "))
    product = Product(name,preis,menge)
    #Add the product to the list
    added = False
    for productold in einkaufsliste:
        if productold == product:
            einkaufsliste.remove(productold)
            einkaufsliste.append(Product(product.name, product.price, product.menge+productold.menge))
            added = True
    if not added:
        einkaufsliste.append(product)
    print(f"{product.menge}x {name} wurde zur Einkaufsliste hinzugefügt")

def einkaufsliste_entfernen(einkaufsliste:list[Product]):
    #Get User input
    name = input("Geben Sie den Namen des zu entfernenden Produkts ein: ")
    #Remove the product from the list
    for product in einkaufsliste:
        if product.name == name:
            einkaufsliste.remove(product)
            print(f"{name} wurde von der Einkaufsliste entfernt")
            return
    print(f"{name} konnte nicht gefunden werden")

def einkaufsliste_anzeigen(einkaufsliste:list[Product]):
    if not einkaufsliste:
        print("Einkaufsliste ist leer")
        return

    print("Einkaufsliste:")
    print("*"*20)
    for product in einkaufsliste:
        print(f"{product.menge}x {product.name} zu je {product.price} €")

def einkaufsliste_speichern(einkaufsliste:list[Product], dateiname:str):
    with open(dateiname, "w") as file:
        for product in einkaufsliste:
            file.write(f"{product.name};{product.price};{product.menge}\n")
    print("Einkaufsliste wurde gespeichert")

def einkaufsliste_laden(einkaufsliste:list[Product], dateiname:str):
    try:
        with open(dateiname, "r") as file:
            for line in file:
                name, price, menge = line.strip().split(";")
                einkaufsliste.append(Product(name, float(price), int(menge)))
        print(f"Einkaufsliste wurde aus {dateiname} geladen")
    except FileNotFoundError:
        print("Einkaufsliste konnte nicht geladen werden, da die Datei nicht gefunden wurde")
    return einkaufsliste
def einkaufsliste_loeschen(einkaufsliste:list[Product], dateiname:str):
    einkaufsliste.clear()
    if os.path.exists(dateiname):
        os.remove(dateiname)
        print("Datei {dateiname} wurde gelöscht")
    print("Einkaufsliste wurde gelöscht")

def start():
    einkaufsliste = []
    dateiname = "einkaufsliste.txt"

    while True:
        print("Was möchten Sie tun?")
        print("1: Produkt hinzufügen")
        print("2: Produkt entfernen")
        print("3: Einkaufsliste anzeigen")
        print("4: Einkaufsliste speichern")
        print("5: Einkaufsliste laden")
        print("6: Einkaufsliste löschen")
        print("7: Programm beenden")

        auswahl = int(input("Ihre Auswahl: "))
        if auswahl == 1:
            produkt_hinzufuegen(einkaufsliste)
        elif auswahl == 2:
            einkaufsliste_entfernen(einkaufsliste)
        elif auswahl == 3:
            einkaufsliste_anzeigen(einkaufsliste)
        elif auswahl == 4:
            einkaufsliste_speichern(einkaufsliste, dateiname)
        elif auswahl == 5:
            einkaufsliste = einkaufsliste_laden(einkaufsliste, dateiname)
        elif auswahl == 6:
            einkaufsliste_loeschen(einkaufsliste, dateiname)
        elif auswahl == 7:
            break
        else:
            print(auswahl)
            print("Ungültige Eingabe")

if __name__ == "__main__":
    start()