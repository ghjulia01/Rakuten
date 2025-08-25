#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Génère les fréquences de tokens NETTOYÉS 
# (après suppression du HTML, filtrage des stopwords,
# retrait des « mots vagues », et stemming) et construit 
# un translate_map ciblé (EN/DE -> FR)
# à partir du corpus nettoyé.
#
# Utilisation :
# python features/make_cleaned_frequencies_and_map.py `
#  --x_train_csv data/X_train_update.csv `
#  --out_freq features/token_frequencies_cleaned_stem.csv `
#  --out_map features/translate_map_starter_from_cleaned.json `
#  --config features/config.toml
#
# Notes :
# - Dépendances requises : pandas, nltk
# - Si les stopwords NLTK ne sont pas présents, le script tentera de les télécharger.
# - Le stemming utilise Snowball (français) s’il est disponible ; 
# sinon, on n’applique pas de stemming.
try:
    import tomllib  # Py 3.11+
except Exception:
    tomllib = None

import argparse
import json
import re
import string
from collections import Counter

import pandas as pd

def ensure_nltk_stopwords():
    try:
        from nltk.corpus import stopwords  # noqa: F401
    except Exception:
        print("[i] NLTK stopwords not found. Attempting to download ...")
        import nltk
        nltk.download('stopwords')

def load_stopwords_union():
    from nltk.corpus import stopwords
    stop_fr = set(stopwords.words('french'))
    stop_en = set(stopwords.words('english'))
    stop_de = set(stopwords.words('german'))
    return stop_fr.union(stop_en).union(stop_de)

def get_french_stemmer():
    try:
        from nltk.stem.snowball import SnowballStemmer
        return SnowballStemmer('french')
    except Exception:
        return None

MOTS_VAGUES = {
    "lot","vie","magic","set","produit","produits","article","piece","pieces","pièce","pièces",
    "new","die","life","boite","boîte","pack","format","modele","modèle","kit","assortiment",
    "item","tome","import","accessoire","accessoires","ensemble","collection","gamme","serie",
    "série","version","volumes","volume","edition","édition","edition speciale","édition spéciale",
    "edition limitee","édition limitée","serie limitee","série limitée","petit","petite","grand",
    "grande","gros","grosse","mini","maxi","super","ultra","pcs","pc","dernier","derniere",
    "dernière","nouveau","nouvelle","ancien","ancienne","original","originale","noir","noire",
    "blanc","blanche","rouge","bleu","jaune","vert","rose","orange","gris","grise","marron",
    "violet","violette","turquoise","argent","dore","doré","or","cuivre","beige","ivoire",
    "auucne","aucune","aucun","aucuns","aucunes","aucunement","und","magideal","allemand",
    "allemande","deutsch","deutsche","german","germane","germans","japonais","japonaise",
    "japonaises","francais","français","francaise","française","francophones","francophone",
    "anglais","anglaise","english","englishes","complet","complete","completes","jap","japon",
    "sans","integre","intégré","integree","intégrée","integres","intégrés","pvc","plastique",
    "acier","aluminium","rare","commun","communes","neuf","neuve","neuves","neufs","occasion",
    "occasions","occasionnel","occasionnelle","occasionnels","occasionnelles","occasionnellement",
    "generique","générique","generiques","génériques","anti","tout","toute","tous","toutes",
    "stream","design","home","style","mode","fashion","vol","annee","année","annees","années",
    "voir","largeur","longueur","hauteur","largeure","microns","comment","extension","extensions"
}

EN2FR = {
    "black":"noir","white":"blanc","red":"rouge","blue":"bleu","green":"vert","yellow":"jaune",
    "pink":"rose","purple":"violet","brown":"marron","orange":"orange","grey":"gris","gray":"gris","beige":"beige",
    "gold":"or","silver":"argent",
    "size":"taille","sizes":"tailles","small":"petit","medium":"moyen","large":"grand","xl":"xl","xxl":"xxl","xxxl":"xxxl","xs":"xs","xxs":"xxs",
    "cotton":"coton","leather":"cuir","wool":"laine","silk":"soie","linen":"lin","polyester":"polyester","nylon":"nylon","acrylic":"acrylique",
    "spandex":"elasthanne","elastane":"elasthanne","rubber":"caoutchouc","plastic":"plastique","metal":"metal","stainless":"inox",
    "shirt":"chemise","tshirt":"tshirt","tee":"tshirt","sweater":"pull","hoodie":"sweatcapuche","jacket":"veste","coat":"manteau","dress":"robe",
    "skirt":"jupe","pants":"pantalon","trousers":"pantalon","jeans":"jean","shorts":"short",
    "shoes":"chaussures","sneakers":"baskets","boots":"bottes","sandals":"sandales","belt":"ceinture","bag":"sac","handbag":"sac","backpack":"sacados",
    "cap":"casquette","hat":"chapeau","scarf":"echarpe","gloves":"gants","socks":"chaussettes",
    "charger":"chargeur","cable":"cable","adapter":"adaptateur","case":"coque","cover":"housse","screen":"ecran","protector":"protecteur",
    "headphones":"casque","earphones":"ecouteurs","wireless":"sansfil","bluetooth":"bluetooth","usb":"usb","typec":"typec","type-c":"typec",
    "kitchen":"cuisine","bathroom":"salledebain","bedroom":"chambre","living":"sejour","sofa":"canape","cushion":"coussin",
    "blanket":"couverture","towel":"serviette","lamp":"lampe","light":"lumiere","bulb":"ampoule","curtain":"rideau","carpet":"moquette","rug":"tapis",
    "gaming":"jeu","phone":"telephone","tablet":"tablette","laptop":"ordinateur","notebook":"ordinateur","computer":"ordinateur","camera":"appareilphoto",
    "watch":"montre","smartwatch":"montreconnectee","bracelet":"bracelet","necklace":"collier","ring":"bague","earrings":"bouclesdoreilles"
}

DE2FR = {
    "schwarz":"noir","weiss":"blanc","weiß":"blanc","rot":"rouge","blau":"bleu","grun":"vert","gruen":"vert","gelb":"jaune","rosa":"rose","lila":"violet",
    "braun":"marron","orange":"orange","grau":"gris","beige":"beige","gold":"or","silber":"argent",
    "grosse":"taille","groesse":"taille","größe":"taille","klein":"petit","mittel":"moyen","gross":"grand","groß":"grand","grossen":"tailles","groessen":"tailles",
    "baumwolle":"coton","leder":"cuir","wolle":"laine","seide":"soie","leinen":"lin","polyester":"polyester","nylon":"nylon","acryl":"acrylique",
    "spandex":"elasthanne","elastan":"elasthanne","gummi":"caoutchouc","kunststoff":"plastique","metall":"metal","edelstahl":"inox",
    "hemd":"chemise","tshirt":"tshirt","shirt":"chemise","pullover":"pull","kapuzenpullover":"sweatcapuche","jacke":"veste","mantel":"manteau",
    "kleid":"robe","rock":"jupe","hose":"pantalon","jeans":"jean","shorts":"short",
    "schuhe":"chaussures","turnschuhe":"baskets","stiefel":"bottes","sandalen":"sandales","gurtel":"ceinture","guertel":"ceinture",
    "tasche":"sac","handtasche":"sac","rucksack":"sacados","muetze":"bonnet","mütze":"bonnet","hut":"chapeau","schal":"echarpe","handschuhe":"gants","socken":"chaussettes",
    "ladegerat":"chargeur","ladegeraet":"chargeur","kabel":"cable","adapter":"adaptateur","hulle":"coque","huelle":"coque","folie":"film",
    "kopfhorer":"casque","kopfhörer":"casque","drahtlos":"sansfil","bluetooth":"bluetooth","usb":"usb","typc":"typec","typ-c":"typec",
    "kuche":"cuisine","kueche":"cuisine","badezimmer":"salledebain","schlafzimmer":"chambre","wohnzimmer":"salon","kissen":"coussin",
    "decke":"couverture","handtuch":"serviette","lampe":"lampe","licht":"lumiere","birne":"ampoule","vorhang":"rideau","teppich":"tapis"
}

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]*>", " ", text)

def normalize_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def looks_non_french(token: str) -> bool:
    # Heuristics for EN/DE candidates
    if any(ch.isdigit() for ch in token) or len(token) <= 2:
        return False
    if re.search(r"[äöüß]", token):
        return True
    if re.search(r"(ing|ed|ers|able|less|hood|ware|board|phone|case|cover|pack|set)$", token):
        return True
    if re.fullmatch(r"[a-z]+", token):
        if any(pat in token for pat in ["sch","zimmer","grosse","groesse","weiss","druck"]):
            return True
        if token in {"wireless","charger","cable","adapter","screen","protector","gaming","laptop","notebook","smartwatch","bluetooth"}:
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_train_csv", required=True)
    ap.add_argument("--out_freq", required=True)
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--max_new", type=int, default=None)   # ← None pour laisser le TOML décider
    ap.add_argument("--top_k", type=int, default=None)     # ← None pour laisser le TOML décider
    ap.add_argument("--config", type=str, default=None)    # ← chemin vers config.toml
    args = ap.parse_args()

# Charger le TOML si demandé et si tomllib dispo
    if args.config and tomllib is not None:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)
        vb = (cfg.get("text", {}).get("vocab_build", {}) if cfg else {})
        if args.top_k is None:
            args.top_k = int(vb.get("top_k", 8000))
        if args.max_new is None:
            args.max_new = int(vb.get("max_new", 200))
    else:
        # valeurs par défaut si pas de TOML ni d’arguments
        if args.top_k is None:   args.top_k = 8000
        if args.max_new is None: args.max_new = 200

    print(f"[cfg] top_k={args.top_k} | max_new={args.max_new}")
    
    ensure_nltk_stopwords()
    stop_all = load_stopwords_union()
    stemmer = get_french_stemmer()

    df = pd.read_csv(args.x_train_csv, index_col=0)
    texts = (
        df.get("designation", "").astype(str).fillna("")
        + " "
        + df.get("description", "").astype(str).fillna("")
    ).map(normalize_basic)

    counter = Counter()
    for line in texts:
        tokens = [t for t in line.split() if len(t) > 2 and not t.isdigit()]
        # remove stopwords and "mots vagues"
        tokens = [t for t in tokens if t not in stop_all and t not in MOTS_VAGUES]
        # optional stemming (FR)
        if stemmer is not None:
            tokens = [stemmer.stem(t) for t in tokens]
        counter.update(tokens)

    freq_df = pd.DataFrame(counter.most_common(), columns=["token","count"])
    freq_df.to_csv(args.out_freq, index=False)
    print(f"[ok] Wrote frequencies: {args.out_freq}")

    # Build map
    token_set = set(freq_df["token"].tolist())

    translate_map = {}
    for k, v in EN2FR.items():
        if k in token_set:
            translate_map[k] = v
    for k, v in DE2FR.items():
        if k in token_set:
            translate_map[k] = v

    added = []
    for token, _ in counter.most_common(args.top_k):
        if token in translate_map:
            continue
        if looks_non_french(token):
            translate_map[token] = token  # placeholder
            added.append(token)
            if len(added) >= args.max_new:
                break

    with open(args.out_map, "w", encoding="utf-8") as f:
        json.dump([{"token": k, "translation": v} for k, v in sorted(translate_map.items())],
                  f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote translate_map with {len(translate_map)} entries: {args.out_map}")
    if added:
        print("[i] Placeholder examples:", added[:20])

if __name__ == "__main__":
    main()
