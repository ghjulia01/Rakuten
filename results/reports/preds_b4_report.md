# Rapport — preds_b4
- **F1-macro**: 0.5394
- **F1-weighted**: 0.5913

## Top confusions
| vrai | prédit | count |
|:----|:-------|------:|
| Jouets enfants & bébés (1280) | Drones et modèles réduits (1300) | 23 |
| Mobilier et accessoires de jardin (2582) | Mobilier & articles de maison (1560) | 23 |
| Mobilier & articles de maison (1560) | Décoration & accessoires saisonniers (2060) | 23 |
| Jeux et loisirs enfants (1281) | Jouets enfants & bébés (1280) | 19 |
| Magazines & journaux anciens (2280) | Livres, mangas & partitions (2403) | 19 |
| Livres et ouvrages culturels (10) | Magazines & journaux anciens (2280) | 18 |
| Jouets divers & loisirs créatifs (1302) | Jouets enfants & bébés (1280) | 16 |
| Outils et équipements de jardinage (2585) | Accessoires pour piscines et spas (2583) | 16 |
| Livres, mangas & partitions (2403) | Magazines & journaux anciens (2280) | 14 |
| Décoration & accessoires saisonniers (2060) | Mobilier & articles de maison (1560) | 14 |
| Puériculture & équipement bébé (1320) | Mobilier & articles de maison (1560) | 13 |
| Linge de maison & décoration textile (1920) | Décoration & accessoires saisonniers (2060) | 13 |
| Jouets divers & loisirs créatifs (1302) | Mobilier & articles de maison (1560) | 12 |
| Linge de maison & décoration textile (1920) | Mobilier & articles de maison (1560) | 12 |
| Jeux vidéo et accessoires (40) | Livres, mangas & partitions (2403) | 11 |
| Livres, mangas & partitions (2403) | Livres et ouvrages culturels (10) | 11 |
| Livres et ouvrages culturels (10) | Livres, mangas & partitions (2403) | 11 |
| Mobilier & articles de maison (1560) | Mobilier et accessoires de jardin (2582) | 11 |
| Drones et modèles réduits (1300) | Jouets enfants & bébés (1280) | 10 |
| Puériculture & équipement bébé (1320) | Jouets enfants & bébés (1280) | 10 |

## Top 30 classes par support
| id | nom | support | precision | recall | f1 |
|---:|:-----|-------:|----------:|-------:|----:|
| 2583 | Accessoires pour piscines et spas | 334 | 0.782 | 0.922 | 0.846 |
| 1560 | Mobilier & articles de maison | 196 | 0.520 | 0.658 | 0.581 |
| 2403 | Livres, mangas & partitions | 188 | 0.562 | 0.601 | 0.581 |
| 1300 | Drones et modèles réduits | 186 | 0.647 | 0.758 | 0.698 |
| 2522 | Fournitures de papeterie | 183 | 0.709 | 0.760 | 0.734 |
| 2060 | Décoration & accessoires saisonniers | 174 | 0.555 | 0.730 | 0.630 |
| 2280 | Magazines & journaux anciens | 168 | 0.667 | 0.690 | 0.678 |
| 1280 | Jouets enfants & bébés | 166 | 0.385 | 0.464 | 0.421 |
| 1920 | Linge de maison & décoration textile | 137 | 0.782 | 0.759 | 0.770 |
| 1160 | Cartes à collectionner | 127 | 0.717 | 0.717 | 0.717 |
| 1320 | Puériculture & équipement bébé | 105 | 0.500 | 0.390 | 0.439 |
| 2705 | Essais & livres d’histoire | 102 | 0.759 | 0.804 | 0.781 |
| 1302 | Jouets divers & loisirs créatifs | 100 | 0.479 | 0.350 | 0.405 |
| 2585 | Outils et équipements de jardinage | 97 | 0.523 | 0.351 | 0.420 |
| 1140 | Figurines Pop & licences geek | 96 | 0.706 | 0.500 | 0.585 |
| 2582 | Mobilier et accessoires de jardin | 95 | 0.421 | 0.421 | 0.421 |
| 10 | Livres et ouvrages culturels | 94 | 0.329 | 0.277 | 0.301 |
| 40 | Jeux vidéo et accessoires | 90 | 0.301 | 0.244 | 0.270 |
| 50 | Accessoires gaming | 71 | 0.548 | 0.479 | 0.511 |
| 1281 | Jeux et loisirs enfants | 65 | 0.194 | 0.092 | 0.125 |
| 2462 | Lots jeux vidéo et consoles | 50 | 0.545 | 0.480 | 0.511 |
| 2220 | Accessoires pour animaux | 37 | 0.692 | 0.243 | 0.360 |
| 1180 | Jeux de figurines & wargames | 35 | 0.400 | 0.171 | 0.240 |
| 2905 | Jeux PC à télécharger & éditions spéciales | 29 | 0.880 | 0.759 | 0.815 |
| 60 | Consoles rétro | 28 | 0.720 | 0.643 | 0.679 |
| 1940 | Alimentation & boissons | 26 | 0.778 | 0.269 | 0.400 |
| 1301 | Chaussettes & accessoires enfants | 21 | 0.846 | 0.524 | 0.647 |

## Synthèse par thématique
| thématique | support | f1 | precision | recall |
|:-----------|--------:|---:|----------:|-------:|
| Maison & jardin | 859 | 0.608 | 0.606 | 0.622 |
| Livres & presse | 552 | 0.585 | 0.579 | 0.593 |
| Jeux & gaming | 526 | 0.541 | 0.602 | 0.499 |
| Jouets & enfance | 457 | 0.407 | 0.481 | 0.364 |
| Modélisme & drones | 186 | 0.698 | 0.647 | 0.758 |
| Fournitures & papeterie | 183 | 0.734 | 0.709 | 0.760 |
| Décoration & saisonnier | 174 | 0.630 | 0.555 | 0.730 |
| Animaux | 37 | 0.360 | 0.692 | 0.243 |
| Alimentation & boissons | 26 | 0.400 | 0.778 | 0.269 |