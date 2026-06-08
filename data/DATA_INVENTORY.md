# Data Inventory for Song et al. (submitted) — RFR-only Pipeline

All URLs/DOIs are extracted from the paper's **Data Availability Statement**.

---

## Primary: Author's Zenodo Data Releases

| Dataset | DOI / URL | Contents |
|---------|-----------|----------|
| **Song 2025 (main data)** | https://doi.org/10.5281/zenodo.17508960 | Full RFR pipeline data archive |
| **Song 2025 (RFR nitrate)** | https://doi.org/10.5281/zenodo.14510704 | SOGOS RFR nitrate estimates on gliders |
| **Balwada 2023 (gliders)** | https://doi.org/10.5281/zenodo.8361656 | Processed Seaglider datasets |

---

## 1. BGC-Argo Float Data (Training + Test)

| Dataset | Source | URL / DOI |
|---------|--------|-----------|
| Argo GDAC (all floats) | International Argo Program | https://doi.org/10.17882/42182 |
| Argo info | UCSD | https://argo.ucsd.edu |
| Argo operations | JCOMMOPS | https://www.ocean-ops.org |
| SOGOS float (WMO 5906030) | Part of Argo GDAC | — independent test |
| 9 training BGC-Argo floats | Part of Argo GDAC | Antarctic Southern Zone, 5°E–65°E |

---

## 2. GO-SHIP Ship Data

| Dataset | Cruise ID | CCHDO URL |
|---------|-----------|-----------|
| **I06 (2019)** — training | 325020190403 | https://cchdo.ucsd.edu/cruise/325020190403 |
| **I07 (2019)** — independent test | 49NZ20191229 | https://cchdo.ucsd.edu/cruise/49NZ20191229 |
| GO-SHIP program | — | http://www.go-ship.org/ |

---

## 3. Seaglider Data (SG659, SG660) — Application

| Dataset | Platform | NOAA NCEI URL |
|---------|----------|---------------|
| SG659 | Seaglider | https://www.ncei.noaa.gov/archive/accession/0228185 |
| SG660 | Seaglider | https://www.ncei.noaa.gov/archive/accession/0228187 |

Also available via Balwada 2023 Zenodo (DOI 10.5281/zenodo.8361656).

---

## 4. Satellite Data (Analysis only — not needed for core RFR)

### 4.1 CMEMS DUACS Altimetry
| Product | DOI |
|---------|-----|
| SEALEVEL_GLO_PHY_L4_MY_008_047 | https://doi.org/10.48670/moi-00148 |

### 4.2 MODIS PAR
| Product | URL |
|---------|-----|
| MODIS Aqua PAR 8-day, 4km | https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1par08day.html |

### 4.3 FSLE
| Product | DOI |
|---------|-----|
| FSLE (LOCEAN/CLS/CTOH/CNES 2021) | https://doi.org/10.24400/527896/a01-2022.002 |

---

## 5. Synthetic Floats (E3SMv2.1 / LIGHT-bgcArgo-1.0)

Provided by Cara Nissen (acknowledged in paper). Not yet on a public repository.

---

## 5. E3SMv2.1 Synthetic Float Data
| Dataset | Source | URL |
|---------|--------|-----|
| LIGHT-bgcArgo-1.0 | E3SM project / ESGF / Zenodo | https://github.com/E3SM-Project/ |

May be available via ESGF nodes or Zenodo repository linked from E3SM publications.

---

## Recommended Download Strategy
1. **Argo floats**: Use `argopy` (easiest, handles QC)
2. **GO-SHIP**: Direct HTTP from CCHDO
3. **Seaglider**: AODN API or IMOS THREDDS
4. **Satellite (CMEMS)**: `copernicusmarine` package
5. **MODIS PAR**: Direct NASA OB.DAAC HTTP
6. **FSLE**: AVISO FTP (requires registration)
7. **E3SM**: Check Zenodo/ESGF
