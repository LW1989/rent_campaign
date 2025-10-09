# Output Comparison: Old vs New Pipeline Run

**Date:** 2025-09-30  
**Comparison:** output/squares/ vs output/auswahl_theo_out/squares/

---

## 📊 Dataset Differences

### Old Output (`output/squares/`)
- **Source:** `data/all_stimmbezirke/` 
- **Level:** Individual voting districts (Stimmbezirke)
- **Files:** 402 districts
- **Example:** `Aachen_87_htwk_hochburg` = Voting district 3105 in Aachen
- **Naming:** Pipe-separated format: `"BTW 2017 | Stimmbezirk 3105 | Linke 14.35 %"`

### New Output (`output/auswahl_theo_out/squares/`)
- **Source:** `data/auswahl_theo_gesamt/`
- **Level:** Entire cities/municipalities
- **Files:** 1,411 cities
- **Example:** `Aachen_heating_shape` = All of Aachen city
- **Naming:** Numeric format: `"16.0"` (handled by new `extract_district_name()` function)

---

## ✅ Structure Comparison (Identical!)

### Columns
Both outputs have **IDENTICAL column structure**:
```
GITTER_ID_100m, central_heating_flag, fossil_heating_flag, fernwaerme_flag,
heating_pie, energy_pie, AnteilUeber65, AnteilAuslaender, 
durchschnFlaechejeBew, Einwohner, durchschnMieteQM, wucher_miete_flag,
district_name, metric_cards, conversation_start, name, _umap_options, geometry
```

### Data Quality
| Feature | Old (Voting Districts) | New (Cities) | Status |
|---------|----------------------|--------------|--------|
| Metric Cards | ✅ Present | ✅ Present | ✅ Identical |
| Conversation Starters | ✅ Present | ✅ Present | ✅ Identical |
| Heating Flags | ✅ Working | ✅ Working | ✅ Identical |
| Wucher Detection | ✅ Working | ✅ Working | ✅ Identical |
| uMap Options | ✅ Present | ✅ Present | ✅ Identical |

---

## 🔍 Example: Aachen Comparison

### Old: Aachen Voting District (Stimmbezirk 3105)
```
Squares: 69
District Name: "Stimmbezirk 3105"
Central Heating: 29.0%
Fossil Heating: 94.2%
Fernwärme: 10.1%
Wucher Miete: 0.0%
Avg Rent: 8.11 EUR/sqm
Conversation Starter: "Hallo, ich bin von der Linken. Wie geht es Ihnen..."
```

### New: Aachen City (Entire Municipality)
```
Squares: 796
District Name: "16.0"
Central Heating: 37.8%
Fossil Heating: 85.4%
Fernwärme: 19.5%
Wucher Miete: 10.9%
Avg Rent: 7.96 EUR/sqm
Conversation Starter: "Manche Nachbar*innen berichten, dass sie mit der..."
```

---

## 🎯 Key Findings

### ✅ What's Working Perfectly
1. **Column structure** - 100% identical
2. **Metric cards** - Properly calculated with group statistics
3. **Conversation starters** - Context-appropriate text generation
4. **Heating/energy flags** - Correctly identified based on thresholds
5. **Wucher detection** - Outlier detection functioning
6. **uMap styling** - Color-coded based on flags
7. **Name parsing** - New `extract_district_name()` handles both formats:
   - Pipe-separated: `"BTW 2017 | Stimmbezirk 3105 | Linke 14.35 %" → "Stimmbezirk 3105"`
   - Numeric: `"16.0" → "16.0"`

### 📋 Differences (Expected)
- **Geographic scope** - Districts vs. entire cities
- **Coverage** - 402 vs. 1,411 areas
- **Statistics** - Different because analyzing different geographic units

---

## 🚀 Performance

### New Pipeline Run
- **Cities processed:** 1,411
- **Runtime:** ~12 minutes 24 seconds (with `--skip-addresses`)
- **Status:** ✅ All cities processed successfully
- **Errors:** 0

---

## ✅ Conclusion

**The pipeline output structure is IDENTICAL between old and new runs.**  
The only differences are expected variations due to:
1. Different input data sources (voting districts vs. cities)
2. Different geographic granularity
3. Different naming conventions (both now supported)

**No regressions detected** - all functionality working as expected! 🎉

