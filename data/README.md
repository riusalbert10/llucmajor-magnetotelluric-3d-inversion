# Data

The magnetotelluric field data used in this project are **not included** in this repository.

- **Origin:** field campaign carried out in summer 2002 in the Llucmajor area (Mallorca, Spain).
  It covered 42 stations: 36 measured with ADU06 (AMT) and 6 with MMS03E (MT), with both systems
  used at 4 locations. The data belong to the research group at the Universitat de Barcelona that
  ran the campaign.
- **Format:** one EDI file per station (`mallXX.edi`), converted to the ModEM data format
  (full impedance tensor) for the 3D inversion.

To reproduce the processing, place your EDI files in a local folder (for example `data/edi/`, which
is git-ignored) and update the paths in the scripts under `src/`.

What *is* included in `results/final_model/`:
- the final inverted resistivity model (`.rho`, ModEM format, ln ρ);
- the starting model and covariance file of the final run;
- the inversion log and a table with every inversion parameter;
- the observed vs predicted apparent-resistivity fit for each station (PNG).
