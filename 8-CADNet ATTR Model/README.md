# Cadnet Inference Pipeline

This repository provides a minimal implementation of the inference pipeline for the Cadnet (ATTRACTnet) model, designed to identify patients at risk for transthyretin amyloid cardiomyopathy (ATTR-CM) using ECG and echocardiographic data.

This code is intended for peer review purposes only and is not suitable for clinical use.

## Repository Structure
```
8-CADNet ATTR Model/
├── run_cadnet_inference.py         # Main script to run inference
├── utils.py                        # Preprocessing and utility functions
├── model.py                        # Model definition: ResNet1d with tabular input
├── sample_data/
│   └── example_input.npy           # Example input data
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

# Feature Specification: `cadnet_model_features`

The model expects a structured input table with the following features, derived from ECG and echocardiographic studies:

## Identifiers
- `ecg_key`: Unique ECG identifier
- `echo_key`: Unique echo identifier
- `patient_key`: Patient identifier

## Demographics
- `sex`: Biological sex (`male` or `female`)
- `birth_date`: Date of birth
- `age_at_ecg`: Age at time of ECG

## Timestamps
- `acquisition_datetime`: ECG acquisition time
- `study_datetime`: Echo study time
- `partition_datetime`: Max of ECG or echo datetime

## ECG Features
- `ventricular_pacing_flag`
- `poor_data_quality_flag`
- `ventricular_rate`
- `atrial_rate`
- `pr_interval`
- `qrs_duration`
- `qt_corrected`

## Echo Features
- `lvef_value`: Left ventricular ejection fraction
- `ivs_measurement`: Interventricular septum thickness
- `lvpw_measurement`: Left ventricular posterior wall thickness
- `lv_d_measurement`: LV diastolic dimension
- `max_lv_wall_thickness`: Max of IVS or LVPW

## Waveform Path
- `processed_ecg_blob_path`: Path to preprocessed ECG waveform blob

## Comorbidity Flags
- `carpal_tunnel_syndrome`
- `lumbar_spine_stenosis`
- `degenerative_joint_disease`
