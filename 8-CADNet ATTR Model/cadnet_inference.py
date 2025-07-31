from model import generate_pred
from utils import preprocess_cadnet


def run_cadnet_inference(config, ecg_echo_pairs):
    ecg_tabular_transformer_path = (
        f"{config.model_asset_dir}/ecg_preprocessing_pipeline.joblib"
    )
    echo_tabular_transformer_path = (
        f"{config.model_asset_dir}/echo_preprocessing_pipeline.joblib"
    )
    waveform_params_path = (
        f"{config.model_asset_dir}/ecg_waveform_preprocessing_params.json"
    )
    checkpoint = f"{config.model_asset_dir}/weights.pt"

    tabular, waveforms = preprocess_cadnet(
        ecg_echo_pairs,
        ecg_tabular_transformer_path,
        echo_tabular_transformer_path,
        waveform_params_path,
    )

    preds = generate_pred(
        config.model_version.split(":")[0],
        config.model_params,
        checkpoint,
        waveforms,
        tabular,
        batch_size=config.batch_size,
    )
    results = ecg_echo_pairs[["ecg_key", "echo_key"]].copy()
    results["prediction"] = preds

    return results
