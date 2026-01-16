import torch
import torch.quantization
import doanet_parameters
import doanet_model
import cls_feature_class

MODEL_PATH = "model/1_New_Data_foa_dev_split1_model.h5"
AUDIO_PATH = "sound_file/Recording.wav"
device = torch.device("cpu")
params = doanet_parameters.get_params('1')
in_channels = 7
data_in = (params['batch_size'], in_channels, params['feature_sequence_length'], params['nb_mel_bins'])
data_out = [params['batch_size'], params['label_sequence_length'], 6]
model = doanet_model.CRNN(data_in, data_out, params)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8
)
feature_extractor = cls_feature_class.FeatureClass(params)
features = feature_extractor.get_features_for_file(AUDIO_PATH)
time_steps = features.shape[0]
num_channels = 7
mel_bins = params['nb_mel_bins']
features_tensor = torch.from_numpy(features.reshape(time_steps, num_channels, mel_bins)).unsqueeze(0)
features_tensor = features_tensor.permute(0, 2, 1, 3).float().to(device)

with torch.no_grad():
        output, _ = quantized_model(features_tensor)

output = output.squeeze(0)

s1_locations = []
s2_locations = []
for frame_output in output:
    s1_coords = frame_output[0:3]
    s2_coords = frame_output[3:6]
    if torch.sqrt(torch.sum(s1_coords**2)) > 0.2:
        s1_locations.append(s1_coords)
    if torch.sqrt(torch.sum(s2_coords**2)) > 0.2:
        s2_locations.append(s2_coords)

if s1_locations:
    s1_average = torch.mean(torch.stack(s1_locations), dim=0)
    print(f"Source 1: X={s1_average[0]:.2f}, Y={s1_average[1]:.2f}, Z={s1_average[2]:.2f}")
else:
    print("Source 1: Not detected.")

if s2_locations:
    s2_average = torch.mean(torch.stack(s2_locations), dim=0)
    print(f"Source 2: X={s2_average[0]:.2f}, Y={s2_average[1]:.2f}, Z={s2_average[2]:.2f}")
else:
    print("Source 2: Not detected.")