# path
data_path = './data/LJSpeech-1.1'
checkpoint_path = './checkpoint'
sample_path = './samples'

# Audio
num_mels = 80
n_mels = 80 # Number of Mel banks to generate
sr = 22050
n_fft = 2048
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr*frame_shift)  # samples.
win_length = int(sr*frame_length)  # samples.
preemphasis = 0.97
fmin = 70
fmax = 8000
max_db = 100
ref_db = 20

# num_freq = 1024
# frame_length_ms = 50.
# frame_shift_ms = 12.5
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20

# encoder & decoder
embedding_size = 256
hidden_size = 256

n_encoder_layers = 4
n_encoder_attention_heads = 2
encoder_conv1d_kernel = 9
encoder_conv1d_filter_size = 1024

n_decoder_layers = 4
n_decoder_heads = 2
decoder_conv1d_kernel = 9
decoder_conv1d_filter_size = 1024


n_iter = 60
# power = 1.5
outputs_per_step = 1

# train_config
batch_size = 48
epochs = 10000
lr = 0.001
save_step = 2000
image_step = 500

cleaners='english_cleaners'

